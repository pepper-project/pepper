#include <iostream>
#include <math.h>
#include <cstring>
#include <cstdlib>
#include <cassert>

#include <cuda_runtime_api.h>
#include <helper_cuda.h>
#include "mpz_utils.h"


using namespace std;

static inline void devToHost(void *host, void *dev, int size)
{
  //cout << "H: " << host << " | D: " << dev << " | S: " << size << endl;
  checkCudaErrors(cudaMemcpy(host, dev, size, cudaMemcpyDeviceToHost));
}

static inline void hostToDev(void *dev, void *host, int size)
{
  checkCudaErrors(cudaMemcpy(dev, host, size, cudaMemcpyHostToDevice));
}

int align_pow2(int a)
{
    if (a == (a & (1 << (ffs(a) - 1))))
        return a;
    else
        return 1 << ((int)ceil(log2((double)a)));
}

gmpz_array::
gmpz_array() : maxSize(0), numElements(0), elementSize(0), isAllocated(false)
{}

gmpz_array::
~gmpz_array()
{
    if (isAllocated) {
        checkCudaErrors(cudaFree(data));
        free(host_mem_pool);
    }
}

int gmpz_array::
arraySize() const
{
    //cout << "arraySize. ES: " << elementSize << " | NE: " << numElements << endl;
    return sizeof(gmp_limb_t) * elementSize * numElements;
}

void gmpz_array::
alloc(int n_elems, int elem_size, bool free_old)
{
    if (isAllocated && n_elems <= maxSize && elem_size <= elementSize)
      return;

    // Force a max elementSize.
    elem_size = max(elem_size, elementSize);

    if (isAllocated && free_old) {
        checkCudaErrors(cudaFree(data));
        free(host_mem_pool);
    }

    maxSize = n_elems;
    elementSize = align_pow2(elem_size);
    numElements = n_elems; 
    checkCudaErrors(cudaMalloc((void **)&data, arraySize()));
    host_mem_pool = (gmp_limb_t *)malloc(arraySize());
    isAllocated = true;

    //cout << "ALLOC. H: " << host_mem_pool << " | D: " << data << " | S: " << arraySize() << endl;

    setSize(maxSize);
}

void gmpz_array::
alloc(int n_elems, int elem_size)
{
    alloc(n_elems, elem_size, true);
}

void gmpz_array::
resize(int elem_size)
{
    if (isAllocated && (elementSize < elem_size)) {
        gmp_limb_t* old_data = data;
        int old_elem_size = elementSize;

        alloc(numElements, elem_size, false);

        int size = min(old_elem_size, elem_size);

        for (int i = 0; i < numElements; i++) {
            checkCudaErrors(cudaMemcpy(
                  (char*)(old_data + i * old_elem_size),
                  (char*)(data + i * elementSize),
                  sizeof(gmp_limb_t) * size,
                  cudaMemcpyDeviceToDevice));
        }

        checkCudaErrors(cudaFree(old_data));
    }
}

void gmpz_array::
writeToDevice()
{
    hostToDev(data, host_mem_pool, arraySize());
}

void gmpz_array::
readFromDevice()
{
    devToHost(host_mem_pool, data, arraySize());
}

void gmpz_array::
fromMPZ(const mpz_t n, int size, bool write_to_dev)
{
    const int elemSize = n->_mp_size * sizeof(n->_mp_d[0]) / sizeof(gmp_limb_t);
    alloc(size, elemSize);

    size_t count;
    mpz_export(host_mem_pool, &count, -1, sizeof(gmp_limb_t), -1, 0, n);
    memset(host_mem_pool + count, 0, (elementSize - count) * sizeof(gmp_limb_t));

    gmp_limb_t* hostData = host_mem_pool + elementSize;
    for (int i = 0; i < size; i++) {
        memcpy(hostData, host_mem_pool, elementSize * sizeof(gmp_limb_t));
        hostData += elementSize;
    }

    if (write_to_dev)
        writeToDevice();
}

void gmpz_array::
fromMPZArray(const mpz_t *array, int size, int stride, bool write_to_dev)
{
    int max_elem_size = compute_num_limbs(array, size, stride);

    alloc(size, max_elem_size);
    to_gpu_format(host_mem_pool, array, size, elementSize, stride);
    setSize(size);

    if (write_to_dev)
      writeToDevice();
}

void gmpz_array::
toMPZArray(mpz_t *array, int stride, bool read_from_dev)
{
    gmp_limb_t* host_data = host_mem_pool;

    if (read_from_dev)
      readFromDevice();

    for (int i = 0; i < numElements; i++) {
        mpz_import(array[i * stride], elementSize, -1, sizeof(gmp_limb_t), -1, 0, host_data);
        host_data += elementSize;
    }
}

int
compute_num_limbs(const mpz_t *array, int size, int stride)
{
    int num_limbs = 1;
    for (int i = 0; i < size; i++) {
        num_limbs = max(num_limbs, array[stride * i]->_mp_size);
    }
    return num_limbs * sizeof(array[0]->_mp_d[0]) / sizeof(gmp_limb_t);
}

void
to_gpu_format(gmp_limb_t* data, const mpz_t *array, int size, int num_limbs, int stride)
{
    size_t count;
    for (int i = 0; i < size; i++) {
        mpz_export(data, &count, -1, sizeof(gmp_limb_t), -1, 0, array[stride * i]);
        assert((int) count <= num_limbs);
        memset(data + count, 0, (num_limbs - count) * sizeof(gmp_limb_t));
        data += num_limbs;
    }
}

WORD*
to_gpu_format(int arr_size, const mpz_t *n, int num_limbs)
{
  WORD *gpu_n = (WORD*)malloc(num_limbs * sizeof(WORD) * arr_size);
  to_gpu_format(gpu_n, n, arr_size, num_limbs);
  return gpu_n;
}

WORD*
to_gpu_format(const mpz_t n, int num_limbs)
{
  size_t count;
  WORD *gpu_n = (WORD*)malloc(num_limbs * sizeof(WORD));
  mpz_export(gpu_n, &count, -1, sizeof(WORD), -1, 0, n);
  memset(&gpu_n[count], 0, (num_limbs - count) * sizeof(WORD));
  return gpu_n;
}

WORD*
to_gpu_format_ui(const WORD n, int num_limbs)
{
  size_t count;

  WORD *gpu_n = (WORD*)malloc(num_limbs * sizeof(WORD));
  memset(gpu_n, 0, num_limbs * sizeof(WORD));
  gpu_n[0] = n;
  return gpu_n;
}

WORD*
to_dev_memory(const mpz_t n, int num_limbs)
{
  WORD *dev_mem, *host_mem;
  const int array_size = num_limbs * sizeof(WORD);

  host_mem = to_gpu_format(n, num_limbs);
  assert(host_mem);

  checkCudaErrors(cudaMalloc((void**)&dev_mem, array_size));
  hostToDev(dev_mem, host_mem, array_size);
  free(host_mem);

  return dev_mem;
}

