#include <cuda.h>
#include <helper_cuda.h>

#include "common.h"

/**
 * Sum all elements in the array. The result is stored in
 * array[0]. array is assumed to have size equal to blockDim.x,
 * which in turn needs to be a power of 2.
 */
__device__ void device_sum_aligned(int *array)
{
    unsigned int local_i = threadIdx.x;
    unsigned int num_participants = blockDim.x / 2;

    while (num_participants > 0) {
        if (local_i < num_participants)
            array[local_i] += array[local_i + num_participants];

        num_participants >>= 1;
        __syncthreads();
    }
}

/**
 * Performs element-wise multiplication of two vectors.
 * C may be the same array as either A or B.
 */
__global__ void vec_product(int *C, const int *A, const int *B, size_t n)
{
    unsigned int global_i = blockDim.x * blockIdx.x + threadIdx.x;

    if (global_i < n)
        C[global_i] = A[global_i] * B[global_i];
}

/**
 * Partially sum the first n elements of the array and store it
 * in the first ceil(n / blockDim.x) elements of partial_sums.
 * It is *not* safe to have partial_sums == array.
 */
__global__ void partial_sum(int *partial_sums, const int *array, size_t n)
{
    unsigned int local_i = threadIdx.x;
    unsigned int global_i = blockDim.x * blockIdx.x + local_i;

    // Holds the individual multiplies.
    extern __shared__ int subvec_c[];

    if (global_i < n)
        subvec_c[local_i] = array[global_i];
    else
        subvec_c[local_i] = 0;

    __syncthreads();

    device_sum_aligned(subvec_c);
    partial_sums[blockIdx.x] = subvec_c[0];
}

int dot_product(const int *A, const int *B, size_t n)
{
    if (n == 0)
        return 0;

    int result;
    int *d_A, *d_B, *temp;
    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int smem_size = threadsPerBlock * sizeof(int);
    size_t size = n * sizeof(int);

    checkCudaErrors(cudaMalloc(&d_A, size));
    checkCudaErrors(cudaMalloc(&d_B, size));

    checkCudaErrors(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    vec_product<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A, d_B, n);

    while (blocksPerGrid > 1)
    {
        partial_sum<<<blocksPerGrid, threadsPerBlock, smem_size>>>(d_B, d_A, n);

        // Swap partial sums and regular array since we must now sum the
        // partial sums.
        // d_B => partial_sums, d_A => array
        temp = d_A;
        d_A = d_B;
        d_B = temp;

        n = blocksPerGrid;
        blocksPerGrid = (blocksPerGrid + threadsPerBlock - 1) / threadsPerBlock;
    }
    partial_sum<<<blocksPerGrid, threadsPerBlock, smem_size>>>(d_B, d_A, n);

    checkCudaErrors(cudaMemcpy(&result, d_B, sizeof(int), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));

    return result;
}

