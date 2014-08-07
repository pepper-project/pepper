#ifndef GPU_MP_H
#define GPU_MP_H
#include <stdint.h>
#include <gmp.h>
#include "mp_modexp.h"
#include <assert.h>

/*
#define LIMB_SIZE 32

#if LIMB_SIZE == 32
  typedef uint32_t gmp_limb_t;
#elif LIMB_SIZE == 64
  typedef uint64_t gmp_limb_t;
#endif
*/

typedef WORD gmp_limb_t;

#define DIV_ROUNDUP(dividend, divisor) (((dividend) + (divisor) - 1) / (divisor))
#define BITS_TO_LIMBS(bits) DIV_ROUNDUP(bits,  8 * sizeof(WORD))

struct gmpz_array {
  int maxSize;
  int numElements;
  int elementSize; // in limbs
  bool isAllocated;
  gmp_limb_t* data;
  gmp_limb_t* host_mem_pool;

private:
  void alloc(int n_elems, int elem_size, bool free_old);

public:
  gmpz_array();
  ~gmpz_array();
  inline int getLimb(int elem_idx, int limb_idx) {
      return data[elem_idx * elementSize + limb_idx];
  }

  inline gmp_limb_t* getElemHost(int elem_idx) {
      //assert(elem_idx < numElements);
      return &host_mem_pool[elem_idx * elementSize];
  }

  inline void setSize(int size) { numElements = size; }
  int arraySize() const;

  void alloc(int n_elems, int elem_size);
  void resize(int elem_size);

  // Read from/write to the data array and 
  // Write the contents of host mem pool to the data array.
  void writeToDevice();

  // Read the contents of the data array and place it in the host mem pool.
  void readFromDevice();

  void fromMPZ(const mpz_t n, int size, bool write_to_dev = true);
  void fromMPZArray(const mpz_t *array, int size, int stride = 1, bool write_to_dev = true);
  void toMPZArray(mpz_t *array, int stride = 1, bool read_from_dev = true);
};

static inline int
div_roundup(int dividend, int divisor) { return (dividend + divisor - 1) / divisor; }

int align_pow2(int a);

void add(gmpz_array *rop, gmpz_array *op1, gmpz_array *op2);
void mul(gmpz_array *rop, gmpz_array *op1, gmpz_array *op2);

int  compute_num_limbs(const mpz_t *array, int size, int stride = 1);
void to_gpu_format(gmp_limb_t* data, const mpz_t *array, int size, int num_limbs, int stride = 1);

WORD* to_gpu_format(int arr_size, const mpz_t *n, int num_limbs); 
WORD* to_gpu_format(const mpz_t n, int num_limbs);
WORD* to_gpu_format_ui(const WORD n, int num_limbs);
WORD* to_dev_memory(const mpz_t n, int num_limbs);

#endif
