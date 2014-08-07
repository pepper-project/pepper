#include <cassert>
#include <iostream>
#include <ctime>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include "lib_gpu/mpz_utils.h"
#include "lib_gpu/mp_modexp.h"
#include "lib_gpu/mp_modexp_gpu.h"
#include "lib_gpu_tests/test_utils.h"

#define NBITS 1024
#define NBITS_EXP 192 
#define ARRY_CHUNK_SIZE 32
#define ARRY_TOTAL_SIZE (32*4)

using namespace std;

static void
montmul_cpu(mpz_t rop, const mpz_t op1, const mpz_t op2,
            const mpz_t n, const mpz_t np, int num_bits)
{
  mpz_t t, m;
  mpz_inits(t, m, 0);

  mpz_mul(t, op1, op2);
  mpz_mul(m, t, np);
  mpz_tdiv_r_2exp(m, m, num_bits);

  mpz_mul(rop, m, n);
  mpz_add(rop, rop, t);
  mpz_tdiv_q_2exp(rop, rop, num_bits);

  if (mpz_cmp(rop, n) >= 0)
    mpz_sub(rop, rop, n);

  mpz_clears(t, m, 0);
}

static void
mod(mpz_t *arry, const mpz_t prime, int size)
{
  for (int i = 0; i < size; i++) {
    mpz_mod(arry[i], arry[i], prime);
  }
}

int main()
{
  mpz_t temp, n, ninv, n_square, prime, primep, mpz_one, output_gpu, output_cpu;
  init(temp, NBITS);
  init(n, NBITS);
  init(ninv, NBITS);
  init(n_square, NBITS);
  init(prime, NBITS);
  init(primep, NBITS);
  init(mpz_one, NBITS);
  init(output_gpu, NBITS);
  init(output_cpu, NBITS);

  mpz_t A[ARRY_TOTAL_SIZE];
  mpz_t E[ARRY_TOTAL_SIZE];
  mpz_t C[ARRY_TOTAL_SIZE];
  mpz_t gpuC[ARRY_TOTAL_SIZE];
  gmpz_array g_A;

  mpz_set_ui(mpz_one, 1);

  // Calculate an NBITS prime
  mpz_set_ui(n, 0);
  mpz_setbit(n, NBITS - 1);
  mpz_nextprime(prime, n);

  // Calculate 2^NBITS
  mpz_setbit(n, NBITS);
  mpz_clrbit(n, NBITS - 1);

  // Invert calculate primep such that n*n^{-1} - prime*primep = 1
  mpz_invert(ninv, n, prime);
  mpz_mul(temp, n, ninv);
  mpz_sub_ui(temp, temp, 1);
  mpz_tdiv_q(primep, temp, prime);

  // Calculate n^2 mod prime
  mpz_mul(n_square, n, n);
  mpz_mod(n_square, n_square, prime);

  rand_init(A, ARRY_TOTAL_SIZE, NBITS);
  mod(A, prime, ARRY_TOTAL_SIZE);
 
  rand_init(E, ARRY_TOTAL_SIZE, NBITS_EXP);
  mod(E, prime, ARRY_TOTAL_SIZE);

  init(C, ARRY_TOTAL_SIZE, NBITS);
  init(gpuC, ARRY_TOTAL_SIZE, NBITS);

  int num_limbs = (NBITS/8)/ sizeof(WORD);
  int num_limbs_exp = (NBITS_EXP/8)/ sizeof(WORD);

  WORD *c = to_gpu_format_ui(0, num_limbs);
  WORD *one = to_gpu_format_ui(1, num_limbs);
  WORD *n_sq = to_gpu_format(n_square, num_limbs);
  WORD *p = to_gpu_format(prime, num_limbs);
  WORD *pp = to_gpu_format(primep, num_limbs);
 
  struct mp_sw sw[ARRY_CHUNK_SIZE];
	struct mp_sw *sw_d;
	WORD *a_d;
  
  WORD *r_sq_d;
	WORD *n_d;
	WORD *np_d;
	WORD *ret_d;

	checkCudaErrors(cudaMalloc(&sw_d, sizeof(struct mp_sw) * ARRY_CHUNK_SIZE));
  checkCudaErrors(cudaMalloc(&a_d, sizeof(WORD) * num_limbs * ARRY_CHUNK_SIZE));
	checkCudaErrors(cudaMalloc(&r_sq_d, sizeof(WORD) * num_limbs));
	checkCudaErrors(cudaMalloc(&n_d, sizeof(WORD) * num_limbs)); 
	checkCudaErrors(cudaMalloc(&np_d, sizeof(WORD) * num_limbs));
	checkCudaErrors(cudaMalloc(&ret_d, sizeof(WORD) * num_limbs * ARRY_CHUNK_SIZE));

 
  // copy the fixed parameters	
  checkCudaErrors(cudaMemcpy(r_sq_d, n_sq, sizeof(WORD) * num_limbs, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(n_d, p, sizeof(WORD) * num_limbs, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(np_d, pp, sizeof(WORD) * num_limbs, cudaMemcpyHostToDevice));

  // Divide the arrays into multiple chunks and offload each chunk at a time
  mpz_set_ui(output_gpu, 1);
  for (int k=0; k<ARRY_TOTAL_SIZE; k+=ARRY_CHUNK_SIZE) {
    WORD *a = to_gpu_format(ARRY_CHUNK_SIZE, &A[k], num_limbs);
    WORD *e = to_gpu_format(ARRY_CHUNK_SIZE, &E[k], num_limbs_exp);
  
    // copy the bases
    //checkCudaErrors(cudaMemcpy(a_d, a, sizeof(WORD) * num_limbs * ARRY_CHUNK_SIZE, cudaMemcpyHostToDevice));
    g_A.fromMPZArray(&A[k], ARRY_CHUNK_SIZE, true);

    // format and copy the exponents
    for (int i=0; i<ARRY_CHUNK_SIZE; i++)
	    mp_get_sw(&sw[i], &e[i*num_limbs_exp], num_limbs_exp);
	  checkCudaErrors(cudaMemcpy(sw_d, sw, sizeof(struct mp_sw) * ARRY_CHUNK_SIZE, cudaMemcpyHostToDevice));

    // invoke GPU
    mp_many_modexp_mont_gpu_nocopy(ARRY_CHUNK_SIZE, ret_d, (WORD *)g_A.data, sw_d, r_sq_d, n_d, np_d, num_limbs);
    
    // copy the result back
	  checkCudaErrors(cudaMemcpy(a, ret_d, sizeof(WORD) * num_limbs * ARRY_CHUNK_SIZE, cudaMemcpyDeviceToHost));
    for (int i=0; i<ARRY_CHUNK_SIZE; i++) {
      mpz_set_ui(gpuC[k+i], 0);
      mpz_import(gpuC[k+i], num_limbs, -1, sizeof(WORD), -1, 0, &a[i*num_limbs]);
      mpz_mul(output_gpu, output_gpu, gpuC[k+i]);
      mpz_mod(output_gpu, output_gpu, prime);
    }
      
    if (a) free(a);
    if (e) free(e);
  }

  mpz_set_ui(output_cpu, 1);
  for (int k=0; k<ARRY_TOTAL_SIZE; k+=1) {
    mpz_powm(A[k], A[k], E[k], prime);
    mpz_mul(output_cpu, output_cpu, A[k]);
    mpz_mod(output_cpu, output_cpu, prime);
  }

  printf("Encrypted dot product .. ");
  if (mpz_cmp(output_cpu, output_gpu)) {
    gmp_printf("FAILED!\n");
  }
  else {
    gmp_printf("PASSED\n");
  }

  if (c) free(c);
  if (one) free(one);
  if (n_sq) free(n_sq);
  if (p) free(p);
  if (pp) free(pp);

  checkCudaErrors(cudaFree(sw_d));
  checkCudaErrors(cudaFree(a_d));
  checkCudaErrors(cudaFree(r_sq_d));
  checkCudaErrors(cudaFree(n_d));
  checkCudaErrors(cudaFree(np_d));
  checkCudaErrors(cudaFree(ret_d));

  return 0;
}
