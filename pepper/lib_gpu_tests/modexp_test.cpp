#include <cassert>
#include <iostream>
#include <cuda_runtime_api.h>
#include <ctime>
#include <common/measurement.h>
#include "lib_gpu/mpz_utils.h"
#include "lib_gpu/mp_modexp.h"
#include "lib_gpu/mp_modexp_gpu.h"
#include "lib_gpu_tests/test_utils.h"

#define NBITS 1024
#define NBITS_EXP 64
#define ARRY_CHUNK_SIZE 2048 
#define ARRY_TOTAL_SIZE (1024*8)

using namespace std;

static void
montmul_cpu(mpz_t rop, const mpz_t op1, const mpz_t op2,
            const mpz_t n, const mpz_t np, int num_bits)
{
  mpz_t t, m;
  mpz_init_set_ui(t, 0);
  mpz_init_set_ui(m, 0);

  mpz_mul(t, op1, op2);
  mpz_mul(m, t, np);
  mpz_tdiv_r_2exp(m, m, num_bits);

  mpz_mul(rop, m, n);
  mpz_add(rop, rop, t);
  mpz_tdiv_q_2exp(rop, rop, num_bits);

  if (mpz_cmp(rop, n) >= 0)
    mpz_sub(rop, rop, n);

  mpz_clear(t);
  mpz_clear(m);
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
  mpz_t temp, n, ninv, n_square, prime, primep, mpz_one;
  init(temp, NBITS);
  init(n, NBITS);
  init(ninv, NBITS);
  init(n_square, NBITS);
  init(prime, NBITS);
  init(primep, NBITS);
  init(mpz_one, NBITS);

  mpz_t *A = new mpz_t[ARRY_TOTAL_SIZE];
  mpz_t *E = new mpz_t[ARRY_TOTAL_SIZE];
  mpz_t *C = new mpz_t[ARRY_TOTAL_SIZE];
  mpz_t *gpuC = new mpz_t[ARRY_TOTAL_SIZE];
  gmpz_array g_A, g_E, g_C;

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
  g_A.fromMPZArray(A, ARRY_TOTAL_SIZE);
 
  rand_init(E, ARRY_TOTAL_SIZE, NBITS_EXP);
  mod(E, prime, ARRY_TOTAL_SIZE);

  // Special case the exponent 0
  mpz_set_ui(E[0], 0);
  g_E.fromMPZArray(E, ARRY_TOTAL_SIZE);

  init(C, ARRY_TOTAL_SIZE, NBITS);
  init(gpuC, ARRY_TOTAL_SIZE, NBITS);
  g_C.alloc(ARRY_TOTAL_SIZE, 1);

  g_C.fromMPZArray(C, ARRY_TOTAL_SIZE);
  g_C.toMPZArray(gpuC);

  cout << "Test Memcopy function...";
  RUN_TEST(cmp(C, gpuC, ARRY_TOTAL_SIZE));

  int num_limbs = (NBITS/8)/ sizeof(WORD);
  int num_limbs_exp = (NBITS_EXP/8)/ sizeof(WORD);

  WORD *c = to_gpu_format_ui(0, num_limbs);
  WORD *one = to_gpu_format_ui(1, num_limbs);
  WORD *n_sq = to_gpu_format(n_square, num_limbs);
  WORD *p = to_gpu_format(prime, num_limbs);
  WORD *pp = to_gpu_format(primep, num_limbs);
  
  // Divide the arrays into multiple chunks and offload each chunk at a time
  Measurement m;
  START_TEST("gpu modexp");
  for (int k=0; k<ARRY_TOTAL_SIZE; k+=ARRY_CHUNK_SIZE) {
    WORD *a = to_gpu_format(ARRY_CHUNK_SIZE, &A[k], num_limbs);
    WORD *e = to_gpu_format(ARRY_CHUNK_SIZE, &E[k], num_limbs_exp);

    // invoke GPU
    if (k == 0) m.begin_with_init();
    else m.begin_with_history();
    mp_many_modexp_mont_gpu(ARRY_CHUNK_SIZE, a, a, e, n_sq, p, pp, num_limbs, num_limbs_exp);
    m.end();
    
    for (int i=0; i<ARRY_CHUNK_SIZE; i++) {
      mpz_set_ui(gpuC[k+i], 0);
      mpz_import(gpuC[k+i], num_limbs, -1, sizeof(WORD), -1, 0, &a[i*num_limbs]);
      mpz_powm(C[k+i], A[k+i], E[k+i], prime);
      if (mpz_cmp(gpuC[k+i], C[k+i])) {
        gmp_printf("FAILED %d, %d!\n", k, i);
        gmp_printf("A   : %Zx\n", A[k+i]);
        gmp_printf("E   : %Zx\n", E[k+i]);
        gmp_printf("C   : %Zx\n", C[k+i]);
        gmp_printf("gpuC: %Zx\n", gpuC[k+i]);
        exit(1);
      }
      else {
      }
    }
    if (a) free(a);
    if (e) free(e);
  }
  PASS();
  cout<<"Time taken per exponentiation = "<<m.get_papi_elapsed_time()/ARRY_TOTAL_SIZE<<" usec"<<endl;
  cout<<"#exps/sec = "<<(ARRY_TOTAL_SIZE*1000.0*1000)/m.get_papi_elapsed_time()<<endl;
  if (c) free(c);
  if (one) free(one);
  if (n_sq) free(n_sq);
  if (p) free(p);
  if (pp) free(pp);
  return 0;
}
