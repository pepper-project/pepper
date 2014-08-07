#ifndef CODE_PEPPER_CMTGKR_CMTGKR_ENV_H_
#define CODE_PEPPER_CMTGKR_CMTGKR_ENV_H_

#include <cstdlib>

#include <crypto/prng.h>

#include <common/poly_utils.h>

#include "circuit/circuit.h"
#include "circuit/circuit_layer.h"

extern Prng prng;

typedef unsigned long long uint64;

void evaluate_beta_z_ULL(mpz_t rop, int mi, const mpz_t* z, const uint64_t k, const mpz_t prime);
void evaluate_beta(mpz_t rop, const mpz_t* z, const mpz_t* r, int n, const mpz_t prime);
void evaluate_V_chis(mpz_t rop,
                  const CircuitLayer& level_i,
                  const MPZVector& chis,
                  const mpz_t prime);

void evaluate_V_i(mpz_t rop,
                  const CircuitLayer& level_i,
                  const mpz_t* r,
                  const mpz_t prime);

void check_level(
    mpz_t rop,
    Circuit& c, int i, mpz_t* z, mpz_t ri, 
    int* com_ct, int* rd_ct, mpz_t* r, mpz_t** poly
);

void zero(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

void check_wiring(mpz_t rop, const mpz_t* r, int out, int in1, int in2, int mi, int mip1, const mpz_t prime);

void check_equal(mpz_t rop, const vector<mpz_t*>& r, int start, int end, const mpz_t prime);
void check_equal(mpz_t rop, const mpz_t* p, const mpz_t* in1, const mpz_t* in2, int start, int end, const mpz_t prime);
void check_equal(mpz_t rop, const mpz_t* p, const mpz_t* in1, int start, int end, const mpz_t prime);

void check_zero(mpz_t rop, const mpz_t* p, const mpz_t* in1, int start, int end, const mpz_t prime);
void check_zero(mpz_t rop, const mpz_t* p, const mpz_t* in1, const mpz_t* in2, int start, int end, const mpz_t prime);
void check_zero(mpz_t rop, const vector<mpz_t*>& r, int start, int end, const mpz_t prime);

/* wiremle_flt.cpp */
void FLT_mul_lvl2(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void FLT_add_lvl2(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void FLT_mul_lvl1(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

template<int bit> void FLT_add(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template<int bit> void FLT_mul(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);


/* wiremle_hamdist.cpp */
void hamdist_add_last_two(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void hamdist_sum_vec(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void hamdist_reduce(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void hamdist_flt_mul_lvl1(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void hamdist_flt_add_lvl2(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void hamdist_flt_mul_lvl2(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

template<int bit> void hamdist_flt_add(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template<int bit> void hamdist_flt_mul(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template<mle_fn add_ifn> void hamdist_add_wrap(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);


/* cmtgkr_env.cpp */
void F2_mult_d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void reduce(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

void F0add_dp1to58pd(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void F0mult_dp1to58pd(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void F0mult_d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

void mat_add_below_63_p3d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void mat_add_63_p3d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void mat_mult_63_p3d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void mat_add_61_p2dp1(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
void mat_add_61_p2d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

void update_a_fast(uint64 num_effective, mpz_t rjm1, mpz_t* vals, mpz_t prime);
void check_first_level(mpz_t rop, Circuit& c, mpz_t* r, mpz_t* zi, int d);

#endif

