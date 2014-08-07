
#ifndef CODE_PEPPER_COMMON_POLY_UTILS_H_
#define CODE_PEPPER_COMMON_POLY_UTILS_H_

#include <cassert>
#include <gmpxx.h>
#include <stdint.h>

#include "math.h"
#include "mpnvector.h"

void computeChiAll(MPZVector& rop, const MPZVector& r, const mpz_t prime);
void computeChiAll(MPZVector& rop, size_t n, const MPZVector& r, size_t startAt, const mpz_t prime);

void mul_chi(mpz_t rop, const uint64_t v, const mpz_t* r, int n, const mpz_t prime);
void chi(mpz_t rop, const uint64_t v, const mpz_t* r, int n, const mpz_t prime);
void computeChiAll(MPZVector& rop, const MPZVector& r, const mpz_t prime);

void bary_precompute_weights(MPZVector& weights, const mpz_t r, const mpz_t prime);
void bary_precompute_weights3(MPZVector& weights, const mpz_t r, const mpz_t prime);
void bary_extrap(MPZVector& rop, const MPZVector& vec, const MPZVector& weights, const mpz_t prime);

void extrap3(mpz_t rop, const mpz_t* vec, const mpz_t r, const mpz_t prime);
void extrap(mpz_t rop, const mpz_t* vec, const uint64_t n, const mpz_t r, const mpz_t prime);
void extrap_ui(mpz_t rop, const mpz_t* vec, const uint64_t n, const uint64_t r, const mpz_t prime);

template<typename Fn0, typename Fn1> void
computeMLEAll(
    MPZVector& rop, size_t n,
    const MPZVector& r, size_t startAt,
    const mpz_t prime,
    Fn0 fn0, Fn1 fn1)
{
  size_t logn = log2i(n);
  assert(r.size() >= logn);

  mpz_class tmp;

  // Interleave the computations of all the betars.
  // Basically, compute:
  //   1 :  fn0(r1)            fn1(r1)
  //   2 :  fn0(r2) fn0(r1)    fn0(r2) fn1(r1)   fn1(r2) fn0(r1)   fn1(r2) fn1(r1)
  //   etc.
  mpz_set_ui(rop[0], 1);
  for (size_t logi = 0; logi < logn; logi++)
  {
    size_t base = 1 << logi;

    fn1(tmp.get_mpz_t(), r[logi + startAt]);
    if (tmp != 1)
    {
      for (size_t i = base; i < std::min(2 * base, n); i++)
      {
        //modmult(rop[i], rop[i - base], r[logi + startAt], prime);
        modmult(rop[i], rop[i - base], tmp.get_mpz_t(), prime);
      }
    }

    //one_sub(one_sub_r, r[logi + startAt]);
    fn0(tmp.get_mpz_t(), r[logi + startAt]);
    if (tmp != 1)
    {
      for (size_t i = 0; i < base; i++)
      {
        modmult(rop[i], rop[i], tmp.get_mpz_t(), prime);
      }
    }
  }
}


#endif
