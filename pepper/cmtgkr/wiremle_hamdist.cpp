#include <gmp.h>

#include <common/math.h>

#include "cmtgkr_env.h"
  
void
hamdist_add_last_two(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t tmp, ans;
  mpz_init(tmp);
  mpz_init(ans);

  // First check that
  //   - p == ni - 1 and in1 == nip1 - 1 OR
  //   - p == ni - 2 and in1 == nip1 - 2
  chi(tmp, ni-1, r, mi, prime);
  mul_chi(tmp, nip1-1, &r[mi], mip1, prime);

  chi(ans, ni-2, r, mi, prime);
  mul_chi(ans, nip1-2, &r[mi], mip1, prime);
  mpz_add(ans, ans, tmp);

  // Now check that in2 == nip1 - 1
  mul_chi(ans, nip1-1, &r[mi+mip1], mip1, prime);

  mpz_set(rop, ans);

  mpz_clear(tmp);
  mpz_clear(ans);
}

void
hamdist_sum_vec(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t tmp, tmp2, temp, ans1;
  mpz_init(tmp);
  mpz_init(tmp2);
  mpz_init(temp);
  mpz_init_set_ui(ans1, 1);

  // Check that p = (any, 0), in1 = (any, 0, 0), in2 = (any, 1, 0)

  // First check that the lower bits are the same.
  check_equal(ans1, r, &r[mi], &r[mi+mip1], 0, mi-1, prime);

  // Now check that the upper bits are valid.
  one_sub(tmp, r[mi-1]);
  mpz_mul(ans1, ans1, tmp);

  one_sub(tmp, r[mi+mip1-2]);
  mpz_mul(ans1, ans1, tmp);

  one_sub(tmp, r[mi+mip1-1]);
  mpz_mul(ans1, ans1, tmp);

  mpz_mul(ans1, ans1, r[mi+2*mip1-2]);

  one_sub(tmp, r[mi+2*mip1-1]);
  mpz_mul(ans1, ans1, tmp);

  mpz_set(rop, ans1);

  mpz_clear(tmp);
  mpz_clear(tmp2);
  mpz_clear(temp);
  mpz_clear(ans1);
}

void
hamdist_reduce(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t ans, tmp;
  mpz_init(tmp);
  mpz_init_set_ui(ans, 1);

  // If top bit of r is 0, then we have a standard reduce

  // Check that p == (in1 >> 1) == (in2 >> 1), ignoring the top bit of p.
  check_equal(ans, r, &r[mi+1], &r[mi+mip1+1], 0,    mi-1, prime);
  check_zero (ans, r, &r[mi+1], &r[mi+mip1+1], mi-1, mi,   prime);

  // Check that in1[0] == 0 and in2[0] == 1
  one_sub(tmp, r[mi]);
  mpz_mul(ans, ans, tmp);
  mpz_mul(ans, ans, r[mi+mip1]);

  mpz_set(rop, ans);

  mpz_clear(ans);
  mpz_clear(tmp);
}

void
hamdist_flt_mul_lvl1(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t ans;
  mpz_init_set_ui(ans, 1);

  // all gates p in1=in2=p.
  check_equal(ans, r, &r[mi], &r[mi+mip1], 0,    mi-1, prime);
  check_zero (ans, r, &r[mi], &r[mi+mip1], mi-1, mi,   prime);

  mpz_set(rop, ans);

  mpz_clear(ans);
}

void
hamdist_flt_mul_lvl2(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
   mpz_t ans;
   mpz_init_set(ans, r[0]);

   //odds ps go to p/2 for both in1 and in2

   //now make sure p>>1 matches in1 and in2
   check_equal(ans, &r[1], &r[mi], &r[mi+mip1], 0,      mip1-1, prime);
   check_zero (ans, &r[1], &r[mi], &r[mi+mip1], mip1-1, mip1,   prime);

   mpz_set(rop, ans);
   //return ans;

   mpz_clear(ans);
}

void
hamdist_flt_add_lvl2(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
   mpz_t tmp, ans;
   mpz_init(tmp);
   mpz_init(ans);

   //even ps go to nip1-1 for in1 and p/2 for both in2
   one_sub(ans, r[0]);

   //now make sure p>>1 matches in1
   check_equal(ans, &r[1], &r[mi], 0,      mip1-1, prime);
   check_zero (ans, &r[1], &r[mi], mip1-1, mip1,   prime);

   //make sure in1 matches nip1-1
   chi(tmp, nip1-1, &r[mi+mip1], mip1, prime);
   modmult(rop, tmp, ans, prime);

   mpz_clear(tmp);
   mpz_clear(ans);
}

template<int bit> void
hamdist_flt_add(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  if (bit)
  {
    mpz_set_ui(rop, 0);
  }
  else
  {
    mpz_t ans1;
    mpz_init_set_ui(ans1, 1);

    // Even p's, in1 = p, in2 = last gate
    check_zero (ans1, r, &r[mi], 0,    1,    prime);
    check_equal(ans1, r, &r[mi], 1,    mi-1, prime);
    check_zero (ans1, r, &r[mi], mi-1, mi,   prime);

    // Now check that in2 == nip1-1
    mul_chi(ans1, nip1-1, &r[mi+mip1], mip1, prime);

    mpz_set(rop, ans1);
    mpz_clear(ans1);
  }
}
template void hamdist_flt_add<0>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template void hamdist_flt_add<1>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

template<int bit> void
hamdist_flt_mul(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t ans1, ans2, tmp, tmp2;
  mpz_init(tmp);
  mpz_init(tmp2);
  mpz_init(ans1);
  mpz_init(ans2);

  mpz_set_ui(ans1, 1);

  // First make sure all but least and most significant bits of in1 and in2 match p
  check_equal(ans1, r, &r[mi], &r[mi+mip1], 1,      mi - 1, prime);
  check_zero (ans1, r, &r[mi], &r[mi+mip1], mi - 1, mi,     prime);

  mpz_set(ans2, ans1);

  if (bit)
  {
    //first handle even p contribution
    //first term in product makes sure p is even
    one_sub(tmp, r[0]);
    mpz_mul(ans1, ans1, tmp);

    //finally check that least significant bit of in1 is 0 and lsb of in2 is 1
    one_sub(tmp, r[mi]);
    mpz_mul(tmp2, tmp, r[mi+mip1]);
    mpz_mul(ans1, ans1, tmp2);
  }
  else
  {
    mpz_set_ui(ans1, 0);
  }

  //now handle odd p contribution
  mpz_mul(ans2, ans2, r[0]);
  //uint64 ans2 = r[0];

  //finally check that least significant bit of in1 and in2 are 1
  mpz_mul(tmp, r[mi], r[mi+mip1]);
  mpz_mul(ans2, ans2, tmp);

  mpz_add(rop, ans1, ans2); 

  mpz_clear(tmp);
  mpz_clear(tmp2);
  mpz_clear(ans1);
  mpz_clear(ans2);
}
template void hamdist_flt_mul<1>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template void hamdist_flt_mul<0>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

template<mle_fn add_ifn> void
hamdist_add_wrap(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t tmp;
  mpz_init(tmp);

  add_ifn(rop, r, mi, mip1, ni, nip1, prime);
  hamdist_add_last_two(tmp, r, mi, mip1, ni, nip1, prime);
  modadd(rop, rop, tmp, prime);
  
  mpz_clear(tmp);
}

template void hamdist_add_wrap<hamdist_sum_vec>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template void hamdist_add_wrap<hamdist_reduce>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template void hamdist_add_wrap< hamdist_flt_add<0> >(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template void hamdist_add_wrap< hamdist_flt_add<1> >(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template void hamdist_add_wrap<hamdist_flt_add_lvl2>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

