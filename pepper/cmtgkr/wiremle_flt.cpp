#include <gmp.h>
#include <vector>

#include <common/math.h>

#include "cmtgkr_env.h"


//evaluates mult_i polynomial for layer 59+d of the F0 circuit
void
FLT_mul_lvl2(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
   mpz_t ans;
   mpz_init_set(ans, r[0]);

   //odds ps go to p/2 for both in1 and in2
   /*
   uint64 ans = r[0];
   uint64 temp;
   */

   //now make sure p>>1 matches in1 and in2
   check_equal(ans, &r[1], &r[mi], &r[mi+mip1], 0, mip1, prime);

   mpz_set(rop, ans);
   //return ans;

   mpz_clear(ans);
}

//evaluates add_i polynomial for layer 59+d of the F0 circuit
void
FLT_add_lvl2(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
   mpz_t tmp, ans;
   mpz_init(tmp);
   mpz_init(ans);

   //even ps go to nip1-1 for in1 and p/2 for both in2
   one_sub(ans, r[0]);
   /*
   uint64 ans1 = 1+PRIME-r[0];
   uint64 temp;
   */

   //now make sure p>>1 matches in2
   check_equal(ans, &r[1], &r[mi+mip1], 0, mip1, prime);

   //make sure in1 matches nip1-1
   chi(tmp, nip1-1, r+mi, mip1, prime);
   modmult(rop, tmp, ans, prime);
   /*
   ans1 = myModMult(ans1, chi(nip1-1, r+mi, mip1));
   return ans1;
   */

   mpz_clear(tmp);
   mpz_clear(ans);
}

//evaluates the mult_i polynomial for layer 60+d of the F0 circuit
void
FLT_mul_lvl1(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  //all gates p but n-2 have in1=in2=p.
  F2_mult_d(rop, r, mi, mip1, ni, nip1, prime);
  /*
  uint64 ans = F2_mult_d(r, mi, mip1, ni, nip1);
  
  return ans;
  */
}

template<int bit> void
FLT_add(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t ans1;
  mpz_init(ans1);

  if (bit)
  {
    mpz_set_ui(ans1, 0);

    // Check that the last gate is added to the second last gate.
    check_wiring(ans1, r, ni-1, nip1-1, nip1-1, mi, mip1, prime);
  }
  else
  {
    one_sub(ans1, r[0]);

    // Even p's, in1 = p, in2 = last gate
    check_equal(ans1, r, &r[mi], 0, mi, prime);

    // Now check that in2 == nip1-1
    mul_chi(ans1, nip1-1, &r[mi+mip1], mip1, prime);
  }

  mpz_set(rop, ans1);

  mpz_clear(ans1);
}
template void FLT_add<1>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template void FLT_add<0>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

template<int bit> void
FLT_mul(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t ans1, ans2, tmp, tmp2;
  mpz_init(tmp);
  mpz_init(tmp2);
  mpz_init(ans1);
  mpz_init(ans2);

  mpz_set_ui(ans1, 1);

  // First make sure all but least and most significant bits of in1 and in2 match p
  check_equal(ans1, r, &r[mi], &r[mi+mip1], 0, mi - 1, prime);

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

    //subtract contribution of last even gate
    mpz_set_ui(tmp, 0);
    check_wiring(tmp, r, ni-1, nip1-1, nip1-1, mi, mip1, prime);
    mpz_sub(ans1, ans1, tmp);
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

  mpz_add(tmp, ans1, ans2); 

  // Make sure that highest order bit of p is 0.
  one_sub(tmp2, r[mi-1]);
  modmult(rop, tmp, tmp2, prime);

  mpz_clear(tmp);
  mpz_clear(tmp2);
  mpz_clear(ans1);
  mpz_clear(ans2);
}
template void FLT_mul<1>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);
template void FLT_mul<0>(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);


