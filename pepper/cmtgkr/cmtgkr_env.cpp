/*************************************
//Justin Thaler
//January 20, 2011. 
//Implementation of protocol described in 
//"Delegating Computation: Interactive Proofs for Muggles" by Goldwasser, Kalai, and Rothblum.
//This implementation supports a verifier who can efficiently
//evaluate multinear extension of the add_i and mult_i functions at random locations
//rather than doing so offline or via an implicit "streaming pass" over the circuit.
//Currently this implementation supports, F2, F0, pattern matching, and matrix multiplication. 
//This implementation was originally written for the paper "Practical Verified Computation with Streaming
//Interactive Proofs" by Cormode, Mitzenmacher, and Thaler. The matrix multiplication circuit was implemented 
//for the paper "Verifying Computations with Massively Parallel Interactive Proofs" by Thaler,
//Roberts, Mitzenmacher, and Pfister.
**************************************/

#include <cstdlib>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>
#include <gmp.h>

#include <crypto/prng.h>
#include <common/measurement.h>
#include <common/math.h>

#include "cmtgkr_env.h"

static int mark = 0;
#define mark     \
  cout << "mark: " << mark++ << endl

#define out(__i__) \
  cout << __i__ << endl

using namespace std;

static uint64 counter = 0;

/* DEBUG
static u8 key[256] = {};
static u8 iv[64] = {};

static Prng prng(PNG_CHACHA, key, iv);
*/

// Global prng object. Probably shouldn't do this...
Prng prng(PNG_CHACHA);

// Global measurement object. For testing.
static Measurement m;
static Measurement m2;

//computes x^b
uint64 powi(uint64 x, uint64 b) 
{
  uint64 i = 1;
  for (uint64 j = 0; j < b; j++)  i *= x;
  return i;
}

// Calculates log2 of number.  
double Log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( (double) n ) / log( (double) 2 );  
}

//evaluates beta_z polynomial (described in GKR08)
//at location k (the bits of k are interpreted as a boolean vector). mi is dimension of k and r.
void
evaluate_beta_z_ULL(mpz_t rop, int mi, const mpz_t* z, const uint64_t k, const mpz_t prime)
{
    chi(rop, k, z, mi, prime);
}

void
evaluate_beta(mpz_t rop, const mpz_t* z, const mpz_t* r, int n, const mpz_t prime)
{
  mpz_t prod, one_sub_r;
  mpz_init_set_ui(prod, 1);
  mpz_init(one_sub_r);

  for (int i = 0; i < n; i++)
  {
    one_sub(one_sub_r, r[i]);
    mulmle(prod, z[i], one_sub_r, r[i], prime);
  }

  mpz_set(rop, prod);

  mpz_clear(prod);
  mpz_clear(one_sub_r);
}


void
evaluate_V_chis(
        mpz_t rop,
        const CircuitLayer& level,
        const MPZVector& chis,
        const mpz_t prime)
{
  mpz_t ans;
  mpz_init_set_ui(ans, 0);

  for (int i = 0; i < level.size(); i++)
  {
    if (mpz_sgn(level.gate(i).zValue()) != 0)
      mpz_addmul(ans, level.gate(i).zValue(), chis[i]);
  }

  mpz_mod(rop, ans, prime);
  mpz_clear(ans);
}

//evaluates V_i polynomial at location r.
//Here V_i is described in GKR08; it is the multi-linear extension
//of the vector of gate values at level i of the circuit
void
evaluate_V_i(
        mpz_t rop,
        const CircuitLayer& level_i,
        const mpz_t* r,
        const mpz_t prime)
{
    mpz_t op2, ans;
    mpz_init(op2);
    mpz_init_set_ui(ans, 0);
    for (int k = 0; k < level_i.size(); k++)
    {
        chi(op2, k, r, level_i.logSize(), prime);
        addmodmult(ans, level_i.gate(k).zValue(), op2, prime);
    }
    mpz_set(rop, ans);
    mpz_clear(op2);
    mpz_clear(ans);
}

static void
init_vals_array(mpz_t* vals, Circuit& c, int i, int nip1)
{
    int num_effective = powi(2, c[i+1].logSize());
    for(int k = 0; k < num_effective; k++)
    {
       if(k < nip1)
         mpz_set(vals[k], c[i + 1].gate(k).zValue());
       else
         mpz_set_ui(vals[k], 0);
    }
}

static void
generate_randomness(mpz_t* r, int n, mpz_t prime)
{
  for(int j = 0; j < n; j++)
  {
      prng.get_random(r[j], prime);
      // r[j] = rand();
      // DEBUG
      // mpz_set_ui(r[j], counter++);
  }
}

//Run through the GKR protocol at level i.
//Expects all values in circuit to be filled in (circuit has been evaluated).
//This protocol reduces verifying a claim that V_i(z)=ri to verifying that V_i+1(z')=ri+1.
//Upon exit, the z array contain the new value z' for checking the next level of the circuit
//and the function returns the new value ri+1.
//Explanation of parameters:
//c is circuit being checked. z and ri are described above. p should equal 2^61-1.
//com_ct and rd_ct are used for tracking communcation and message costs.
//r is used to store V's random coin tosses during this iteration of the protocol.
//The poly array is used to store P's messages in this iteration of the protocol.
//vals is an array used to store values of the polynomial Vi+1 at various locations
//The last two parameters are functions for evaluating the multilinear extension of add_i and mult_i,
//and are used by V to perform one of her checks on P's messages.
void
check_level(
    mpz_t rop,
    Circuit& c, int i, mpz_t* z, mpz_t ri, 
    int* com_ct, int* rd_ct, mpz_t* r, mpz_t** poly
)
{
  //Although GKR is stated assuming that all levels of the circuit have same number of
  //gates, our implementation does not require this. Thus, ni will be number of gates at level i
  //and nip1 will be number of gates at level i+1; mi and mip1 are ceil(log(ni or nip1)) respectively.
  //This yields significant speedups in practice.

  /*
  int mi    = c[i].logSize();
  int ni    = c[i].size();
  int mip1  = c[i+1].logSize();
  int nip1  = c[i+1].size();
  int nvars = mi+2*mip1; //number of variables being summed over in this iteration of GKR

  // DEBUG printf("i: %d | mi: %d | mip1: %d | nip1: %d | nvars: %d\n", i, mi, mip1, nip1, nvars);

  *com_ct = *com_ct + 3*nvars;

  //r is V's random coin tosses for this iteration.
  generate_randomness(r, nvars, c.prime);

  *rd_ct = *rd_ct + mi+2*mip1;

  // DEBUG gmp_printf("i: %d | ri: %Zd\n", i, ri);

  mpz_t V1, V2, tmp;
  mpz_init(V1);
  mpz_init(V2);
  mpz_init(tmp);

  m.begin_with_history();
  CMTSumCheckProver scp(c);
  scp.run(poly, V1, V2, i, z, ri, r);
  m.end();
  if (i > 260) cout << "TIME: " << m.get_papi_elapsed_time() << endl;

  CMTSumCheckVerifier scv(c);
  scv.run(poly, V1, V2, i, z, ri, r);

  //now reduce claim that V_{i+1}(r1)=V1 and V_{i+1}(r2)=V2 to V_{i+1}(r3)=V3.
  //Let gamma be the line such that gamma(0)=r1, gamma(1)=r2
  //P computes V_{i+1)(gamma(0))... V_{i+1}(gamma(mip1))
  //t1=clock();

  mpz_t *lpoly, *vec;
  alloc_init_vec(&lpoly, mip1 + 1);
  alloc_init_vec(&vec, 2);

  CMTMiniIPProver sipp(c);
  sipp.run(lpoly, i, &r[mi], &r[mi + mip1]);

  CMTMiniIPVerifier sipv(c);
  sipv.run(lpoly, i, V1, V2);

  for(int j = 0; j < mip1; j++)
  {
      mpz_set(vec[0], r[mi + j]);
      mpz_set(vec[1], r[mi + mip1 + j]);
      extrap_ui(z[j], vec, 2, 0, c.prime);
  }

  extrap_ui(rop, lpoly, mip1 + 1, 0, c.prime);

  clear_del_vec(lpoly, mip1 + 1);
  clear_del_vec(vec, 2);

  mpz_clear(V1);
  mpz_clear(V2);
  mpz_clear(tmp);

  */
}

//evaluates the polynomial mult_d for the F2 circuit at point r
void
F2_mult_d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t ans, temp, tmp, tmp2;
  mpz_init(tmp);
  mpz_init(tmp2);
  mpz_init(temp);
  mpz_init_set_ui(ans, 1);

  for(int i = 0; i < mi; i++)
  {
     mpz_mul(temp, r[mi+i], r[mi+mip1+i]);
     mpz_mul(temp, temp, r[i]);

     one_sub(tmp,  r[mi+i]);
     one_sub(tmp2, r[mi+mip1+i]);
     mpz_mul(tmp, tmp, tmp2);
     one_sub(tmp2, r[i]);
     mpz_addmul(temp, tmp, tmp2);

     modmult(ans, ans, temp, prime);
  }
  mpz_set(rop, ans);

  mpz_clear(tmp);
  mpz_clear(tmp2);
  mpz_clear(temp);
  mpz_clear(ans);
}


//this function is used for add_i and mult_i polynomials that are identically zero
//(i.e. no gates of a certain type appear at level i of the circuit)
void
zero(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
    mpz_set_ui(rop, 0);
}

// 
void
check_wiring(mpz_t rop, const mpz_t* r, int out, int in1, int in2, int mi, int mip1, const mpz_t prime)
{
  mpz_t ans;
  mpz_init(ans);

  chi(ans, out, r, mi, prime);
  mul_chi(ans, in1, &r[mi], mip1, prime);
  mul_chi(ans, in2, &r[mi+mip1], mip1, prime);

  modadd(rop, rop, ans, prime);
  mpz_clear(ans);
}

template<typename T>
static void
check_equal_tmpl(mpz_t rop, const vector<T*>& r, int start, int end, const mpz_t prime)
{
  mpz_t tmp, part1, part2, ans;
  mpz_init(tmp);
  mpz_init(part1);
  mpz_init(part2);
  mpz_init_set_ui(ans, 1);

  for (int i = start; i < end; i++)
  {
    mpz_set_ui(part1, 1);
    mpz_set_ui(part2, 1);

    typename vector<T*>::const_iterator it;
    for (it = r.begin(); it != r.end(); it++)
    {
      mpz_mul(part1, part1, (*it)[i]);
      one_sub(tmp, (*it)[i]);
      mpz_mul(part2, part2, tmp);
    }

    mpz_add(part1, part1, part2);
    modmult(ans, ans, part1, prime);
  }

  modmult(rop, rop, ans, prime);

  mpz_clear(tmp);
  mpz_clear(part1);
  mpz_clear(part2);
  mpz_clear(ans);
}

void
check_equal(mpz_t rop, const mpz_t* p, const mpz_t* in1, const mpz_t* in2, int start, int end, const mpz_t prime)
{
  vector<const mpz_t*> r;
  r.push_back(p);
  r.push_back(in1);
  r.push_back(in2);
  check_equal_tmpl(rop, r, start, end, prime);
}

void
check_equal(mpz_t rop, const mpz_t* p, const mpz_t* in1, int start, int end, const mpz_t prime)
{
  vector<const mpz_t*> r;
  r.push_back(p);
  r.push_back(in1);
  check_equal_tmpl(rop, r, start, end, prime);
}

void
check_equal(mpz_t rop, const vector<mpz_t*>& r, int start, int end, const mpz_t prime)
{
  check_equal_tmpl(rop, r, start, end, prime);
}

template<typename T>
static void
check_zero_tmpl(mpz_t rop, const vector<T*>& r, int start, int end, const mpz_t prime)
{
  mpz_t tmp, ans;
  mpz_init(tmp);
  mpz_init_set_ui(ans, 1);

  for (int i = start; i < end; i++)
  {
    typename vector<T*>::const_iterator it;
    for (it = r.begin(); it != r.end(); it++)
    {
      one_sub(tmp, (*it)[i]);
      mpz_mul(ans, ans, tmp);
    }
    mpz_mod(ans, ans, prime);
  }

  modmult(rop, rop, ans, prime);

  mpz_clear(tmp);
  mpz_clear(ans);
}

void
check_zero(mpz_t rop, const mpz_t* p, const mpz_t* in1, const mpz_t* in2, int start, int end, const mpz_t prime)
{
  vector<const mpz_t*> r;
  r.push_back(p);
  r.push_back(in1);
  r.push_back(in2);
  check_zero_tmpl(rop, r, start, end, prime);
}

void
check_zero(mpz_t rop, const mpz_t* p, const mpz_t* in1, int start, int end, const mpz_t prime)
{
  vector<const mpz_t*> r;
  r.push_back(p);
  r.push_back(in1);
  check_zero_tmpl(rop, r, start, end, prime);
}

void
check_zero(mpz_t rop, const vector<mpz_t*>& r, int start, int end, const mpz_t prime)
{
  check_zero_tmpl(rop, r, start, end, prime);
}

//evaluates the polynomial add_i for any layer of the F2 circuit other than the d'th layer
void
reduce(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t temp, ans, tmp, tmp2;
  mpz_init(tmp);
  mpz_init(tmp2);
  mpz_init(temp);
  mpz_init_set_ui(ans, 1);

  //this checks that p=2omega1 and p=2omega2+1, ignoring the first bit of omega1 and omega2
  if (ni == 1)
  {
    one_sub(ans, r[0]);
  }
  else
  {
    check_equal(ans, r, &r[mi+1], &r[mi+mip1+1], 0, mip1-1, prime);
  }

  //finally check that first bit of omega1=0 and first bit of omega2=1
  one_sub(tmp, r[mi]);
  mpz_mul(ans, ans, tmp);
  modmult(rop, ans, r[mi+mip1], prime);

  mpz_clear(tmp);
  mpz_clear(tmp2);
  mpz_clear(temp);
  mpz_clear(ans);
}

void
F0add_dp1to58pd(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
    mpz_set_ui(rop, 0);
}

//evaluates mult_i for the F0 circuit for any i between d+1 and 58+d
void
F0mult_dp1to58pd(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t temp, ans1, ans2, tmp, tmp2;
  mpz_init(tmp);
  mpz_init(tmp2);
  mpz_init(temp);
  mpz_init(ans1);
  mpz_init(ans2);

  mpz_set_ui(ans1, 1);

  // First make sure all but least significant bits of in1 and in2 match p
  for(int i = 1; i < mi; i++)
  {
    mpz_mul(temp, r[mi+i], r[mi+mip1+i]);
    mpz_mul(temp, temp, r[i]);

    one_sub(tmp,  r[mi+i]);
    one_sub(tmp2, r[mi+mip1+i]);
    mpz_mul(tmp, tmp, tmp2);
    one_sub(tmp2, r[i]);
    mpz_addmul(temp, tmp, tmp2);

    modmult(ans1, ans1, temp, prime);
  }

  mpz_set(ans2, ans1);


  //even p's are connected to p and p+1, odd ps are connected to p and p
  
  //first handle even p contribution
  //first term in product makes sure p is even
  one_sub(tmp, r[0]);
  mpz_mul(ans1, ans1, tmp);

  //finally check that least significant bit of in1 is 0 and lsb of in2 is 1
  one_sub(tmp, r[mi]);
  mpz_mul(tmp2, tmp, r[mi+mip1]);
  mpz_mul(ans1, ans1, tmp2);

  //now handle odd p contribution
  mpz_mul(ans2, ans2, r[0]);
  //uint64 ans2 = r[0];

  //finally check that least significant bit of in1 and in2 are 1
  mpz_mul(tmp, r[mi], r[mi+mip1]);
  mpz_mul(ans2, ans2, tmp);
 
  modadd(rop, ans1, ans2, prime); 

  mpz_clear(tmp);
  mpz_clear(tmp2);
  mpz_clear(temp);
  mpz_clear(ans1);
  mpz_clear(ans2);
}

//evaluates mult_i polynomial for layer 59+d of the F0 circuit
void
F0mult_59pd(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
   mpz_t temp, ans, tmp, tmp2;
   mpz_init(tmp);
   mpz_init(tmp2);
   mpz_init(temp);
   mpz_init_set(ans, r[0]);

   //odds ps go to p/2 for both in1 and in2

   //now make sure p>>1 matches in1 and in2
   for(int i = 0; i < mip1; i++)
   {
    mpz_mul(temp, r[mi+i], r[mi+mip1+i]);
    mpz_mul(temp, temp, r[i+1]);

    one_sub(tmp,  r[mi+i]);
    one_sub(tmp2, r[mi+mip1+i]);
    mpz_mul(tmp, tmp, tmp2);
    one_sub(tmp2, r[i+1]);
    mpz_addmul(temp, tmp, tmp2);

    modmult(ans, ans, temp, prime);
   }
   mpz_set(rop, ans);

   mpz_clear(tmp);
   mpz_clear(tmp2);
   mpz_clear(temp);
   mpz_clear(ans);
}

//evaluates add_i polynomial for layer 59+d of the F0 circuit
void
F0add_59pd(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
   mpz_t temp, ans, tmp, tmp2;
   mpz_init(tmp);
   mpz_init(tmp2);
   mpz_init(temp);
   mpz_init(ans);

   //even ps go to nip1-1 for in1 and p/2 for both in2
   one_sub(ans, r[0]);

   //now make sure p>>1 matches in2
   for(int i = 0; i < mip1; i++)
   {
       mpz_mul(temp, r[i+1], r[mi+mip1+i]);
       
       one_sub(tmp,  r[i+1]);
       one_sub(tmp2, r[mi+mip1+i]);
       mpz_addmul(temp, tmp, tmp2);
       modmult(ans, ans, temp, prime);
   }

   //make sure in1 matches nip1-1
   chi(tmp, nip1-1, r+mi, mip1, prime);
   modmult(rop, tmp, ans, prime);

   mpz_clear(tmp);
   mpz_clear(tmp2);
   mpz_clear(temp);
   mpz_clear(ans);
}

//evaluates the mult_i polynomial for the d'th layer of the F0 circuit
void
F0mult_d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  //all gates p have in1=2p, in2=2p+1
  reduce(rop, r, mi, mip1, ni, nip1, prime);
}

//evaluates the mult_i polynomial for layer 60+d of the F0 circuit
void
F0mult_60pd(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  //all gates p but n-2 have in1=in2=p.
  F2_mult_d(rop, r, mi, mip1, ni, nip1, prime);
}


void
mat_add_63_p3d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
    int d=(mi-1)/3;
    
    //with the exception of the final gate at this level, everything should be of the following form
    //z=(j, i, 0, 1) where j, i, and 0 are d bits long and 1 is just 1 bit
    //in1 = (j, i, 0, 1) where j and i are d bits long, and 0 and 1 are just 1 bit each
    //in2 = nip1-1 
    
    //test that the high order bit of z is 1
    mpz_t ans, tmp, tmp2;
    mpz_init(ans);
    mpz_init(tmp);
    mpz_init(tmp2);

    mpz_set(ans, r[mi-1]);
    
    //test that the next d highest order bits of z are 0
    for(int j = 0; j < d; j++) 
    {
        mpz_set_ui(tmp, 1);
        mpz_sub(tmp, tmp, r[2*d+j]);
        modmult(ans, ans, tmp, prime);
    }
    
    //test that the highest-order 2 bits of in1 are 0, 1 (highest-order bit is 1)

    mpz_set_ui(tmp, 1);
    mpz_sub(tmp, tmp, r[mi+2*d]);
    mpz_mul(tmp, tmp, r[mi+2*d+1]);
    modmult(ans, ans, tmp, prime);
    
    //test that the lowest order 2d bits of in1 are (j, i)
    for(int j = 0; j < 2*d; j++)
    { //test that in1 is the i'th entry of C, where z is n^3+i
        mpz_set_ui(tmp, 1);
        mpz_sub(tmp, tmp, r[mi+j]);

        mpz_set_ui(tmp2, 1);
        mpz_sub(tmp2, tmp2, r[j]);

        mpz_mul(tmp, tmp, tmp2);
        mpz_mul(tmp2, r[mi+j], r[j]);
        mpz_add(tmp, tmp, tmp2);
        modmult(ans, ans, tmp, prime);
    }
    
    //test that in2 equals nip1-1
    mul_chi(ans, nip1 - 1, &r[mi + mip1], mip1, prime);
    
    //handle the case where z=ni-1, in1=nip1-1, in2=nip1-1
    check_wiring(ans, r, ni-1, nip1-1, nip1-1, mi, mip1, prime);

    mpz_set(rop, ans);

    mpz_clear(ans);
    mpz_clear(tmp);
    mpz_clear(tmp2);
}

void
mat_mult_63_p3d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
    int d=(mi-1)/3;

    mpz_t tmp, tmp2;
    mpz_init(tmp);
    mpz_init(tmp2);

    one_sub(rop, r[mi - 1]);
    mpz_mod(rop, rop, prime);

    //test that high order two bits in in1 are 0 (i.e. you're looking at an entry of A < n^2)
    one_sub(tmp, r[mi+mip1-1]);
    mpz_mul(rop, rop, tmp);

    one_sub(tmp, r[mi+mip1-2]); 
    modmult(rop, rop, tmp, prime);

    //make sure in2 >= n^2, <= 2n^2 (second highest bit is 1, highest bit is 0)
    one_sub(tmp, r[mi+2*mip1-1]);
    mpz_mul(rop, rop, tmp);

    modmult(rop, rop, r[mi+2*mip1-2], prime);
    
    //make sure in1 = (k, i, 0, 0), where z=(k, j, i, 0). This loop tests i
    for(int j = 0; j < d; j++)
    {
        //mi+2*d+j is pulling out high order d bits of z, mi+d+j is pulling out second d bits of in1
        one_sub(tmp, r[2*d+j]);
        one_sub(tmp2,  r[mi+d+j]);
        mpz_mul(tmp, tmp, tmp2);

        mpz_addmul(tmp, r[2*d+j], r[mi+d+j]);
        modmult(rop, rop, tmp, prime);
    }
    //make sure in1 = (k, i, 0, 0) and in2=(j, k, 1, 0). This loop tests k for both
    for(int j = 0; j < d; j++)
    {
        //2*d+j is pulling k out of z, mi+j is pulling first d bits out of in1, 
        //and mi+mip1+d+j is pulling second d bits out of in2
        one_sub(tmp, r[mi+j]);
        one_sub(tmp2, r[mi+mip1+d+j]);
        mpz_mul(tmp, tmp, tmp2);

        one_sub(tmp2, r[j]);
        mpz_mul(tmp, tmp, tmp2);

        mpz_mul(tmp2, r[mi+j], r[mi+mip1+d+j]);
        mpz_addmul(tmp, tmp2, r[j]);

        modmult(rop, rop, tmp, prime);
    }
    //make sure in2 = (j, k, 1, 0). This loop tests j
    for(int j = 0; j < d; j++)
    {
        one_sub(tmp, r[d+j]);
        one_sub(tmp2, r[mi+ mip1 + j]);
        mpz_mul(tmp, tmp, tmp2);

        mpz_addmul(tmp, r[d+j], r[mi + mip1  + j]);
        modmult(rop, rop, tmp, prime);
    }

    mpz_clear(tmp);
    mpz_clear(tmp2);
}

void
mat_add_below_63_p3d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
    int three_d_min_i = mi-1;
    int n_squared = ni-powi(2, mi-1)-1;
    int d = floor(Log2((double) n_squared)/2 + 0.5);
    
    //threshold in wiring structure is at 3d-i (gate 2^{3d-i} is first entry of C)
    int i = (three_d_min_i - 3*d) * (-1);

    mpz_t tmp, tmp2;
    mpz_init(tmp);
    mpz_init(tmp2);
    
    //all gates at this layer are of the form z=(anything, 0), in1=(2z, 0), in2=(2z+1, 0) where anything is 3d-i bits
    //or z=(anything, (d-i zeros), 1) in1=(anything, (d-i+1 zeros), 1), in2=nip1-1

    //check high order bit of z is 0, and same with high order bit of in1 and in2
    one_sub(tmp,  r[mi+mip1-1]);
    one_sub(tmp2, r[mi+2*mip1-1]);
    mpz_mul(tmp, tmp, tmp2);

    one_sub(rop, r[mi-1]);
    modmult(rop, rop, tmp, prime);
   
    mpz_t temp;
    mpz_init(temp);
    //uint64 temp=0;

    //this checks that z=2omega1 ignoring high order bit of each and z=2omega2+1, ignoring the low-order bit of omega1 and omega2
    for(int j = 0; j < mi-1; j++)
    {
        mpz_mul(temp, r[mi+j+1], r[mi+mip1+j+1]);
        mpz_mul(temp, temp, r[j]);

        one_sub(tmp, r[mi+j+1]);
        one_sub(tmp2, r[mi+mip1+j+1]);
        mpz_mul(tmp, tmp, tmp2);
        one_sub(tmp2, r[j]);
        mpz_addmul(temp, tmp, tmp2);

        modmult(rop, rop, temp, prime);
    }
    //finally check that low-order bit of in1=0 and low-order bit of in2=1
    one_sub(tmp, r[mi]);
    mpz_mul(rop, rop, tmp);
    modmult(rop, rop, r[mi+mip1], prime);
    
    //now handle the z=(anything, (d-i zeros), 1) in1=(anything, (d-i+1 zeros), 1), in2=nip1-1 case
    
    //check that the high order bits of z are (d-i zeros) followed by a 1
    mpz_t part_two;
    mpz_init_set(part_two, r[mi-1]);
    for(int j = 0; j < d-i; j++)
    {
        one_sub(tmp, r[2*d+j]);
        modmult(part_two, part_two, tmp, prime);
    }
    
    //check that highest order bit of in1 is a 1, and then next d-i+1 highest order bits are 0

    modmult(part_two, part_two, r[mi+mip1-1], prime);
    for(int j = 0; j < d-i+1; j++)
    {
        one_sub(tmp, r[mi+2*d+j]);
        modmult(part_two, part_two, tmp, prime);
    }
    
    //check that lowest order 2*d bits of z and in1 agree
    for(int j = 0; j < 2*d; j++)
    {
        one_sub(tmp, r[j]);
        one_sub(tmp2, r[mi+j]);
        mpz_mul(tmp, tmp, tmp2);
        mpz_addmul(tmp, r[j], r[mi+j]);
        modmult(part_two, part_two, tmp, prime);
    }
    
    
    //check that in2 = nip1-1
    mul_chi(part_two, nip1-1, &r[mi+mip1], mip1, prime);
    
    mpz_t part_three;
    mpz_init_set_ui(part_three, 0);

    check_wiring(part_three, r, ni-1, nip1-1, nip1-1, mi, mip1, prime);

    mpz_add(tmp, part_two, part_three);
    modadd(rop, rop, tmp, prime);

    mpz_clear(tmp);
    mpz_clear(tmp2);
    mpz_clear(part_two);
    mpz_clear(part_three);
    mpz_clear(temp);
}



void
mat_add_61_p2dp1(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
	int three_d_min_i = mi-1;
	int n_squared = ni-powi(2, mi-2)-1;
	int d = floor(Log2((double) n_squared)/2 + 0.5);
	
	//threshold in wiring structure is at 3d-i (gate 2^{3d-i} is first entry of C)
	int i = (three_d_min_i - 3*d) * (-1);
	
	//all gates at this layer are of the form z=(anything, 0, 0), in1=(2z, 0), in2=(2z+1, 0) where anything is 2d bits
	//or z=(anything, 1, 0) in1=(anything, 0, 1), in2=nip1-1

        mpz_t tmp, tmp2;
        mpz_init(tmp);
        mpz_init(tmp2);

	//check high order bit of z is 0, and same with high order bit of in1 and in2
        one_sub(tmp, r[mi + 2*mip1 - 1]);
        one_sub(tmp2, r[mi + mip1 - 1]);
        one_sub(rop, r[mi-1]);
        mpz_mul(tmp, tmp, tmp2);
        modmult(rop, rop, tmp, prime);
	
	//check second highest order bit of z is 0
        one_sub(tmp, r[mi - 2]);
        modmult(rop, rop, tmp, prime);
	
        mpz_t temp;
        mpz_init(temp);

  	//this checks that z=2omega1 ignoring high order bit of each and z=2omega2+1, ignoring the low-order bit of omega1 and omega2
  	for(int j = 0; j < mi-2; j++)
  	{
            mpz_mul(temp, r[mi+j+1], r[mi+mip1+j+1]);
            mpz_mul(temp, temp, r[j]);

            one_sub(tmp, r[mi+j+1]);
            one_sub(tmp2, r[mi+mip1+j+1]);
            mpz_mul(tmp, tmp, tmp2);
            one_sub(tmp2, r[j]);
            mpz_addmul(temp, tmp, tmp2);

            modmult(rop, rop, temp, prime);
  	}

  	//finally check that low-order bit of in1=0 and low-order bit of in2=1
        one_sub(tmp, r[mi]);
        mpz_mul(rop, rop, tmp);
        modmult(rop, rop, r[mi+mip1], prime);
  	
  	//now handle the z=(anything, 1, 0) in1=(anything,  0, 1), in2=nip1-1 case
  	
  	//check that the high order bits of z a 1 followed by a zero

        mpz_t part_two;
        mpz_init(part_two);
        one_sub(tmp, r[mi-1]);
        modmult(part_two, tmp, r[mi-2], prime);
  	
  	
  	//check that highest order bits of in1 are a 0 followed by a 1,
        one_sub(tmp, r[mi+mip1-2]);
        mpz_mul(tmp, tmp, r[mi+mip1-1]);
        modmult(part_two, part_two, tmp, prime);

  	
  	//check that lowest order 2*d bits of z and in1 agree
  	for(int j = 0; j < 2*d; j++)
  	{
            one_sub(tmp, r[j]);
            one_sub(tmp2, r[mi+j]);
            mpz_mul(tmp, tmp, tmp2);

            mpz_addmul(tmp, r[j], r[mi+j]);
            modmult(part_two, part_two, tmp, prime);
  	}
  	
  	
  	//check that in2 = nip1-1
        mul_chi(part_two, nip1-1, &r[mi+mip1], mip1, prime);
	
	
        mpz_t part_three;
        mpz_init_set_ui(part_three, 0);
        check_wiring(part_three, r, ni-1, nip1-1, nip1-1, mi, mip1, prime);

    mpz_add(tmp, part_two, part_three);
    modadd(rop, rop, tmp, prime);

    mpz_clear(tmp);
    mpz_clear(tmp2);
    mpz_clear(part_two);
    mpz_clear(part_three);
    mpz_clear(temp);
}


void
mat_add_61_p2d(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  //first handle case where z=(any, 0), in1 = (z, 0, 0), in2 = (z, 1, 0)
  //the other case is z=ni-1, in1=in2=nip1-1
  
  //test high order bit of z is 0
  one_sub(rop, r[mi-1]);

  mpz_t tmp, tmp2;
  mpz_init(tmp);
  mpz_init(tmp2);

  //test in1 and in2 low order bits match z
  for(int j = 0; j < mi-1; j++)
  {
    one_sub(tmp,  r[mi+j]);
    one_sub(tmp2, r[mi+mip1+j]);
    mpz_mul(tmp, tmp, tmp2);

    one_sub(tmp2, r[j]);
    mpz_mul(tmp, tmp, tmp2);

    mpz_mul(tmp2, r[mi+j], r[mi+mip1+j]);
    mpz_mul(tmp2, tmp2, r[j]);

    mpz_add(tmp, tmp, tmp2);
    modmult(rop, rop, tmp, prime);
  }	
  //test high order two bits of in1 and in2. in1 first
  one_sub(tmp,  r[mi+mip1-2]);
  one_sub(tmp2, r[mi+mip1-1]);
  mpz_mul(tmp, tmp, tmp2);
  modmult(rop, rop, tmp, prime);

  one_sub(tmp,  r[mi+2*mip1-1]);
  mpz_mul(tmp2, tmp, r[mi+2*mip1-2]);
  modmult(rop, rop, tmp2, prime);
  
  //now handle the case of the last gate
  check_wiring(rop, r, ni-1, nip1-1, nip1-1, mi, mip1, prime);

    mpz_clear(tmp);
    mpz_clear(tmp2);
}
	


//update vals fast
void update_a_fast(uint64 num_effective, mpz_t rjm1, mpz_t* vals, mpz_t prime)
{
  uint64 index;
  for(uint64 k=0; k < num_effective; k++)
  {
     index = k >> 1;
     if( k & 1)
     {
         addmodmult(vals[index], vals[k], rjm1, prime);
     }
     else
     {
         // Computes vals[index] = vals[k] * (1 - rjm1) = vals[k] + (-rjm1*vals[k])
         mpz_set(vals[index], vals[k]);
         mpz_neg(rjm1, rjm1);
         addmodmult(vals[index], vals[k], rjm1, prime);
         mpz_neg(rjm1, rjm1);
     }
  }
}


void
check_first_level(mpz_t rop, Circuit& c, mpz_t* r, mpz_t* zi, int d)
{
  int num_effective = powi(2, c[d].logSize());

  mpz_t* a;
  mpz_t* vals;
  mpz_t** y;

  alloc_init_vec(&vals, num_effective);
  alloc_init_vec(&a, num_effective);
  alloc_init_vec_array(&y, c[d].logSize(), 2);

  for(int i = 0; i < c[d].size(); i++)
      mpz_set(a[i], c[d].gate(i).zValue());

  mpz_t check, tmp;
  mpz_init_set_ui(check, 0);
  mpz_init(tmp);

  uint64 ct=0; //check time

  generate_randomness(r, c[d].logSize(), c.prime);

  clock_t t=clock();
  for(int j=0; j < c[d].logSize(); j++) // computes all the messages from prover to verifier
  {
      for(int k = 0; k < (num_effective>>1); k++)
      {
        //a[k] gives f(r0...rj-1, k). Want to plug in m for x_j
        mpz_add(y[j][1], y[j][1], a[(k<<1)+1]);
        mpz_add(y[j][0], y[j][0], a[(k<<1)]);
      }

      mpz_mod(y[j][1], y[j][1], c.prime);
      mpz_mod(y[j][0], y[j][0], c.prime);

      update_a_fast(num_effective, r[j], a, c.prime);
      num_effective = num_effective >> 1;
  }
  clock_t pt=clock()-t; //prover time

  //cout << "claimed circuit value is: " << (uint64) myMod(y[0][0] + y[0][1]) << endl;
  mpz_t c_value;
  mpz_init(c_value);
  modadd(c_value, y[0][0], y[0][1], c.prime);

  gmp_printf("claimed circuit value is: %Zd\n", c_value);
  t=clock();
  for(int j = 0; j < c[d].logSize(); j++) //checks all messages from prover to verifier
  {
    if (j>0)
    {
      if (mpz_cmp(check, c_value) != 0)
      {
        gmp_printf("Check failed: j is %d\n", j);
        gmp_printf("check is %Zd\n", check);
        gmp_printf("c_value is %Zd\n", c_value);
        gmp_printf("y[%d][0]: %Zd\n", j, y[j][0]);
        gmp_printf("y[%d][1]: %Zd\n", j, y[j][1]);
        cout << endl;
      }
    }
    mpz_mul(check, y[j][1], r[j]);
    one_sub(tmp, r[j]);
    addmodmult(check, y[j][0], tmp, c.prime);
  }

  for(int i = 0; i < c[d].logSize(); i++)
  {
      mpz_set(zi[i], r[i]);
  }

  //return check;
  mpz_set(rop, check);
  mpz_clear(check);
  mpz_clear(tmp);

  clear_del_vec(vals, 1 << c[d].logSize());
  clear_del_vec(a, c[d].size());
  clear_del_vec_array(y, c[d].logSize(), 2);
}

