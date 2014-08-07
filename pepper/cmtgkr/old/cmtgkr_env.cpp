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

#include "cmtgkr_env.h"
#include "cmt_sumcheck_p.h"

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

void
free_vec(mpz_t *arr, int size)
{
    for (int i = 0; i < size; i++)
    {
        mpz_clear(arr[i]);
    }
    delete[] arr;
}

//computes x^b
uint64 powi(uint64 x, uint64 b) 
{
  uint64 i = 1;
  for (uint64 j = 0; j < b; j++)  i *= x;
  return i;
}


void
modmult(mpz_t rop, const mpz_t op1, const mpz_t op2, const mpz_t prime)
{
    mpz_mul(rop, op1, op2);
    mpz_mod(rop, rop, prime);
}

void
modmult_si(mpz_t rop, const mpz_t op1, const long op2, const mpz_t prime)
{
    mpz_mul_si(rop, op1, op2);
    mpz_mod(rop, rop, prime);
}

void
addmodmult(mpz_t rop, const mpz_t op1, const mpz_t op2, const mpz_t prime)
{
    mpz_addmul(rop, op1, op2);
    mpz_mod(rop, rop, prime);
}

void
addmodmult_ui(mpz_t rop, const mpz_t op1, const unsigned long op2, const mpz_t prime)
{
    mpz_addmul_ui(rop, op1, op2);
    mpz_mod(rop, rop, prime);
}

void
addmodmult_si(mpz_t rop, const mpz_t op1, const long op2, const mpz_t prime)
{
    mpz_addmul_ui(rop, op1, abs(op2));
    if (op2 < 0)
      mpz_neg(rop, rop);
    mpz_mod(rop, rop, prime);
}

void
modadd(mpz_t rop, const mpz_t op1, const mpz_t op2, const mpz_t prime)
{
    mpz_add(rop, op1, op2);
    mpz_mod(rop, rop, prime);
}

// Calculates log2 of number.  
double Log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( (double) n ) / log( (double) 2 );  
}

void
mul_chi(mpz_t rop, const uint64 v, const mpz_t *r, const int n, const mpz_t prime)
{
    mpz_t tmp;
    mpz_init(tmp);

    for(int i = 0; i < n; i++)
    {
        if( (v >> i) & 1)
        {
            modmult(rop, rop, r[i], prime);
        }
        else
        {
            one_sub(tmp, r[i]);
            modmult(rop, rop, tmp, prime);
        }
    }

    mpz_clear(tmp);
}

//computes chi_v(r), where chi is the Lagrange polynomial that takes
//boolean vector v to 1 and all other boolean vectors to 0. (we view v's bits as defining
//a boolean vector. n is dimension of this vector.
//all arithmetic done mod p
void
chi(mpz_t rop, const uint64 v, const mpz_t* r, const int n, const mpz_t prime)
{
    mpz_t chi;
    mpz_init_set_ui(chi, 1);

    mul_chi(chi, v, r, n, prime);

    mpz_set(rop, chi);
    mpz_clear(chi);
}

//evaluates beta_z polynomial (described in GKR08)
//at location k (the bits of k are interpreted as a boolean vector). mi is dimension of k and r.
void
evaluate_beta_z_ULL(mpz_t rop, const int mi, const mpz_t* z, const uint64 k, const mpz_t prime)
{
    chi(rop, k, z, mi, prime);
}

//evaluates V_i polynomial at location r.
//Here V_i is described in GKR08; it is the multi-linear extension
//of the vector of gate values at level i of the circuit
void
evaluate_V_i(
        mpz_t rop, const int mi, const int ni,
        const Gate* level_i, const mpz_t* r,
        const mpz_t prime)
{
    mpz_t op2, ans;
    mpz_init(op2);
    mpz_init_set_ui(ans, 0);
    for (int k = 0; k < ni; k++)
    {
        chi(op2, k, r, mi, prime);
        addmodmult(ans, level_i[k].val, op2, prime);
    }
    mpz_set(rop, ans);
    mpz_clear(op2);
    mpz_clear(ans);
}

void
extrap3(mpz_t rop, const mpz_t* vec, const mpz_t r, const mpz_t prime)
{
  mpz_t tmp1, tmp2;
  mpz_init(tmp1);
  mpz_init(tmp2);

  mpz_sub_ui(tmp1, r, 1);
  mpz_sub_ui(tmp2, r, 2);
  mpz_mul(rop, tmp1, tmp2);
  mpz_mul(rop, rop, vec[0]);

  mpz_mul_2exp(tmp2, tmp2, 1);
  mpz_mul(tmp2, tmp2, vec[1]);

  mpz_mul(tmp1, tmp1, vec[2]);

  mpz_sub(tmp1, tmp1, tmp2);
  mpz_mul(tmp1, tmp1, r);

  mpz_add(rop, rop, tmp1);

  mpz_add_ui(tmp1, prime, 1);
  mpz_tdiv_q_2exp(tmp1, tmp1, 1);

  mpz_mul(rop, rop, tmp1);
  mpz_mod(rop, rop, prime);

  mpz_clear(tmp1);
  mpz_clear(tmp2);
}


//extrapolate the polynomial implied by vector vec of length n to location r
void
extrap(mpz_t rop, const mpz_t* vec, const uint64 n, const mpz_t r, const mpz_t prime)
{
    mpz_t mult, inv;
    mpz_init(mult);
    mpz_init(inv);
    mpz_set_ui(rop, 0);

    for (uint64 i = 0; i < n; i++)
    {
        mpz_set_ui(mult, 1);
        for (uint64 j = 0; j < n; j++)
        {
            if (i != j)
            {
                mpz_set_si(inv, i - j);
                mpz_invert(inv, inv, prime);

                mpz_mul(mult, mult, inv);
                mpz_sub_ui(inv, r, j);
                modmult(mult, mult, inv, prime);
            }
        }
        addmodmult(rop, mult, vec[i], prime);
    }

    mpz_clear(mult);
    mpz_clear(inv);
}

void
extrap_ui(mpz_t rop, const mpz_t* vec, const uint64 n, const uint64 r, const mpz_t prime)
{
    mpz_t mpzr;
    mpz_init_set_ui(mpzr, r);
    extrap(rop, vec, n, mpzr, prime);
    mpz_clear(mpzr);
}

static void
init_vals_array(mpz_t* vals, Circuit& c, int i, int nip1)
{
    int num_effective = powi(2, c[i+1].logSize());
    for(int k = 0; k < num_effective; k++)
    {
       if(k < nip1)
         mpz_set(vals[k], c[i + 1][k].val);
       else
         mpz_set_ui(vals[k], 0);
    }
}

void
one_sub(mpz_t rop, const mpz_t op)
{
  // Be careful in case op and rop are the same number.
  mpz_neg(rop, op);
  mpz_add_ui(rop, rop, 1);
}

static void
generateRandomness(mpz_t* r, Circuit& c, int lvl)
{
  const int n = c[lvl].logSize() + 2 * c[lvl+1].logSize();
  for(int j = 0; j < n; j++)
  {
      prng.get_random(r[j], c.prime);
      // r[j] = rand();
      // DEBUG
      // mpz_set_ui(r[j], counter++);
  }
}

static void
computeBetaZAll(CircuitLayer& l, const mpz_t* z, const int n, const mpz_t prime)
{
  int logn = ceil(log2(n));
  mpz_t* zInv = new mpz_t[2 * logn];

  for (int i = 0; i < 2 * logn; i++)
    mpz_init(zInv[i]);

  mpz_t* zInv1 = &zInv[logn];
  mpz_t* zInv0 = zInv;

  // Compute all the zi and (1 - zi) that we need.
  for (int i = 0; i < logn; i++)
  {
    one_sub(zInv0[i], z[i]);
    mpz_set(zInv1[i], z[i]);
  }

  if (false)
  {
    for (int i = 0; i < n; i++)
    {
      evaluate_beta_z_ULL(l[i].betar, logn, z, i, prime);
    }
  }
  else
  {
    mpz_set_ui(l[0].betar, 1);

    // Interleave the computations of all the betars.
    // Basically, compute at step
    //   1 :  (1 - z1)            z1
    //   2 :  (1 - z1) (1 - z2)   z1 (1 - z2)   (1 - z1) z2   z1 z2
    //   etc.
    for (int logi = 0; logi < logn; logi++)
    {
      const int base = 1 << logi;
      for (int i = base; i < min(2 * base, n); i++)
      {
        modmult(l[i].betar, l[i - base].betar, zInv1[logi], prime);
      }

      for (int i = 0; i < base; i++)
      {
        modmult(l[i].betar, l[i].betar, zInv0[logi], prime);
      }
    }
  }

  for (int i = 0; i < 2 * logn; i++)
    mpz_clear(zInv[i]);
  delete[] zInv;
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
    int* com_ct, int* rd_ct, mpz_t* r, mpz_t** poly, mpz_t* vals
)
{
  //Although GKR is stated assuming that all levels of the circuit have same number of
  //gates, our implementation does not require this. Thus, ni will be number of gates at level i
  //and nip1 will be number of gates at level i+1; mi and mip1 are ceil(log(ni or nip1)) respectively.
  //This yields significant speedups in practice.

  int mi    = c[i].logSize();
  int ni    = c[i].size();
  int mip1  = c[i+1].logSize();
  int nip1  = c[i+1].size();
  int nvars = mi+2*mip1; //number of variables being summed over in this iteration of GKR

  // DEBUG printf("i: %d | mi: %d | mip1: %d | nip1: %d | nvars: %d\n", i, mi, mip1, nip1, nvars);

  *com_ct = *com_ct + 3*nvars;

  //r is V's random coin tosses for this iteration.
  generateRandomness(r, c, i);

  *rd_ct = *rd_ct + mi+2*mip1;

  // DEBUG gmp_printf("i: %d | ri: %Zd\n", i, ri);

  //initialize betar and addormultr values for this iteration of protocol
  //these values are used to track each gate's contribution to beta_z, add_i, or mult_i polynomials
  m.begin_with_history();
  computeBetaZAll(c[i], z, ni, c.prime);
  for(int k = 0; k < ni; k++)
  {
    //evaluate_beta_z_ULL(c[i][k].betar, mi, z, k, c.prime);
    mpz_set_ui(c[i][k].addormultr, 1);
      // DEBUG { int j = 0; if (j == 0 && i == i) gmp_printf("i: %d | j: %d | k: %d | c[i][k].betar: %Zd\n", i, j, k, c[i][k].betar); }

  }

  mpz_t betar, addormultr, V1, V2, tmp;

  mpz_init_set_ui(betar, 0); //represents contribution to beta(z) by a given gate
  mpz_init_set_ui(addormultr, 0); //represents contribution to tilde{add} or tilde{mult} by a given gate
  mpz_init_set_ui(V1, 0); //represents contribution to V_{i+1}(omega1) by a given gate
  mpz_init_set_ui(V2, 0); //represents contribution to V_{i+1}(omega2) by a given gate
  mpz_init(tmp);

  int q;
  int idx;

  //initialize p array, which will store P's messages for this iteration of the protocol
  for(int k = 0; k < nvars; k++)
    for(int j = 0; j <= 2; j++)
      mpz_set_ui(poly[k][j], 0);

  //num_effective is used for computing the necessary V(omega1) or V(omega2) values at any round of the protocol
  int num_effective = powi(2, c[i+1].logSize());
  uint64 kshiftj = 0;

  //initialize vals array, which will store evaluate of Vi+1 at various points
  //intially, the protocol needs to know Vi+1 at all points in boolean hypercube
  //so that's what we initialize to.
  init_vals_array(vals, c, i, nip1);

  mpz_t betartimesV1;
  mpz_init(betartimesV1);
  //uint64 betartimesV1;
  //run through each round in this iteration, one round for each of the mi+2mip1 variables
  for(int j = 0; j < nvars; j++)
  {
    // DEBUG cout << j << "/" << nvars << endl;
    //if we're in a round > mi, we need to know evaluations of Vi+1 at some non-boolean locations
    //The following logic computes these values
    if(j > mi && (j != mi+mip1))
    { 
      for(int k=0; k < num_effective; k++)
      {
        // DEBUG cout << "  " << k << "/" << num_effective << endl;
        idx = k >> 1;
        if(k & 1)
        {
            addmodmult(vals[idx], vals[k], r[j-1], c.prime);
        }
        else
        {
            one_sub(tmp, r[j-1]);
            modmult(vals[idx], vals[k], tmp, c.prime);
        }
      }
      //num_effective tracks the number of points at which we need to evaluate Vi+1
      //for this round of the protocol (it halves every round, as one more variable becomes 'bound')
      num_effective >>= 1;
    }
    if(j == mi + mip1)
    {
        num_effective = powi(2, c[i+1].logSize());
        init_vals_array(vals, c, i, nip1);
    	evaluate_V_i(V1, mip1, nip1, c[i+1].gates, r+mi, c.prime);
        modmult(betartimesV1, betar, V1, c.prime);
    }

    if(j == mi)
    {   //by round mi of this iteration, the first mi variables have been bound, 
        //so beta is fixed for the rest of this iteration
        mpz_set(betar, c[i][0].betar);
    }
    
  // DEBUG if (i == 63 && j == 8) for (int k = 0; k < num_effective;k++) gmp_printf("vals[%d]: %Zd\n", k, vals[k]);

    //for each gate in level i of the circuit, compute its contribution to P's message in round j
    for(int k=0; k < ni; k++)
    {
      //cout << "  " << k << "/" << ni << endl;
      //cout << "    j: " << j << ", mi: " << mi << endl;
      //in cases where V1 or V2 are trivial to compute, compute them now
      if(j < mi)
      {
          mpz_set(V1, c[i+1][c[i][k].in1].val);
          mpz_set(V2, c[i+1][c[i][k].in2].val);
      }
      else if(j < mi+mip1)
      {
          mpz_set(V2, c[i+1][c[i][k].in2].val);
      }

      //prep betar fields because in this round variable j will take on values 0, 1, or 2 rather than just 0,1
      if(j < mi)
      {
         kshiftj = k>>j;
         if (kshiftj & 1)
             mpz_invert(tmp, z[j], c.prime);
         else
         {
             one_sub(tmp, z[j]);
             mpz_invert(tmp, tmp, c.prime);
         }
         modmult(c[i][k].betar, c[i][k].betar, tmp, c.prime);
      }
      //now we iterate over the points at which we evaluate the polynomial for this round
      for(int m = 0; m <= 2; m++)
      {
        if(j < mi)
        {
           //if (k == 0 && i == i) gmp_printf("[old] j: %d | m: %d | betar: %Zd | betargate: %Zd\n", j, m, betar, c[i][k].betar);
           // Compute betar for this gate for this round, and update betar field
           // if we're done with it for this round (m==2)
           mpz_set_ui(tmp, 0);
           if(m != 1)
           {
                one_sub(tmp, z[j]);
                mpz_mul_si(tmp, tmp, 1 - m);
           }
           if(m != 0)
           {
                mpz_addmul_ui(tmp, z[j], m);
           }
           else
           {
           }
           modmult(betar, c[i][k].betar, tmp, c.prime);
           //if (k == 0 && i == i) gmp_printf("[old] j: %d | m: %d | betar: %Zd | betargate: %Zd\n", j, m, betar, c[i][k].betar);

           if(m==2)
           {
               // Compute (1 - z[j])(1 - r[j]) + z[j] * r[j] =
               //         2 z[j] r[j] - z[j] - r[j] + 1
               mpz_mul(tmp, z[j], r[j]);
               mpz_add(tmp, tmp, tmp);
               mpz_sub(tmp, tmp, z[j]);
               mpz_sub(tmp, tmp, r[j]);
               mpz_add_ui(tmp, tmp, 1);

               modmult(c[i][k].betar, c[i][k].betar, tmp, c.prime);
           }

          //compute addormult for this gate for this round, and update the field
          //if we're done with it for this round (m==2)
          if(kshiftj & 1)
          {
            if(m==0) 
              continue;
            else
              modmult_si(addormultr, c[i][k].addormultr, m, c.prime);
            if(m==2)
            {
              modmult(c[i][k].addormultr,
                      c[i][k].addormultr,
                      r[j],
                      c.prime);
            }
          }
          else
          {
            if(m==1)
              continue;
            else 
              modmult_si(addormultr, c[i][k].addormultr, 1 - m, c.prime);
            if(m==2)
            {
              one_sub(tmp, r[j]);
              modmult(c[i][k].addormultr,
                      c[i][k].addormultr,
                      tmp,
                      c.prime);
            }
          }
        }

        if(j >= mi && j < mi+mip1)
        {
           //if (k == 1) gmp_printf("[old] j: %d | m: %d | V1: %Zd | addm: %Zd\n", j, m, V1, addormultr);
           //now compute contribution to V_i+1(omega1) for this gate
           idx=c[i][k].in1 >> (j-mi);
           if(idx & 1)
           {
             if(m==0)
              continue; 
             else if(m==1)
                mpz_set(V1, vals[idx]);
             else
             {
                 mpz_mul_si(V1, vals[idx - 1], 1 - m);
                 addmodmult_ui(V1, vals[idx], m, c.prime);
             }
              //now compute contribution to tilde{add}_i or tilde{multi}_i for this gate
             modmult_si(addormultr, c[i][k].addormultr, m, c.prime);
             if(m==2)
             {
               modmult(c[i][k].addormultr,
                       c[i][k].addormultr,
                       r[j],
                       c.prime);
             }
           }
           else
           {
             if(m==1)
               continue;
             else if(m==0)
                mpz_set(V1, vals[idx]);
             else
             {
                 mpz_mul_si(V1, vals[idx], 1 - m);
                 addmodmult_ui(V1, vals[idx + 1], m, c.prime);
             }
             //now compute contribution to tilde{add}_i or tilde{multi}_i for this gate
             modmult_si(addormultr, c[i][k].addormultr, 1 - m, c.prime);
             if(m==2)
             {
                 one_sub(tmp, r[j]);
                 modmult(c[i][k].addormultr,
                         c[i][k].addormultr,
                         tmp,
                         c.prime);
             }
	   }  
           //if (k == 1) gmp_printf("[old] j: %d | m: %d | V1: %Zd | addm: %Zd\n", j, m, V1, addormultr);
        }
        else if(j >= mi+mip1)
        {
          //compute contribution to V_{i+1}(omega2) for this gate
          idx=c[i][k].in2 >> (j-mi-mip1);
          // DEBUGif ((j == 8) && (i == 63)) gmp_printf("m: %d | vals[idx]: %Zd\n", m, vals[idx]);
          if(idx & 1)
          {
            if(m==0)
              continue;
            if(m==1)
                mpz_set(V2, vals[idx]);
            else 
             {
                 mpz_mul_si(V2, vals[idx - 1], 1 - m);
                 addmodmult_ui(V2, vals[idx], m, c.prime);
             }

            modmult_si(addormultr, c[i][k].addormultr, m, c.prime);
            if(m==2)
            {
                 modmult(c[i][k].addormultr,
                         c[i][k].addormultr,
                         r[j],
                         c.prime);
            }

            if ((j == 8) && (i == 63) && false) // DEBUG
            {
              gmp_printf("V1: %Zd\n", V1);
              gmp_printf("V2: %Zd\n", V2);
              gmp_printf("betar: %Zd\n", betar);
              gmp_printf("addormultr: %Zd\n", addormultr);
            }

           } 
           else
           {
             if(m==1)
               continue;
             if(m==0)
               mpz_set(V2, vals[idx]);
             else   
             {
                 mpz_mul_si(V2, vals[idx], 1 - m);
                 addmodmult_ui(V2, vals[idx+1], m, c.prime);
             }

            modmult_si(addormultr, c[i][k].addormultr, 1 - m, c.prime);
            if(m==2)
            {
                 one_sub(tmp, r[j]);
                 modmult(c[i][k].addormultr,
                         c[i][k].addormultr,
                         tmp,
                         c.prime);
            }
           }
        }

        //finally, update the evaluation of this round's polynomial at m based on this gate's contribution
        //to beta, add_i or multi_i and V_{i+1}(omega1) and V_{i+1}(omega2)

        if(c[i][k].type==0)
        {
          mpz_add(tmp, V1, V2);
          mpz_mul(tmp, tmp, addormultr);
          addmodmult(poly[j][m], betar, tmp, c.prime);
           //if (k == 1) gmp_printf("[old] j: %d | m: %d | V1: %Zd | poly: %Zd\n", j, m, V1, poly[j][m]);
        }
        else if(c[i][k].type==1)
        {
           if(j<mi+mip1)
           {
             mpz_mul(tmp, V1, V2);
             mpz_mul(tmp, tmp, addormultr);
             addmodmult(poly[j][m], betar, tmp, c.prime);
           //if (k == 1) gmp_printf("[old] j: %d | m: %d | V1: %Zd | poly: %Zd\n", j, m, V1, poly[j][m]);
           }
           else
           {
             mpz_mul(tmp, addormultr, V2);
             addmodmult(poly[j][m], betartimesV1, tmp, c.prime);
           }
        }
        else
        { 
          mpz_powm_ui(tmp, V2, c[i][j].type, c.prime);
          mpz_mul(tmp, tmp, addormultr);
          addmodmult(poly[j][m], betar, tmp, c.prime);
           //if (k == 1) gmp_printf("[old] j: %d | m: %d | V1: %Zd | poly: %Zd\n", j, m, V1, poly[j][m]);
        }
      }     
    }
  }
  m.end();
  if (i > 260) cout << "TIME: " << m.get_papi_elapsed_time() << endl;

  mpz_t** poly2 = new mpz_t*[nvars];
  for(int j=0; j < nvars; j++)
    alloc_init_vec(&poly2[j], 3);

  m2.begin_with_history();
  CMTSumCheckProver sc(c);
  sc.computePoly(poly2, i, z, ri, r);
  m2.end();
  if (i > 260) cout << "TIME2: " << m2.get_papi_elapsed_time() << endl;

  for(int j=0; j < nvars; j++)
    for (int m = 0; m < 3; m++)
    {
      if (mpz_cmp(poly[j][m], poly2[j][m]) != 0)
      {
        cout << "PANIC: j: " << j << " | m: " << m << endl;
        gmp_printf("old: %Zd | new: %Zd\n", poly[j][m], poly2[j][m]);
        exit(1);
      }
    }

  if (i == 64 && false) // DEBUG
  {
    for (int k = 0; k < 18; k++)
    {
      int j = 0;
      //gmp_printf("poly[%d-1][%d] = %Zd\n", j, k, poly[j-1][k]);
      gmp_printf("poly[%d][%d] = %Zd\n", j, k, poly[j][k]);
    }
    cout << endl;
  }

  //have verifier check that all of P's messages in the sum-check protocol for this level of the circuit are valid
  //t1=clock();
  modadd(tmp, poly[0][0], poly[0][1], c.prime);
  if (mpz_cmp(tmp, ri) != 0)
  {
    gmp_printf("poly[0][0]+poly[0][1] != ri\n");
    gmp_printf("poly[0][0] is: %Zd\n", poly[0][0]);
    gmp_printf("poly[0][1] is: %Zd\n", poly[0][1]);
    gmp_printf("ri is: %Zd\n", ri);
    gmp_printf("i is: %d\n", i);
    cout << endl;
  }

  mpz_t check;
  mpz_init(check);

  for (int j = 1; j < nvars; j++)
  {
    extrap(check, poly[j-1], 3, r[j-1], c.prime);
    modadd(tmp, poly[j][0], poly[j][1], c.prime);
    if (mpz_cmp(check, tmp) != 0)
    {
        gmp_printf("poly[j][0]+poly[j][1] != extrap.\n");
        gmp_printf("poly[j][0] is: %Zd\n", poly[j][0]);
        gmp_printf("poly[j][1] is: %Zd\n", poly[j][1]);
        gmp_printf("sum is: %Zd\n", tmp);
        gmp_printf("extrap is: %Zd\n", check);
        gmp_printf("j is: %d\n", j);
        cout << endl;
    }
  }
  //*ct+=clock()-t1;


  //finally check whether the last message extrapolates to f_z(r). In practice, verifier would
  //compute beta(z, r), add_i(r), and mult_i(r) on his own, and P would tell him what
  //V_{i+1}(omega1) and Vi+1(omega2) are. (Below, V1 is the true value of V_{i+1}(omega1) and V2
  //is the true value of Vi+1(omega2))
  evaluate_V_i(V2, mip1, nip1, c[i+1].gates, &r[mi+mip1], c.prime);
  // V2 = evaluate_V_i(mip1, nip1, c[i+1], r+mi+mip1);

  mpz_t fz, plus, mult, test_add, test_mult;
  mpz_init(fz);
  mpz_init(plus);
  mpz_init(mult);
  mpz_init(test_add);
  mpz_init(test_mult);

  modadd(plus, V1, V2, c.prime);
  modmult(mult, V1, V2, c.prime);
  
  //Have the verifier evaluate the add_i and mult_i polynomials at the necessary location
  c[i].add_fn(test_add, r, mi, mip1, ni, nip1, c.prime);
  c[i].mul_fn(test_mult, r, mi, mip1, ni, nip1, c.prime);

  // DEBUG if (i == 65) gmp_printf("test_mult: %Zd\ntest_add: %Zd\n", test_mult, test_add);
  //gmp_printf("i: %d\ntest_mult: %Zd\ntest_add: %Zd\n", i, test_mult, test_add);

  mpz_mul(tmp, test_mult, mult);
  mpz_mul(fz, c[i][0].betar, tmp);

  mpz_mul(tmp, test_add, plus);
  addmodmult(fz, c[i][0].betar, tmp, c.prime);

  // DEBUG gmp_printf("mult: %Zd\nadd: %Zd\nV2: %Zd\nV1: %Zd\n", mult, plus, V2, V1);

  //fz now equals the value f_z(r), assuming P truthfully provided V_{i+1}(omega1) and Vi+1(omega2) are.

  //compute the *claimed* value of fz(r) implied by P's last message, and see if it matches 
  extrap(check, poly[nvars-1], 3, r[nvars-1], c.prime);
  if (mpz_cmp(check, fz) != 0)
  {
    gmp_printf("fz != extrap\n");
    gmp_printf("fz is: %Zd\n", fz);
    gmp_printf("poly[nvars-1][0] is: %Zd\n", poly[nvars-1][0]);
    gmp_printf("poly[nvars-1][1] is: %Zd\n", poly[nvars-1][1]);
    gmp_printf("extrap is: %Zd\n", check);
    gmp_printf("i is: %d\n", i);
  }

  //now reduce claim that V_{i+1}(r1)=V1 and V_{i+1}(r2)=V2 to V_{i+1}(r3)=V3.
  //Let gamma be the line such that gamma(0)=r1, gamma(1)=r2
  //P computes V_{i+1)(gamma(0))... V_{i+1}(gamma(mip1))
  //t1=clock();

  mpz_t *lpoly, *point;
  alloc_init_vec(&lpoly, mip1 + 1);
  alloc_init_vec(&point, mip1);

  static mpz_t vec[2];
  mpz_init(vec[0]);
  mpz_init(vec[1]);

  for(int k = 0; k < mip1+1; k++)
  {
    for(int j = 0; j < mip1; j++)
    {
        mpz_set(vec[0], r[mi+j]);
        mpz_set(vec[1], r[mi+mip1+j]);
        extrap_ui(point[j], vec, 2, k, c.prime);
    }
    evaluate_V_i(lpoly[k], mip1, nip1, c[i+1].gates, point, c.prime);
  }

  if(mpz_cmp(V1, lpoly[0]) != 0)
  {
    gmp_printf("V1 != lpoly[0].\n");
    gmp_printf("V1 is: %Zd\n", V1);
    gmp_printf("lpoly[0] is: %Zd\n", lpoly[0]);
    cout << endl;
  }

  if(mpz_cmp(V2, lpoly[1]) != 0)
  {
    gmp_printf("V2 != lpoly[1].\n");
    gmp_printf("V2 is: %Zd\n", V2);
    gmp_printf("lpoly[1] is: %Zd\n", lpoly[1]);
    cout << endl;
  }

  for(int j = 0; j < mip1; j++)
  {
      mpz_set(vec[0], r[mi + j]);
      mpz_set(vec[1], r[mi + mip1 + j]);
      extrap_ui(z[j], vec, 2, 0, c.prime);
  }

  // In case rop is also an input
  mpz_t answer;
  mpz_init(answer);
  extrap_ui(answer, lpoly, mip1 + 1, 0, c.prime);
  mpz_set(rop, answer);

  free_vec(lpoly, mip1 + 1);
  free_vec(point, mip1);

  mpz_clear(answer);
  mpz_clear(betar);
  mpz_clear(betartimesV1);
  mpz_clear(addormultr);
  mpz_clear(V1);
  mpz_clear(V2);
  mpz_clear(tmp);
  mpz_clear(check);
  mpz_clear(fz);
  mpz_clear(plus);
  mpz_clear(mult);
  mpz_clear(test_add);
  mpz_clear(test_mult);
}

//evaluates the polynomial mult_d for the F2 circuit at point r
void
F2_mult_d(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
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
zero(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
{
    mpz_set_ui(rop, 0);
}

// 
void
check_wiring(mpz_t rop, mpz_t *r, int out, int in1, int in2, int mi, int mip1, mpz_t prime)
{
  mpz_t ans;
  mpz_init(ans);

  chi(ans, out, r, mi, prime);
  mul_chi(ans, in1, &r[mi], mip1, prime);
  mul_chi(ans, in2, &r[mi+mip1], mip1, prime);

  modadd(rop, rop, ans, prime);
  mpz_clear(ans);
}

void
check_equal(mpz_t rop, mpz_t *p, mpz_t *in1, mpz_t *in2, int start, int end, mpz_t prime)
{
  vector<mpz_t*> r;
  r.push_back(p);
  r.push_back(in1);
  r.push_back(in2);
  check_equal(rop, r, start, end, prime);
}

void
check_equal(mpz_t rop, mpz_t *p, mpz_t *in1, int start, int end, mpz_t prime)
{
  vector<mpz_t*> r;
  r.push_back(p);
  r.push_back(in1);
  check_equal(rop, r, start, end, prime);
}

void
check_equal(mpz_t rop, vector<mpz_t*>& r, int start, int end, mpz_t prime)
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

    vector<mpz_t*>::iterator it;
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
check_zero(mpz_t rop, mpz_t *p, mpz_t *in1, mpz_t *in2, int start, int end, mpz_t prime)
{
  vector<mpz_t*> r;
  r.push_back(p);
  r.push_back(in1);
  r.push_back(in2);
  check_zero(rop, r, start, end, prime);
}

void
check_zero(mpz_t rop, mpz_t *p, mpz_t *in1, int start, int end, mpz_t prime)
{
  vector<mpz_t*> r;
  r.push_back(p);
  r.push_back(in1);
  check_zero(rop, r, start, end, prime);
}

void
check_zero(mpz_t rop, vector<mpz_t*>& r, int start, int end, mpz_t prime)
{
  mpz_t tmp, ans;
  mpz_init(tmp);
  mpz_init_set_ui(ans, 1);

  for (int i = start; i < end; i++)
  {
    vector<mpz_t*>::iterator it;
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



//evaluates the polynomial add_i for any layer of the F2 circuit other than the d'th layer
void
reduce(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
{
  mpz_t temp, ans, tmp, tmp2;
  mpz_init(tmp);
  mpz_init(tmp2);
  mpz_init(temp);
  mpz_init_set_ui(ans, 1);

  //this checks that p=2omega1 and p=2omega2+1, ignoring the first bit of omega1 and omega2
  for(int i = 0; i < mi; i++)
  {
    mpz_mul(temp, r[mi+i+1], r[mi+mip1+i+1]);
    mpz_mul(temp, temp, r[i]);

    one_sub(tmp,  r[mi+i+1]);
    one_sub(tmp2, r[mi+mip1+i+1]);
    mpz_mul(tmp, tmp, tmp2);
    one_sub(tmp2, r[i]);
    mpz_addmul(temp, tmp, tmp2);

    modmult(ans, ans, temp, prime);
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
F0add_dp1to58pd(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
{
    mpz_set_ui(rop, 0);
}

//evaluates mult_i for the F0 circuit for any i between d+1 and 58+d
void
F0mult_dp1to58pd(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
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
F0mult_59pd(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
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
F0add_59pd(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
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
F0mult_d(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
{
  //all gates p have in1=2p, in2=2p+1
  reduce(rop, r, mi, mip1, ni, nip1, prime);
}

//evaluates the mult_i polynomial for layer 60+d of the F0 circuit
void
F0mult_60pd(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
{
  //all gates p but n-2 have in1=in2=p.
  F2_mult_d(rop, r, mi, mip1, ni, nip1, prime);
}


void
mat_add_63_p3d(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
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
mat_mult_63_p3d(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
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
mat_add_below_63_p3d(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
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
mat_add_61_p2dp1(mpz_t rop, mpz_t* r, int mi, int mip1, int ni, int nip1, mpz_t prime)
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
mat_add_61_p2d(mpz_t rop, mpz_t *r, int mi, int mip1, int ni, int nip1, mpz_t prime)
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
check_first_level(mpz_t rop, Circuit& c, mpz_t* r, mpz_t* zi, uint64 d)
{
  int num_effective = powi(2, c[d].logSize());
  mpz_t *vals;
  alloc_init_vec(&vals, num_effective);

  mpz_t** y = (mpz_t**) malloc(c[d].logSize() * sizeof(mpz_t*));
  mpz_t* a;
  alloc_init_vec(&a, c[d].size());

  for(int i = 0; i < c[d].size(); i++)
	mpz_set(a[i], c[d][i].val);

  mpz_t check, tmp;
  mpz_init_set_ui(check, 0);
  mpz_init(tmp);

  uint64 ct=0; //check time

  for(int i =0; i < c[d].logSize(); i++)
  {
      prng.get_random(r[i], c.prime);
      // r[i] = rand();
      // DEBUG mpz_set_ui(r[i], 76);
  }

  clock_t t=clock();
  for(int j=0; j < c[d].logSize(); j++) // computes all the messages from prover to verifier
  {
      alloc_init_vec(&y[j], 2);
      mpz_set_ui(y[j][0], 0);
      mpz_set_ui(y[j][1], 0);

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
}

