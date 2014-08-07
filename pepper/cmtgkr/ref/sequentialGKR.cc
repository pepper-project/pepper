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

#include <cstdio>

#include <cstdlib>
#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <math.h>

#define MASK 4294967295 //2^32-1
#define PRIME 2305843009213693951 //2^61-1
#define PROOF 1
#define LOG_PAT_LEN 3
#define PAT_MASK 7
#define PAT_LEN 8

typedef unsigned long long uint64;

using namespace std;

uint64 counter = 0;

//circuit data structure
typedef struct circ
{
  int* log_level_length; //needs to be ceiling of log_level_length (controls number of vars in various polynomials)
  uint64* level_length; //number of gates at each level of the circuit
  struct gate** gates; //two-d array of gates
  int num_levels; //number of levels of circuit
} circ;

typedef struct gate
{
  int type; //0 for add, 1 for mult, -1 for input gate, else this gate computes (in1)^type
  int in1; //index of first in-neighbor in the array gates[i-1]
  int in2; //index of second in-neighbor in the array gates[i-1]
  uint64 val; //gates value
  uint64 betar; //used for computing contribution of this gate to beta polynomial
  uint64 addormultr; //used for computing contribution of this get to add_i and mult_i polys
} gate;


void destroy_circ(circ* c)
{
  free(c->log_level_length);
  free(c->level_length);
  for(int i = 0; i < c->num_levels; i++)
    free(c->gates[i]);
  free(c->gates);
  free(c);
}

//computes x^b
uint64 myPow(uint64 x, uint64 b) 
{
  uint64 i = 1;
  for (int j = 0; j <b; j++)  i *= x;
  return i;
}


//efficient modular arithmetic function for p=2^61-1. Only works for this value of p.
//This function might
//return a number slightly greater than p (possibly by an additive factor of 8);
//It'd be cleaner to check if the answer is greater than p
//and if so subtract p, but I don't want to pay that
//efficiency hit here, so the user should just be aware of this.
uint64 myMod(uint64 x)
{
  return (x >> 61) + (x & PRIME);
}  

//efficient modular multiplication function mod 2^61-1
inline uint64 myModMult(uint64 x, uint64 y)
{

  
	uint64 hi_x = x >> 32;
  
	uint64 hi_y = y >> 32;

  
	uint64 low_x = x & MASK;
 
	uint64 low_y = y & MASK;

  

	//since myMod might return something slightly large than 2^61-1,
  
	//we need to multiply by 8 in two pieces to avoid overflow.
  

	uint64 piece1 = myMod((hi_x * hi_y)<< 3);

  

	uint64 z = (hi_x * low_y + hi_y * low_x);
  

	uint64 hi_z = z >> 32;
  

	uint64 low_z = z & MASK;

  

	//Note 2^64 mod (2^61-1) is 8
  

	uint64 piece2 = myMod((hi_z<<3) + myMod((low_z << 32)));

  

	uint64 piece3 = myMod(low_x * low_y);
  

	uint64 result = myMod(piece1 + piece2 + piece3);
  

	return result;

}

//computes b^e mod p using repeated squaring. p should be 2^61-1
uint64 myModPow(uint64 b, uint64 e)
{
  uint64 result;
  if(e==1) return b;
  if(e == 0) return 1;
  if((e & 1) == 0)
  {
    result = myModPow(b, (e>>1));
    return myModMult(result, result);
  }
  else
  {
     return myModMult(myModPow(b, e-1), b);
  }
}

//Performs Extended Euclidean Algorithm
//Used for computing multiplicative inverses mod p
void extEuclideanAlg(uint64 u, uint64* u1, uint64* u2, uint64* u3)
{
  *u1 = 1;
  *u2 = 0;
  *u3 = u;
  uint64 v1 = 0;
  uint64 v2 = 1;
  uint64 v3 = PRIME;
  uint64 q;
  uint64 t1;
  uint64 t2;
  uint64 t3;
  do
  {
    q = *u3 / v3;
    //t1 = *u1 + p - q * v1;
    //t2 = *u2 + p - q*v2;
    //t3 = *u3 + p - q*v3;
    t1 = myMod((*u1) + PRIME - myModMult(q, v1));
    t2 = myMod((*u2) + PRIME - myModMult(q, v2));
    t3 = myMod((*u3) + PRIME - myModMult(q, v3));
    (*u1) = v1;
    (*u2) = v2;
    (*u3) = v3;
    v1 = t1;
    v2 = t2;
    v3 = t3;
  }while(v3 != 0 && v3!= PRIME);
}


//Computes the modular multiplicative inverse of a modulo m,
//using the extended Euclidean algorithm
//only works for p=2^61-1
uint64 inv(uint64 a)
{
  uint64 u1;
  uint64 u2;
  uint64 u3;
  extEuclideanAlg(a, &u1, &u2, &u3);
  if(u3==1)
      return myMod(u1);
  else
      return 0;
}

//computes chi_v(r), where chi is the Lagrange polynomial that takes
//boolean vector v to 1 and all other boolean vectors to 0. (we view v's bits as defining
//a boolean vector. n is dimension of this vector.
//all arithmetic done mod p
uint64 chi(uint64 v, uint64* r, uint64 n)
{ 
  uint64 x=v;
  uint64 c = 1;
  for(uint64 i = 0; i <n; i++)
  {
    if( x&1 )
      c=myModMult(c, r[i]);
    else
      c=myModMult(c, 1+PRIME-r[i]);
    x=x>>1;
  }
  return c;
}

//evaluates beta_z polynomial (described in GKR08)
//at location k (the bits of k are interpreted as a boolean vector). mi is dimension of k and r.
uint64 evaluate_beta_z_ULL(int mi, uint64* z, uint64 k)
{
    return chi(k, z, mi);
  uint64 ans=1;
  uint64 x=k;
  for(int i = 0; i < mi; i++)
  {
    if(x&1)
      ans = myModMult(ans, z[i]);
    else
      ans = myModMult(ans, 1+PRIME-z[i]);
    x = x >> 1;
  }
  return ans;
}

//evaluates V_i polynomial at location r.
//Here V_i is described in GKR08; it is the multi-linear extension
//of the vector of gate values at level i of the circuit
uint64 evaluate_V_i(int mi, int ni, gate* level_i, uint64* r)
{
  uint64 ans=0;
  for(uint64 k = 0; k < ni; k++)
  {
     ans=myMod(ans + myModMult(level_i[k].val, chi(k, r, mi)));
  }
  return ans;
}

//extrapolate the polynomial implied by vector vec of length n to location r
uint64 extrap(uint64* vec, uint64 n, uint64 r)
{
  uint64 result=0;
  uint64 mult=1;
  for(uint64 i = 0; i < n; i++)
  {
    mult=1;
    for(uint64 j=0; j < n; j++)
    {
      if (i>j)
        mult=myModMult(myModMult(mult, myMod(r-j+PRIME)), inv(i-j) );
      if (i<j)
        mult=myModMult(myModMult(mult, myMod(r-j+PRIME)), inv(myMod(i+PRIME-j)) );
    }
    result=myMod(result+myModMult(mult, vec[i]));
  }
  return result;
}

//assuming the input level of circuit c contains the values of the input gates
//evaluate the circuit c, filling in the values of all internal gates as well.
void evaluate_circuit(circ* c)
{
  int d = c->num_levels-1;
  uint64 val1;
  uint64 val2;
  for(int i = d-1; i >= 0; i--)
  {
    for(int j = 0; j < c->level_length[i]; j++)
    {
      val1=c->gates[i+1][c->gates[i][j].in1].val;
      val2=c->gates[i+1][c->gates[i][j].in2].val;

      if(c->gates[i][j].type == 0)
		c->gates[i][j].val=myMod(val1+val2);
      else if(c->gates[i][j].type == 1)
		c->gates[i][j].val=myModMult(val1, val2);
      else
        c->gates[i][j].val=myModPow(val2, c->gates[i][j].type);

      gate *ropgate = &c->gates[i][j];
      if (i == i) {
      printf("i: %d | j: %d | ropgate.type: %d | ropgate.in1: %d | ropgate.in2: %d | ropgate: %Ld\n", i, j, ropgate->type, ropgate->in1, ropgate->in2, ropgate->val);
      }
    }
    cout << endl;
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
uint64 check_level(circ* c, int i, uint64* z, uint64 ri, 
int* com_ct, int* rd_ct, uint64* r, uint64** poly, uint64* vals, 
uint64 (*add_ifnc)(uint64* r, int mi, int mip1, int ni, int nip1), 
uint64 (*mult_ifnc)(uint64* r, int mi, int mip1, int ni, int nip1), int vstream)
{
  //Although GKR is stated assuming that all levels of the circuit have same number of
  //gates, our implementation does not require this. Thus, ni will be number of gates at level i
  //and nip1 will be number of gates at level i+1; mi and mip1 are ceil(log(ni or nip1)) respectively.
  //This yields significant speedups in practice.

  int mi=c->log_level_length[i];
  int ni=c->level_length[i];
  int mip1 = c->log_level_length[i+1];
  int nip1 = c->level_length[i+1];
  int nvars = mi+2*mip1; //number of variables being summed over in this iteration of GKR
 
    //printf("i: %d | mi: %d | mip1: %d | nvars: %d\n", i, mi, mip1, nvars);

  *com_ct = *com_ct + 3*nvars;
 
  //r is V's random coin tosses for this iteration.
  //really should choose r <--F_p, but this chooses r <-- [2^32]
  for(int j = 0; j < nvars; j++)
  {
    r[j] = counter++;//rand();
  }

  *rd_ct = *rd_ct + mi+2*mip1;

  //printf("i: %d | ri: %Ld\n", i, ri);
 
  //initialize betar and addormultr values for this iteration of protocol
  //these values are used to track each gate's contribution to beta_z, add_i, or mult_i polynomials
  uint64 k;

  for(k=0; k < ni; k++)
  {
    c->gates[i][k].betar = evaluate_beta_z_ULL(mi, z, k);
    c->gates[i][k].addormultr=1;
      // { int j = 0; if (j == 0 && i == i) printf("i: %d | j: %d | k: %d | c->gates[i][k].betar: %Lu\n", i, j, k, c->gates[i][k].betar); }

  }

  uint64 betar=0; //represents contribution to beta(z) by a given gate
  uint64 addormultr=0; //represents contribution to tilde{add} or tilde{mult} by a given gate
  uint64 V1=0; //represents contribution to V_{i+1}(omega1) by a given gate
  uint64 V2=0; //represents contribution to V_{i+1}(omega2) by a given gate

  int q;
  uint64 index;

  //initialize p array, which will store P's messages for this iteration of the protocol
  for(int k = 0; k < nvars; k++)
    for(int j = 0; j <= 2; j++)
      poly[k][j]=0;
  //num_effective is used for computing the necessary V(omega1) or V(omega2) values at any round of the protocol
  int num_effective = myPow(2, c->log_level_length[i+1]);
  uint64 kshiftj;

  //initialize vals array, which will store evaluate of Vi+1 at various points
  //intially, the protocol needs to know Vi+1 at all points in boolean hypercube
  //so that's what we initialize to.
  for(int k = 0; k < num_effective;k++)
  {
     if(k < nip1)
       vals[k]=c->gates[i+1][k].val;
     else
       vals[k]=0;
  }


  uint64 betartimesV1;
  //run through each round in this iteration, one round for each of the mi+2mip1 variables
  for(int j = 0; j < nvars; j++)
  {
    //if we're in a round > mi, we need to know evaluations of Vi+1 at some non-boolean locations
    //The following logic computes these values
    if(j > mi && (j != mi+mip1))
    { 
      for(k=0; k < num_effective; k++)
      {
        index = k >> 1;
        if( k & 1)
           vals[index]= myMod(vals[index] + myModMult(vals[k], r[j-1]));
        else
	   vals[index]=myModMult(vals[k], 1+PRIME-r[j-1]);
      }
      //num_effective tracks the number of points at which we need to evaluate Vi+1
      //for this round of the protocol (it halves every round, as one more variable becomes 'bound')
      num_effective=num_effective>>1;
    }
    if(j== mi+mip1)
    {
        num_effective = myPow(2, c->log_level_length[i+1]);
        for(int k = 0; k < num_effective;k++)
        {
           if(k < nip1)
             vals[k]=c->gates[i+1][k].val;
           else
             vals[k]=0;
        }
    	V1=evaluate_V_i(mip1, nip1, c->gates[i+1], r+mi);
    	betartimesV1=myModMult(betar, V1);
    }
    if(j==mi)
    {   //by round mi of this iteration, the first mi variables have been bound, 
        //so beta is fixed for the rest of this iteration
        betar=c->gates[i][0].betar;	
    }
    // if (i == 63 && j == 8) for (int k = 0; k < num_effective;k++) printf("vals[%d]: %Ld\n", k, vals[k]);

    
    //for each gate in level i of the circuit, compute its contribution to P's message in round j
    for(k=0; k < ni; k++)
    {
      //in cases where V1 or V2 are trivial to compute, compute them now
      if(j < mi)
      {
         V1 = c->gates[i+1][c->gates[i][k].in1].val;
         V2 = c->gates[i+1][c->gates[i][k].in2].val;
      }
      else if(j < mi+mip1)
      {
        V2 = c->gates[i+1][c->gates[i][k].in2].val;
      }

      //prep betar fields because in this round variable j will take on values 0, 1, or 2 rather than just 0,1
      if(j < mi)
      {
         kshiftj = k>>j;
         if( kshiftj & 1)
           c->gates[i][k].betar = myModMult(c->gates[i][k].betar, inv(z[j])); 
         else
           c->gates[i][k].betar = myModMult(c->gates[i][k].betar, inv(1+PRIME-z[j])); 
      }
      //now we iterate over the points at which we evaluate the polynomial for this round
      for(int m = 0; m <= 2; m++)
      {
        if(j< mi)
        {
    //if (j == 0 && i == i) printf("i: %d | j: %d | betar: %Lu\n", i, j, betar);

           //compute betar for this gate for this round, and update betar field if we're done with it for this round (m==2)
           if(m==0)
             betar = myModMult(c->gates[i][k].betar, myModMult(1+PRIME-z[j], 1+PRIME-m));
           else if(m==1)
             betar=myModMult(c->gates[i][k].betar, myModMult(z[j], m));
           else
             betar = myModMult(c->gates[i][k].betar, myMod(myModMult(1+PRIME-z[j], 1+PRIME-m) + myModMult(z[j], m)));
           if(m==2)
             c->gates[i][k].betar = myModMult(c->gates[i][k].betar,
                  		myMod(myModMult(1+PRIME-z[j], 1+PRIME-r[j]) + myModMult(z[j], r[j])));
          //compute addormult for this gate for this round, and update the field if we're done with it for this round (m==2)
          if(kshiftj & 1)
          {
            if(m==0) 
              continue;
            else
              addormultr=myModMult(c->gates[i][k].addormultr, m);
            if(m==2)
              c->gates[i][k].addormultr = myModMult(c->gates[i][k].addormultr, r[j]);
          }
          else
          {
            if(m==1)
              continue;
            else 
              addormultr=myModMult(c->gates[i][k].addormultr, 1+PRIME-m);
            if(m==2)
              c->gates[i][k].addormultr = myModMult(c->gates[i][k].addormultr, 1+PRIME-r[j]);
          }
        }

        if(j >= mi && j < mi+mip1)
        {
           //now compute contribution to V_i+1(omega1) for this gate
           index=c->gates[i][k].in1 >> (j-mi);
           if(index & 1)
           {
             if(m==0)
              continue; 
             else if(m==1)
              V1=vals[index];
             else
              V1=myMod(myModMult(vals[index], m)+myModMult(vals[index-1], PRIME+1-m));
              //now compute contribution to tilde{add}_i or tilde{multi}_i for this gate
             addormultr=myModMult(c->gates[i][k].addormultr, m);
             if(m==2)
               c->gates[i][k].addormultr = myModMult(c->gates[i][k].addormultr, r[j]);
           }
           else
           {
             if(m==1)
               continue;
             else if(m==0)
               V1=vals[index];
             else
               V1=myMod(myModMult(vals[index+1], m)+myModMult(vals[index], PRIME+1-m));
             //now compute contribution to tilde{add}_i or tilde{multi}_i for this gate
             addormultr=myModMult(c->gates[i][k].addormultr, 1+PRIME-m);
             if(m==2)
               c->gates[i][k].addormultr = myModMult(c->gates[i][k].addormultr, 1+PRIME-r[j]);
	   }  
        }
        else if(j >= mi+mip1)
        { 
          //compute contribution to V_{i+1}(omega2) for this gate
          index=c->gates[i][k].in2 >> (j-mi-mip1);
          //if ((j == 8) && (i == 63)) printf("m: %d | vals[idx]: %Ld\n", m, vals[index]);

          if(index & 1)
          {
            if(m==0)
              continue;
            if(m==1)
              V2=vals[index];
            else 
              V2=myMod(myModMult(vals[index], m)+myModMult(vals[index-1], PRIME+1-m));
            addormultr=myModMult(c->gates[i][k].addormultr, m);
            if(m==2)
              c->gates[i][k].addormultr = myModMult(c->gates[i][k].addormultr, r[j]);
           } 
           else
           {
             if(m==1)
               continue;
             if(m==0)
               V2=vals[index];
             else   
               V2=myMod(myModMult(vals[index+1], m)+myModMult(vals[index], PRIME+1-m));
             addormultr=myModMult(c->gates[i][k].addormultr, 1+PRIME-m);
             if(m==2)
               c->gates[i][k].addormultr = myModMult(c->gates[i][k].addormultr, 1+PRIME-r[j]);
           }

        if ((j == 8) && (i == 63) && false)
        {
            cout << "V1: " << V1 << endl;
            cout << "V2: " << V2 << endl;
            cout << "betar: " << betar << endl;
            cout << "addormultr: " << addormultr << endl;
        }

        }

        //finally, update the evaluation of this round's polynomial at m based on this gate's contribution
        //to beta, add_i or multi_i and V_{i+1}(omega1) and V_{i+1}(omega2)
        if(c->gates[i][k].type==0)
        {
          poly[j][m]=myMod(poly[j][m]+myModMult(betar, myModMult(addormultr, myMod(V1 + V2))));
        }
        else if(c->gates[i][k].type==1)
        {
           if(j<mi+mip1)
              poly[j][m]=myMod(poly[j][m]+myModMult(betar, myModMult(addormultr, myModMult(V1, V2))));
           else
              poly[j][m]=myMod(poly[j][m]+myModMult(betartimesV1, myModMult(addormultr, V2)));	
        }
        else
        { 
              poly[j][m]=myMod(poly[j][m]+myModMult(betar, 
                         myModMult(addormultr, myModPow(V2, c->gates[i][k].type))));
        }
      }     
    }
  }
 
  //have verifier check that all of P's messages in the sum-check protocol for this level of the circuit are valid
  //t1=clock();
  if( (myMod(poly[0][0]+poly[0][1]) != ri) && (myMod(poly[0][0]+poly[0][1]) != ri-PRIME) 
       && (myMod(poly[0][0]+poly[0][1]) != ri+PRIME))
  {
    cout << "poly[0][0]+poly[0][1] != ri. poly[0][0] is: " << poly[0][0] << " poly[0][1] is: ";
    cout << poly[0][1] << " ri is: " << ri << "i is: " << i << endl;
  }

  if (i == 64 && false)
  {
      for (int k = 0; k < 18; k++)
      {
          int j = 0;
        //cout << "poly[" << j << "-1][" << k << "] = " << poly[j-1][k] << "\n";
        cout << "poly[" << j << "][" << k << "] = " << poly[j][k] << "\n";
      }
      cout << endl;
  }

  uint64 check=0;
  for(int j =1; j < nvars; j++)
  {
    check=extrap(poly[j-1], 3, r[j-1]);
    if((check != myMod(poly[j][0]+poly[j][1])) && (check != (myMod(poly[j][0]+poly[j][1]) + PRIME))
        && (check != myMod(poly[j][0]+poly[j][1])-PRIME))
    {
      cout << "poly[j][0]+poly[j][1] != extrap. poly[j][0] is: " << poly[j][0] << " poly[j][1] is: ";
      cout << poly[j][1] << " extrap is: " << check << " j is: " << j << " i is: " << i << endl; 
    }
  }
  //*ct+=clock()-t1;
  //finally check whether the last message extrapolates to f_z(r). In practice, verifier would
  //compute beta(z, r), add_i(r), and mult_i(r) on his own, and P would tell him what
  //V_{i+1}(omega1) and Vi+1(omega2) are. (Below, V1 is the true value of V_{i+1}(omega1) and V2
  //is the true value of Vi+1(omega2))
  V2 = evaluate_V_i(mip1, nip1, c->gates[i+1], r+mi+mip1);

  uint64 fz=0;
  uint64 plus=myMod(V1+V2);
  uint64 mult=myModMult(V1, V2);

  uint64 test_add;
  uint64 test_mult;
  
  //Have the verifier evaluate the add_i and mult_i polynomials at the necessary location
  test_add = (*add_ifnc)(r, mi, mip1, ni, nip1);
  test_mult = (*mult_ifnc)(r, mi, mip1, ni, nip1);

  //if (i == 65) printf("test_mult: %Ld\ntest_add: %Ld\n", test_mult, test_add);

  fz=myModMult(c->gates[i][0].betar, myMod(myModMult(test_add, plus) + myModMult(test_mult, mult)));
  
  //fz now equals the value f_z(r), assuming P truthfully provided V_{i+1}(omega1) and Vi+1(omega2) are.
  
	if(vstream)
  	{
    	fz = 0;
  		for(k=0; k < ni; k++)
  		{
    		if(c->gates[i][k].type==0)
			{
      			fz=myMod(fz+myModMult(betar, myModMult(c->gates[i][k].addormultr, plus)));
			}
    		else
      		fz=myMod(fz+myModMult(betar, myModMult(c->gates[i][k].addormultr, mult)));
		}
	}

  //compute the *claimed* value of fz(r) implied by P's last message, and see if it matches 
  check=extrap(poly[nvars-1], 3, r[nvars-1]);  
  if( (check != fz) && (check != fz + PRIME) && (check != fz-PRIME))
  {
      cout << "fzaz != extrap.\n";
      cout << "fz is: " << fz << endl;
      cout << "poly[nvars-1][0] is: " << poly[nvars-1][0];
      cout << "\npoly[nvars-1][1] is: " << poly[nvars-1][1] << "\nextrap is: " << check << "\ni is: " << i << endl; 
  }

  //now reduce claim that V_{i+1}(r1)=V1 and V_{i+1}(r2)=V2 to V_{i+1}(r3)=V3.
  //Let gamma be the line such that gamma(0)=r1, gamma(1)=r2
  //P computes V_{i+1)(gamma(0))... V_{i+1}(gamma(mip1))
  //t1=clock();
  uint64* lpoly = (uint64*) calloc(mip1+1, sizeof(uint64));

  uint64* point=(uint64*) calloc(mip1, sizeof(uint64));

  static uint64 vec[2];

  for(int k = 0; k < mip1+1; k++)
  {
    for(int j = 0; j < mip1; j++)
    {
      vec[0]=r[mi+j];
      vec[1]=r[mi+mip1+j];
      point[j] = extrap(vec, 2, k);
    }
    lpoly[k]=evaluate_V_i(mip1, nip1, c->gates[i+1], point); 
  }

  if( (V1 != lpoly[0]) && (V1 != lpoly[0] + PRIME) && (V1 != lpoly[0]-PRIME))
    cout << "V1 != lpoly[0]. V1 is: " << V1 << " and lpoly[0] is: " << lpoly[0] << endl;
  if( (V2 != lpoly[1]) && (V2 != lpoly[1] + PRIME) && (V2 != lpoly[1]-PRIME))
    cout << "V2 != lpoly[1]. V2 is: " << V2 << " and lpoly[1] is: " << lpoly[1] << endl;

  uint64 t = myMod(0);
  for(int j = 0; j < mip1; j++)
  {
    vec[0]=r[mi+j];
    vec[1]=r[mi+mip1+j];
    z[j] = extrap(vec, 2, t);
  }

   uint64 answer= extrap(lpoly, mip1+1, t);
   free(point);
   free(lpoly);

   return answer;
}

//constructs the circuit computing F2, initializes the stream items too
//d is log of universe size
circ* construct_F2_circ(int d)
{                    
  uint64 n = myPow(2, d);
  circ* c = (circ*) malloc(sizeof(circ)); 
  c->level_length = (uint64*) calloc(d+2, sizeof(uint64));
  c->log_level_length = (int*) calloc(d+2, sizeof(int));
  c->gates = (gate**) calloc(d+2, sizeof(gate));
  c->num_levels=d+2;


  c->gates[d+1] = (gate*) calloc(n, sizeof(gate));
  uint64 size = n;

  c->level_length[d+1]=n;
  c->log_level_length[d+1]=d;

  uint64 ans = 0;
  for(uint64 j = 0; j < n; j++)
  {
     c->gates[d+1][j].val = rand() % 1000;
     c->gates[d+1][j].type=-1;
     ans+=c->gates[d+1][j].val *c->gates[d+1][j].val;
  }
  cout << "The correct second frequency moment is: " << ans << endl;
  
  for(int i = d; i >= 0; i--)
  { 
    c->gates[i] = (gate*) calloc(size, sizeof(gate));
    c->level_length[i]=size;
    c->log_level_length[i]=i;

    if(i == d)
    {
      for(uint64 j = 0; j < size; j++)
      {
        c->gates[i][j].in1 = c->gates[i][j].in2 = j;
        c->gates[i][j].type=1;
      }
    }
    else
    {
      for(uint64 j = 0; j < size; j++)
      {
        c->gates[i][j].in1=2*j;
        c->gates[i][j].in2=2*j+1;
        c->gates[i][j].type = 0;
      }
    }
    size=size/2;
  }
  return c;
}


//constuct circuit computing F0 using only + and * gates
circ* construct_F0_circ(int d)
{                    
  uint64 n = myPow(2, d);
  circ* c = (circ*) malloc(sizeof(circ)); 
  c->level_length = (uint64*) calloc(62+d, sizeof(uint64));
  c->log_level_length = (int*) calloc(62+d, sizeof(int));
  c->gates = (gate**) calloc(62+d, sizeof(gate));
  c->num_levels=62+d;
 
  //calloc all gates at leaves d+1 and up
  c->gates[61+d] = (gate*) calloc(n, sizeof(gate));
  c->gates[60+d] = (gate*) calloc(n, sizeof(gate));
   //set up input level
  c->level_length[61+d]=n;
  c->log_level_length[61+d]=d;
  c->level_length[60+d]=n;
  c->log_level_length[60+d]=d;
  
  for(int i = d+1; i <= 59+d; i++)
  {
    c->gates[i] = (gate*) calloc(2*n, sizeof(gate));
    c->level_length[i]=2*n;
    c->log_level_length[i]=d+1;
  }

  //c->gates[61+d][n-2].val=0;//need a constant 0 gate. So universe really has size 2^d-1, not 2^d
  c->gates[61+d][n-1].val=0;//need a constant 0 gate. So universe really has size 2^d-1, not 2^d

  int ans=0;
  for(uint64 j = 0; j < n-1; j++)
  {
     c->gates[61+d][j].val = rand() % 30;
     if(c->gates[61+d][j].val > 0)
       ans++;
  }

  cout << "real number of distinct items is: " << ans;
  //input level now set up

  //set up x^2 level
  for(uint64 k = 0; k < n; k++)
  { 
    c->gates[60+d][k].in1=c->gates[60+d][k].in2=k;
    c->gates[60+d][k].type=1;
  }
  //set up x^4 level
  for(uint64 k = 0; k < n; k++)
  {
    c->gates[59+d][2*k].in1=c->level_length[60+d]-1;
    c->gates[59+d][2*k].in2=k;
    c->gates[59+d][2*k].type=0;
    c->gates[59+d][2*k+1].in1=k;
    c->gates[59+d][2*k+1].in2=k;
    c->gates[59+d][2*k+1].type=1;
  }

  //implement gates to compute x^(p-1) for each input x
  //level_length and log_level length already set up for these levels
  for(uint64 i=d+1; i < 59+d; i++)
  {
    for(uint64 k = 0; k < n; k++)
    {
      c->gates[i][2*k].in1=2*k;
      c->gates[i][2*k].in2=2*k+1;
      c->gates[i][2*k].type=1;
   
      c->gates[i][2*k+1].in1=2*k+1;
      c->gates[i][2*k+1].in2=2*k+1;
      c->gates[i][2*k+1].type=1;
    }
  }
  
  //set up level d
  c->gates[d] = (gate*) calloc(n, sizeof(gate));
  c->level_length[d]=n;
  c->log_level_length[d]=d;

  for(uint64 k = 0; k < n; k++)
  {
    c->gates[d][k].in1 = 2*k;
    c->gates[d][k].in2 = 2*k+1;
    c->gates[d][k].type=1;
  }
  
  //set up levels 0 to d-1
  uint64 size = n/2;
  
  for(long long i =d-1; i >= 0; i--)
  {
    c->level_length[i]=size;
    c->log_level_length[i]=i;
    c->gates[i] = (gate*) calloc(size, sizeof(gate));
    for(uint64 k = 0; k < size; k++)
    {
        c->gates[i][k].in1=2*k;
        c->gates[i][k].in2=2*k+1;
        c->gates[i][k].type = 0;
    }
    size=size/2;
  }
  return c;
}

//constuct circuit computing F0 using only + and * gates

circ* construct_mat_circ(int d)

{ 

  clock_t t = clock();

  clock_t start = clock();

  int n = myPow(2, d);

  int n_squared = n*n;

  int n_cubed = n*n*n;

  int two_pow_d = n;

  circ* c = (circ*) malloc(sizeof(circ)); 

  c->level_length = (uint64*) calloc(64+3*d, sizeof(uint64));

  c->log_level_length = (int*) calloc(64+3*d, sizeof(int));

  c->num_levels=64+3*d;

  c->gates = (gate**) calloc(64+3*d, sizeof(gate));

  std::cout << "time to allocate memory in construction function is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;


  c->level_length[63+3*d] = 3*n_squared+1;
  c->gates[63+3*d] = (gate*) calloc((3*n_squared+1), sizeof(gate));

  c->log_level_length[63+3*d] = ceil(log((double)c->level_length[63+3*d])/log((double) 2));

  t = clock();

  //set up input level

  c->gates[63+3*d][3*n_squared].val=0;//need a constant 0 gate. So universe really has size 2^d-1, not 2^d

  for(int i = 0; i < 2*n_squared; i++)
  	c->gates[63+3*d][i].val = 1; //rand() % 10;

   std::cout << "time to randomly generate data is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;

  t=clock();

  for(int i = 0; i < n; i++)
  {
	  for(int j = 0; j < n; j++)
	  {
		  for(int k = 0; k < n; k++)

		  {

			  c->gates[63+3*d][2*n_squared+i*n+j].val = myMod( c->gates[63+3*d][2*n_squared+i*n+j].val 
			  + myModMult(c->gates[63+3*d][i*n+k].val, c->gates[63+3*d][n_squared+k*n+j].val));

			  //c->val[2*n_squared+i*n+j] = myMod( c->val[2*n_squared+i*n+j] +  myModMult(c->val[i*n+k], c->val[n_squared+k*n+j]));
		  }

		   //std::cout << "Setting C_" << i << j << "to: -" << c->val[2*n_squared + i*n + j];

		   c->gates[63+3*d][2*n_squared+i*n+j].val = myModMult(c->gates[63+3*d][2*n_squared+i*n+j].val, PRIME-1);
	  }
  }

  std::cout << "time to compute matvec unverifiably is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;

  c->level_length[62+3*d] = n_cubed + n_squared + 1;

  c->log_level_length[62+3*d] = 3*d + 1;
  
  c->gates[62+3*d] = (gate*) malloc(c->level_length[62+3*d] * sizeof(gate));
  
  for(int index = 0; index < c->level_length[62+3*d]- n_squared - 1; index++)
  {
  	c->gates[62+3*d][index].type = 1;
  	int k = (index & (two_pow_d-1));
	int i = ((index >> (2*d)) & (two_pow_d-1));
	int j = ((index >> d) & (two_pow_d-1));

	c->gates[62+3*d][index].in1 = ((i << d) + k);
	c->gates[62+3*d][index].in2 = ((1 << 2*d) + (k << d) + j);
  }
  for(int index = c->level_length[62+3*d]- n_squared - 1; index < c->level_length[62+3*d]; index++)
  {
  	c->gates[62+3*d][index].type = 0;
  	c->gates[62+3*d][index].in1 = c->level_length[62+3*d+1] - (c->level_length[62+3*d]-index);
  	c->gates[62+3*d][index].in2 = c->level_length[62+3*d+1] - 1;
  }
  	
  //cout << "after 62+3*d\n";
  int step=1;

  for(int j = 61+3*d; j >=62+2*d; j--)

  {
	  //cout << "at level " << j << endl;
	  c->level_length[j] = n_cubed/myPow(2, step) + n_squared + 1;

	  c->log_level_length[j] = ceil(log((double) c->level_length[j])/log((double)2));
	  
	  c->gates[j] = (gate*) malloc(c->level_length[j] * sizeof(gate));

	  for(int k = 0; k < c->level_length[j] - n_squared-1; k++)
	  {
	  	c->gates[j][k].in1 = 2*k;
	  	c->gates[j][k].in2 = 2*k+1;
	  	c->gates[j][k].type = 0;
	  }
	  for(int k = c->level_length[j] - n_squared-1; k < c->level_length[j]; k++)
	  {
	  	c->gates[j][k].in1 = c->level_length[j+1] - (c->level_length[j] - k);
	  	c->gates[j][k].in2 = c->level_length[j+1]-1;
	  	c->gates[j][k].type = 0;;
	  }

	  step++;

  }


  //cout << "starting 61+2*d\n";
  c->level_length[61+2*d]=n_squared+1;
  c->gates[61+2*d] = (gate*) malloc( (n_squared+1) * sizeof(gate));
  
  for(int index = 0; index < (n_squared + 1); index++)
  {
  	c->gates[61+2*d][index].type = 0;
  	c->gates[61+2*d][index].in1 = index;
  	c->gates[61+2*d][index].in2 = index+n_squared;
  }
  c->gates[61+2*d][n_squared].in1 = c->gates[61+2*d][n_squared].in2 = c->level_length[61+2*d+1]-1;
  
  //cout << "starting 60+2*d\n";

  //set up level just below input
  c->level_length[60+2*d]=(n_squared + 1);
  c->gates[60+2*d] = (gate*) malloc( (n_squared+1) * sizeof(gate));
  
 //squaring layer
  for(int k = 0; k < c->level_length[60+2*d]; k++)
  {
  	c->gates[60+2*d][k].type = 1;
  	c->gates[60+2*d][k].in1 = c->gates[60+2*d][k].in2 = k;
  }

  //c->log_level_length[60+2*d]=ceil(log((double) c->level_length[60+2*d])/log((double)2));

  //set up level two levels below input
  c->level_length[59+2*d]=2*(n_squared + 1);
  c->gates[59+2*d] = (gate*) malloc( c->level_length[59+2*d] * sizeof(gate));
  //c->log_level_length[59+2*d]=ceil(log((double) c->level_length[59+2*d])/log((double)2));

 
  for(int k = 0; k < c->level_length[59+2*d]; k++)
  {
  	c->gates[59+2*d][k].type = (k&1);
  	if( (k & 1) == 0)
  		c->gates[59+2*d][k].in1 = c->level_length[60+2*d]-1;
  	else
  		c->gates[59+2*d][k].in1 = (k>>1);
  	
  	c->gates[59+2*d][k].in2 = (k>>1);
  }



  for(int i = 2*d+1; i < 59+2*d; i++)

  {
	  //cout << " on layer " << i << endl;
    c->level_length[i]=2*(n_squared + 1);
	c->gates[i] = (gate*) malloc(2*(n_squared + 1) * sizeof(gate));

	//implement gates to compute x^(p-1) for each input x
  
	 for(int k = 0; k < c->level_length[i]; k++)
  	 {
	  	c->gates[i][k].type = 1;

	  	c->gates[i][k].in1 = k;

		if(k&1)
	  		c->gates[i][k].in2 = k;
	  	else
	  		c->gates[i][k].in2 = k+1;
  	 }
  }


    //set up level 2*d

    c->level_length[2*d]=n*n;

    //c->log_level_length[2*d]=ceil((double) log((double)c->level_length[d])/log((double)2));
	c->gates[2*d] = (gate*) malloc(n*n*sizeof(gate));
	
	for(int k = 0; k < c->level_length[2*d]; k++)
	{
		c->gates[2*d][k].type=1;
		c->gates[2*d][k].in1 = 2*k;
		c->gates[2*d][k].in2=2*k+1;
	}
  

  //set up levels 0 to d-1

  uint64 size = n_squared/2;

  for(long long i = 2*d-1; i >= 0; i--)

  {
    c->level_length[i]=size;
	c->gates[i] = (gate*) malloc(size * sizeof(gate));

    c->log_level_length[i]=i;

    
	for(int k = 0; k < c->level_length[i]; k++)
	{
		c->gates[i][k].type=0;
		c->gates[i][k].in1=2*k;
		c->gates[i][k].in2 = 2*k+1;
	}

	size=size >> 1;

  }

  for(int i = 0; i < c->num_levels; i++)

  {
	  c->log_level_length[i]=ceil((double) log((double)c->level_length[i])/log((double)2));

  }

   std::cout << "total time in construction is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC << std::endl;

  return c;

}

//constuct circuit computing Pattern Matching
//Note: the output of this circuit will be # of places pattern does *not* occur at, minus 2
//Since I chose the pattern to match the last PAT_LEN symbols of the input, and wired the circuit to
//ensure that the n'th location evalutes to 0 too
circ* construct_pm_circ(int d)
{ 
  
	clock_t start = clock();
  
	int n = myPow(2, d);
  
	int t = myPow(2, LOG_PAT_LEN); 
  
  
	circ* c = new circ; 
  
	c->level_length = (uint64*) calloc(64+d+LOG_PAT_LEN, sizeof(uint64));
  
	c->log_level_length = (int*) calloc(64+d+LOG_PAT_LEN, sizeof(int));
  
	c->num_levels=64+d+LOG_PAT_LEN;

  
	c->gates = (gate**) calloc(c->num_levels, sizeof(gate));
  
 
	
	c->level_length[63+d+LOG_PAT_LEN]=n+t;
  
	c->gates[63+d+LOG_PAT_LEN] = (gate*) malloc(c->level_length[63+d+LOG_PAT_LEN] * sizeof(gate));
  
  
	for(int i = 0; i <  n; i++)
		c->gates[63+d+LOG_PAT_LEN][i].val = rand() % 10;
  
		
	for(int i = n; i < n+PAT_LEN; i++)
	{
		c->gates[63+d+LOG_PAT_LEN][i].val = myModMult(PRIME-1, c->gates[63+d+LOG_PAT_LEN][i - PAT_LEN].val);
	}
  
	//done generating input. pattern should appear at end of input

  
	
	//set up addition layer connecting (i, j) to t_j
  
	c->level_length[62+d+LOG_PAT_LEN]=n*t;
  
	c->gates[62+d+LOG_PAT_LEN] = (gate*) malloc(c->level_length[62+d+LOG_PAT_LEN] * sizeof(gate));

  
	for(int i = 0; i < c->level_length[62+d+LOG_PAT_LEN]; i++)
	{
		c->gates[62+d+LOG_PAT_LEN][i].type = 0;
		int i1 = i >> LOG_PAT_LEN;
	  
		int j1 = i & PAT_MASK;
	  
		if(i1 != n-1)
		{
			c->gates[62+d+LOG_PAT_LEN][i].in1 = i1+j1;
			c->gates[62+d+LOG_PAT_LEN][i].in2 = n+j1;
		}
		else
		{
			c->gates[62+d+LOG_PAT_LEN][i].in1 = n-1;
			c->gates[62+d+LOG_PAT_LEN][i].in2 = n-1+PAT_LEN;
		}
	}

	//set up squaring layer
  
	c->level_length[61+d+LOG_PAT_LEN]=n*t;
   
	c->gates[61+d+LOG_PAT_LEN]= (gate*) malloc(c->level_length[61+d+LOG_PAT_LEN]*sizeof(gate));

	for(int i = 0; i < c->level_length[61+d+LOG_PAT_LEN]; i++)
	{
		c->gates[61+d+LOG_PAT_LEN][i].type = 1;
		c->gates[61+d+LOG_PAT_LEN][i].in1 = c->gates[61+d+LOG_PAT_LEN][i].in2 = i;
	}

	//create binary tree of sum gates to finish compute I_i
	//bottom of tree used to have length n+1, but changed to n, 
	//since last gate will be 0 by design of sticking pattern at end of input
  
	int blowup = 1;
  
	for(int j = 0; j < LOG_PAT_LEN; j++)
	{
		c->level_length[61+d+j] = blowup*n;
		c->gates[61+d+j] = (gate*) malloc(c->level_length[61+d+j] * sizeof(gate));
	
		for(int i = 0; i < c->level_length[61+d+j]; i++)
		{
			c->gates[61+d+j][i].type = 0;
			c->gates[61+d+j][i].in1 = 2*i;
			c->gates[61+d+j][i].in2 = 2*i+1;
		}
		blowup = blowup * 2;
	}

	//used to be n+1, changed to n. 60+d is the squaring level for F0 computation
  
	c->level_length[60+d]=n;
  
	c->gates[60+d] = (gate*) malloc(c->level_length[60+d] * sizeof(gate));
  
	for(int i = 0; i < c->level_length[60+d]; i++)
	{
		c->gates[60+d][i].type = 1; 
		c->gates[60+d][i].in1=c->gates[60+d][i].in2 = i;
	}

	//used to be 2*(n+1), changed to 2*n
  
	for(int i = d+1; i <= 59+d; i++)
	{
		c->level_length[i]=2*n;
		c->gates[i] = (gate*) malloc(c->level_length[i] * sizeof(gate));
	}
  
	//set up x^4 level
  for(uint64 k = 0; k < n; k++)
  {
    c->gates[59+d][2*k].in1=c->level_length[60+d]-1;
    c->gates[59+d][2*k].in2=k;
    c->gates[59+d][2*k].type=0;
    c->gates[59+d][2*k+1].in1=k;
    c->gates[59+d][2*k+1].in2=k;
    c->gates[59+d][2*k+1].type=1;
  }

  //implement gates to compute x^(p-1) for each input x
  //level_length and log_level length already set up for these levels
  for(uint64 i=d+1; i < 59+d; i++)
  {
    for(uint64 k = 0; k < n; k++)
    {
      c->gates[i][2*k].in1=2*k;
      c->gates[i][2*k].in2=2*k+1;
      c->gates[i][2*k].type=1;
   
      c->gates[i][2*k+1].in1=2*k+1;
      c->gates[i][2*k+1].in2=2*k+1;
      c->gates[i][2*k+1].type=1;
    }
  }
  
  //set up level d
  c->gates[d] = (gate*) calloc(n, sizeof(gate));
  c->level_length[d]=n;
  c->log_level_length[d]=d;

  for(uint64 k = 0; k < n; k++)
  {
    c->gates[d][k].in1 = 2*k;
    c->gates[d][k].in2 = 2*k+1;
    c->gates[d][k].type=1;
  }
  
  //set up levels 0 to d-1
  uint64 size = n/2;
  
  for(long long i =d-1; i >= 0; i--)
  {
    c->level_length[i]=size;
    c->log_level_length[i]=i;
    c->gates[i] = (gate*) calloc(size, sizeof(gate));
    for(uint64 k = 0; k < size; k++)
    {
        c->gates[i][k].in1=2*k;
        c->gates[i][k].in2=2*k+1;
        c->gates[i][k].type = 0;
    }
    size=size/2;
  }
  
  for(int i = 0; i < c->num_levels; i++)
  {
	  c->log_level_length[i]=ceil((double) log((double)c->level_length[i])/log((double)2));
  }
  
  return c;
}

//evaluates the polynomial mult_d for the F2 circuit at point r
uint64 F2_mult_d(uint64* r, int mi, int mip1, int ni, int nip1)
{
  uint64 ans = 1;
  uint64 temp = 0;
  for(int i = 0; i < mi; i++)
  {
     temp = myModMult(r[i], myModMult(r[mi+i], r[mi+mip1+i]));
     temp = myMod(temp + myModMult(1+PRIME-r[i], myModMult(1+PRIME-r[mi+i], 1+PRIME-r[mi+mip1+i])));
     ans = myModMult(ans, temp);
  }
  return ans;
}

//this function is used for add_i and mult_i polynomials that are identically zero
//(i.e. no gates of a certain type appear at level i of the circuit)
uint64 zero(uint64* r, int mi, int mip1, int ni, int nip1)
{
  return 0;
}

//evaluates the polynomial add_i for any layer of the F2 circuit other than the d'th layer
uint64 F2_add_notd(uint64* r, int mi, int mip1, int ni, int nip1)
{
  uint64 temp=0;
  uint64 ans = 1;
  //this checks that p=2omega1 and p=2omega2+1, ignoring the first bit of omega1 and omega2
  for(int i = 0; i < mi; i++)
  {
    temp = myModMult(r[i], myModMult(r[mi+i+1], r[mi+mip1+i+1]));
    temp = myMod(temp + myModMult(1+PRIME-r[i], myModMult(1+PRIME-r[mi+i+1], 1+PRIME-r[mi+mip1+i+1])));
    ans = myModMult(ans, temp);
  }
  //finally check that first bit of omega1=0 and first bit of omega2=1
  ans = myModMult(ans, 1+PRIME-r[mi]);
  ans = myModMult(ans, r[mi+mip1]);
  return ans;
}

uint64 F0add_dp1to58pd(uint64* r, int mi, int mip1, int ni, int nip1)
{
  return 0;
}

//evaluates mult_i for the F0 circuit for any i between d+1 and 58+d
uint64 F0mult_dp1to58pd(uint64* r, int mi, int mip1, int ni, int nip1)
{
  //even p's are connected to p and p+1, odd ps are connected to p and p
  
  //first handle even p contribution
  //first term in product makes sure p is even
  uint64 ans1 = 1+PRIME-r[0];
  uint64 temp;
  //now make sure all but least significant bits of in1 and in2 match p
  for(int i = 1; i < mi; i++)
  {
    temp = myModMult(r[i], myModMult(r[mi+i], r[mi+mip1+i]));
    temp = myMod(temp + myModMult(1+PRIME-r[i], myModMult(1+PRIME-r[mi+i], 1+PRIME-r[mi+mip1+i])));
    ans1 = myModMult(ans1, temp);
  }
  //finally check that least significant bit of in1 is 0 and lsb of in2 is 1
  ans1=myModMult(ans1, myModMult(1+PRIME-r[mi], r[mi+mip1]));

  //subtract contribution of last even gate
  //uint64 temp1 = chi(ni-1, r, mi, 0);
  //uint64 temp2 = chi(ni-1, r, mip1, mi);
  //uint64 temp3 = chi(ni, r, mip1, mi+mip1);

  //ans1 = myMod(ans1 + PRIME - myModMult(temp1, myModMult(temp2, temp3)));

  //now handle odd p contribution
  uint64 ans2 = r[0];
  //now make sure all but least significant bits of in1 and in2 match p
  for(int i = 1; i < mi; i++)
  {
    temp = myModMult(r[i], myModMult(r[mi+i], r[mi+mip1+i]));
    temp = myMod(temp + myModMult(1+PRIME-r[i], myModMult(1+PRIME-r[mi+i], 1+PRIME-r[mi+mip1+i])));
    ans2 = myModMult(ans2, temp);
  }
  //finally check that least significant bit of in1 and in2 are 1
  ans2=myModMult(ans2, myModMult(r[mi], r[mi+mip1]));

  //subtract contribution of last odd gate
  //temp1 = chi(ni, r, mi, 0);
  //temp2 = chi(ni, r, mip1, mi);
  //temp3 = chi(ni, r, mip1, mi+mip1);

  //ans2 = myMod(ans2 + PRIME - myModMult(temp1, myModMult(temp2, temp3)));

  
  return myMod(ans1 + ans2);
}

//evaluates mult_i polynomial for layer 59+d of the F0 circuit
uint64 F0mult_59pd(uint64* r, int mi, int mip1, int ni, int nip1)
{
   //odds ps go to p/2 for both in1 and in2
   uint64 ans = r[0];
   uint64 temp;
   //now make sure p>>1 matches in1 and in2
   for(int i = 0; i < mip1; i++)
   {
    temp = myModMult(r[i+1], myModMult(r[mi+i], r[mi+mip1+i]));
    temp = myMod(temp + myModMult(1+PRIME-r[i+1], myModMult(1+PRIME-r[mi+i], 1+PRIME-r[mi+mip1+i])));
    ans = myModMult(ans, temp);
   }
   return ans;
}

//evaluates add_i polynomial for layer 59+d of the F0 circuit
uint64 F0add_59pd(uint64* r, int mi, int mip1, int ni, int nip1)
{
   //even ps go to nip1-1 for in1 and p/2 for both in2
   uint64 ans1 = 1+PRIME-r[0];
   uint64 temp;
   //now make sure p>>1 matches in2
   for(int i = 0; i < mip1; i++)
   {
    temp = myModMult(r[i+1], r[mi+mip1+i]);
    temp = myMod(temp + myModMult(1+PRIME-r[i+1], 1+PRIME-r[mi+mip1+i]));
    ans1 = myModMult(ans1, temp);
   }

   //make sure in1 matches nip1-1
   ans1 = myModMult(ans1, chi(nip1-1, r+mi, mip1));
   return ans1;
}

//evaluates the mult_i polynomial for the d'th layer of the F0 circuit
uint64 F0mult_d(uint64* r, int mi, int mip1, int ni, int nip1)
{
  //all gates p have in1=2p, in2=2p+1
  uint64 ans = F2_add_notd(r, mi, mip1, ni, nip1);
  
  return ans;
}

//evaluates the mult_i polynomial for layer 60+d of the F0 circuit
uint64 F0mult_60pd(uint64* r, int mi, int mip1, int ni, int nip1)
{
  //all gates p but n-2 have in1=in2=p.
  uint64 ans = F2_mult_d(r, mi, mip1, ni, nip1);
  
  return ans;
}

uint64 mat_add_63_p3d(uint64* r, int mi, int mip1, int ni, int nip1)
{
	int d=(mi-1)/3;
	
	//with the exception of the final gate at this level, everything should be of the following form
	//z=(j, i, 0, 1) where j, i, and 0 are d bits long and 1 is just 1 bit
	//in1 = (j, i, 0, 1) where j and i are d bits long, and 0 and 1 are just 1 bit each
	//in2 = nip1-1 
	
	//test that the high order bit of z is 1
	uint64 ans = r[mi-1]; 
	
	//test that the next d highest order bits of z are 0
	for(int j = 0; j < d; j++) 
	{
		ans=myModMult(ans, 1+PRIME-r[2*d+j]);
	}
	
	//test that the highest-order 2 bits of in1 are 0, 1 (highest-order bit is 1)

	ans = myModMult(ans, myModMult(1+PRIME-r[mi+2*d], r[mi+2*d+1]));
	
	//test that the lowest order 2d bits of in1 are (j, i)
	for(int j = 0; j < 2*d; j++)
	{ //test that in1 is the i'th entry of C, where z is n^3+i
		ans = myModMult(ans, myMod( myModMult(r[mi+j], r[j]) +
								myModMult(1+PRIME-r[mi+j], 1+PRIME-r[j])));
	}
	
	//test that in2 equals nip1-1
	
	uint64 y = nip1-1;
	
	for(int j = 0; j < mip1; j++) 
	{ //test that in2 is nip1-1
		if((y >> j) & 1)
		{
			ans = myModMult(ans, r[mi+mip1+j]);
		}
		else
		{
			ans = myModMult(ans, 1+PRIME-r[mi+mip1+j]);
		}
	}
	
	//handle the case where z=ni-1, in1=nip1-1, in2=nip1-1
	uint64 z = ni-1;
	uint64 in1= nip1-1;
	uint64 in2= nip1-1;
	

	uint64 temp=1;
	for(int j = 0; j< mi; j++)
	{
		if( (z >> j) & 1)
		{
			temp = myModMult(temp, r[j]);
		}
		else
		{
			temp = myModMult(temp, 1+PRIME-r[j]);
		}
	}
	for(int j = 0; j < mip1; j++)
	{
		if( (in1 >> j) & 1)
		{
			temp = myModMult(temp, r[mi+j]);
		}
		else
		{	
			temp = myModMult(temp, 1+PRIME - r[mi+j]);
		}
	}
	for(int j = 0; j < mip1; j++)
	{
		if( (in2 >> j) & 1)
		{
			temp = myModMult(temp, r[mi+mip1+j]);
		}
		else
		{	
			temp = myModMult(temp, 1+PRIME - r[mi+mip1+j]);
		}
	}
	return myMod(ans + temp);
}

uint64 mat_mult_63_p3d(uint64* r, int mi, int mip1, int ni, int nip1)
{
	int d=(mi-1)/3;

	
	uint64 ans = (PRIME + 1 - r[mi-1]); //make sure z<=n^3-1


	//test that high order two bits in in1 are 0 (i.e. you're looking at an entry of A < n^2)
	ans = myModMult( ans, myModMult((PRIME + 1 - r[mi+mip1-1]), (PRIME + 1 - r[mi+mip1-2]))); 
	//make sure in2 >= n^2, <= 2n^2 (second highest bit is 1, highest bit is 0)
	ans = myModMult(ans, myModMult(PRIME + 1 - r[mi+2*mip1-1], r[mi+2*mip1-2]));
	
	//make sure in1 = (k, i, 0, 0), where z=(k, j, i, 0). This loop tests i
	for(int j = 0; j < d; j++)
	{
		//mi+2*d+j is pulling out high order d bits of z, mi+d+j is pulling out second d bits of in1
		ans = myModMult(ans, myMod(myModMult(r[2*d+j], r[mi+d+j]) + myModMult(1+PRIME-r[2*d+j], 1+PRIME - r[mi+d+j])));
	}
	//make sure in1 = (k, i, 0, 0) and in2=(j, k, 1, 0). This loop tests k for both
	for(int j = 0; j < d; j++)
	{
		//2*d+j is pulling k out of z, mi+j is pulling first d bits out of in1, 
		//and mi+mip1+d+j is pulling second d bits out of in2
		ans = myModMult(ans, myMod(myModMult(r[j], myModMult(r[mi+j], r[mi+mip1+d+j])) +
			myModMult(1+ PRIME - r[j], myModMult(1 + PRIME - r[mi+j], 1 + PRIME - r[mi+mip1+d+j]))));
	}
	//make sure in2 = (j, k, 1, 0). This loop tests j
	for(int j = 0; j < d; j++)
	{
		ans = myModMult(ans, myMod(myModMult(r[d+j], r[mi + mip1  + j]) + 
					 myModMult(1+PRIME-r[d+j], 1+PRIME - r[mi+ mip1 + j])));
	}

	return ans;
}

// Calculates log2 of number.  
double Log2( double n )  
{  
    // log(n)/log(2) is log2.  
    return log( (double) n ) / log( (double) 2 );  
}


uint64 mat_add_below_63_p3d(uint64* r, int mi, int mip1, int ni, int nip1)
{
	int three_d_min_i = mi-1;
	int n_squared = ni-myPow(2, mi-1)-1;
	int d = floor(Log2((double) n_squared)/2 + 0.5);
	
	//threshold in wiring structure is at 3d-i (gate 2^{3d-i} is first entry of C)
	int i = (three_d_min_i - 3*d) * (-1);
	
	//all gates at this layer are of the form z=(anything, 0), in1=(2z, 0), in2=(2z+1, 0) where anything is 3d-i bits
	//or z=(anything, (d-i zeros), 1) in1=(anything, (d-i+1 zeros), 1), in2=nip1-1

	//check high order bit of z is 0, and same with high order bit of in1 and in2
	uint64 ans = myModMult(1 + PRIME - r[mi-1], myModMult( 1 + PRIME - r[mi+mip1-1], 1 + PRIME - r[mi+2*mip1-1]));
	
	 uint64 temp=0;

  	//this checks that z=2omega1 ignoring high order bit of each and z=2omega2+1, ignoring the low-order bit of omega1 and omega2
  	for(int j = 0; j < mi-1; j++)
  	{
    	temp = myModMult(r[j], myModMult(r[mi+j+1], r[mi+mip1+j+1]));
    	temp = myMod(temp + myModMult(1+PRIME-r[j], myModMult(1+PRIME-r[mi+j+1], 1+PRIME-r[mi+mip1+j+1])));
    	ans = myModMult(ans, temp);
  	}
  	//finally check that low-order bit of in1=0 and low-order bit of in2=1
  	ans = myModMult(ans, 1+PRIME-r[mi]);
  	ans = myModMult(ans, r[mi+mip1]);
  	
  	//now handle the z=(anything, (d-i zeros), 1) in1=(anything, (d-i+1 zeros), 1), in2=nip1-1 case
  	
  	//check that the high order bits of z are (d-i zeros) followed by a 1
  	uint64 part_two = r[mi-1];
  	for(int j = 0; j < d-i; j++)
  	{
  		part_two = myModMult(part_two, 1 + PRIME - r[2*d+j]);
  	}
  	
  	//check that highest order bit of in1 is a 1, and then next d-i+1 highest order bits are 0
  	
  	part_two = myModMult(part_two, r[mi+mip1-1]);
  	for(int j = 0; j < d-i+1; j++)
  	{
  		part_two = myModMult(part_two, 1 + PRIME - r[mi+2*d+j]);
  	}
  	
  	//check that lowest order 2*d bits of z and in1 agree
  	for(int j = 0; j < 2*d; j++)
  	{
  		part_two = myModMult(part_two, myMod( myModMult(r[j], r[mi+j]) + myModMult(1 + PRIME - r[j], 1 + PRIME - r[mi+j])));
  	}
  	
  	
  	//check that in2 = nip1-1
  	uint64 in2= nip1-1;
	
	for(int j = 0; j< mip1; j++)
	{
		if( (in2 >> j) & 1)
		{
			part_two = myModMult(part_two, r[mi+mip1+j]);
		}
		else
		{
			part_two = myModMult(part_two, 1+PRIME-r[mi+mip1+j]);
		}
	}
	
	
	uint64 z = ni-1;
	uint64 in1= nip1-1;
	in2= nip1-1;
	
	uint64 part_three=1;
	for(int j = 0; j< mi; j++)
	{
		if( (z >> j) & 1)
		{
			part_three = myModMult(part_three, r[j]);
		}
		else
		{
			part_three = myModMult(part_three, 1+PRIME-r[j]);
		}
	}
	for(int j = 0; j < mip1; j++)
	{
		if( (in1 >> j) & 1)
		{
			part_three = myModMult(part_three, r[mi+j]);
		}
		else
		{	
			part_three = myModMult(part_three, 1+PRIME - r[mi+j]);
		}
	}
	for(int j = 0; j < mip1; j++)
	{
		if( (in2 >> j) & 1)
		{
			part_three = myModMult(part_three, r[mi+mip1+j]);
		}
		else
		{	
			part_three = myModMult(part_three, 1+PRIME - r[mi+mip1+j]);
		}
	}
	
  return myMod(ans + myMod( part_two + part_three) );
}



uint64 mat_add_61_p2dp1(uint64* r, int mi, int mip1, int ni, int nip1)
{
	int three_d_min_i = mi-1;
	int n_squared = ni-myPow(2, mi-2)-1;
	int d = floor(Log2((double) n_squared)/2 + 0.5);
	
	//threshold in wiring structure is at 3d-i (gate 2^{3d-i} is first entry of C)
	int i = (three_d_min_i - 3*d) * (-1);
	
	//all gates at this layer are of the form z=(anything, 0, 0), in1=(2z, 0), in2=(2z+1, 0) where anything is 2d bits
	//or z=(anything, 1, 0) in1=(anything, 0, 1), in2=nip1-1

	//check high order bit of z is 0, and same with high order bit of in1 and in2
	uint64 ans = myModMult(1 + PRIME - r[mi-1], myModMult( 1 + PRIME - r[mi+mip1-1], 1 + PRIME - r[mi+2*mip1-1]));
	
	//check second highest order bit of z is 0
	ans = myModMult(ans, 1 + PRIME - r[mi-2]);
	
	 uint64 temp=0;

  	//this checks that z=2omega1 ignoring high order bit of each and z=2omega2+1, ignoring the low-order bit of omega1 and omega2
  	for(int j = 0; j < mi-2; j++)
  	{
    	temp = myModMult(r[j], myModMult(r[mi+j+1], r[mi+mip1+j+1]));
    	temp = myMod(temp + myModMult(1+PRIME-r[j], myModMult(1+PRIME-r[mi+j+1], 1+PRIME-r[mi+mip1+j+1])));
    	ans = myModMult(ans, temp);
  	}
  	//finally check that low-order bit of in1=0 and low-order bit of in2=1
  	ans = myModMult(ans, 1+PRIME-r[mi]);
  	ans = myModMult(ans, r[mi+mip1]);
  	
  	//now handle the z=(anything, 1, 0) in1=(anything,  0, 1), in2=nip1-1 case
  	
  	//check that the high order bits of z a 1 followed by a zero
  	
  	uint64 part_two = myModMult(1+ PRIME -r [mi-1], r[mi-2]);
  	
  	
  	//check that highest order bits of in1 are a 0 followed by a 1,
  	
  	part_two = myModMult(part_two, myModMult(1 + PRIME - r[mi+mip1-2], r[mi+mip1-1]));

  	
  	//check that lowest order 2*d bits of z and in1 agree
  	for(int j = 0; j < 2*d; j++)
  	{
  		part_two = myModMult(part_two, myMod( myModMult(r[j], r[mi+j]) + myModMult(1 + PRIME - r[j], 1 + PRIME - r[mi+j])));
  	}
  	
  	
  	//check that in2 = nip1-1
  	uint64 in2= nip1-1;
	
	for(int j = 0; j< mip1; j++)
	{
		if( (in2 >> j) & 1)
		{
			part_two = myModMult(part_two, r[mi+mip1+j]);
		}
		else
		{
			part_two = myModMult(part_two, 1+PRIME-r[mi+mip1+j]);
		}
	}
	
	
	uint64 z = ni-1;
	uint64 in1= nip1-1;
	in2= nip1-1;
	
	uint64 part_three=1;
	for(int j = 0; j< mi; j++)
	{
		if( (z >> j) & 1)
		{
			part_three = myModMult(part_three, r[j]);
		}
		else
		{
			part_three = myModMult(part_three, 1+PRIME-r[j]);
		}
	}
	for(int j = 0; j < mip1; j++)
	{
		if( (in1 >> j) & 1)
		{
			part_three = myModMult(part_three, r[mi+j]);
		}
		else
		{	
			part_three = myModMult(part_three, 1+PRIME - r[mi+j]);
		}
	}
	for(int j = 0; j < mip1; j++)
	{
		if( (in2 >> j) & 1)
		{
			part_three = myModMult(part_three, r[mi+mip1+j]);
		}
		else
		{	
			part_three = myModMult(part_three, 1+PRIME - r[mi+mip1+j]);
		}
	}
	
  return myMod(ans + myMod( part_two + part_three) );
}


uint64 mat_add_61_p2d(uint64* r, int mi, int mip1, int ni, int nip1)
{
  //first handle case where z=(any, 0), in1 = (z, 0, 0), in2 = (z, 1, 0)
  //the other case is z=ni-1, in1=in2=nip1-1
  
  //test high order bit of z is 0
  uint64 ans = 1 + PRIME - r[mi-1];
  //test in1 and in2 low order bits match z
  for(int j = 0; j < mi-1; j++)
  {
  	ans = myModMult(ans, myMod( myModMult(r[j], myModMult(r[mi+j], r[mi+mip1+j])) + 
  			myModMult(1+PRIME-r[j], myModMult(1+PRIME-r[mi+j], 1+PRIME-r[mi+mip1+j])) ));
  }	
  //test high order two bits of in1 and in2. in1 first
  ans = myModMult(ans, myModMult(1+PRIME-r[mi+mip1-2], 1+PRIME-r[mi+mip1-1]));
  ans = myModMult(ans, myModMult(r[mi+2*mip1-2], 1+PRIME-r[mi+2*mip1-1]));
  
  //now handle the case of the last gate
  uint64 z = ni-1;
  uint64 in1= nip1-1;
  uint64 in2= nip1-1;
	
	uint64 part_two=1;
	for(int j = 0; j< mi; j++)
	{
		if( (z >> j) & 1)
		{
			part_two = myModMult(part_two, r[j]);
		}
		else
		{
			part_two = myModMult(part_two, 1+PRIME-r[j]);
		}
	}
	for(int j = 0; j < mip1; j++)
	{
		if( (in1 >> j) & 1)
		{
			part_two = myModMult(part_two, r[mi+j]);
		}
		else
		{	
			part_two = myModMult(part_two, 1+PRIME - r[mi+j]);
		}
	}
	for(int j = 0; j < mip1; j++)
	{
		if( (in2 >> j) & 1)
		{
			part_two = myModMult(part_two, r[mi+mip1+j]);
		}
		else
		{	
			part_two = myModMult(part_two, 1+PRIME - r[mi+mip1+j]);
		}
	}
	
  return myMod(ans + part_two);
}
	


void update_a_fast(uint64 num_effective, uint64 rjm1, uint64* vals);

uint64 check_first_level(circ* c, uint64* r, uint64* zi, uint64 d)
{
  int num_effective = myPow(2, c->log_level_length[d]);
  uint64* vals = (uint64*) malloc(num_effective*sizeof(uint64));

  uint64** y = (uint64**) malloc(c->log_level_length[d] * sizeof(uint64*));
  uint64* a = (uint64*) malloc(c->level_length[d] * sizeof(uint64));

  for(uint64 i = 0; i < c->level_length[d]; i++)
	a[i] = c->gates[d][i].val;

  uint64 check=0;
  uint64 ct=0; //check time
  uint64  k = 0;

  for(int i =0; i < c->log_level_length[d]; i++)
	r[i] = 76;//rand();

  clock_t t=clock();
  for(uint64 j=0; j < c->log_level_length[d]; j++) // computes all the messages from prover to verifier
  {
      y[j]=(uint64*) calloc(2, sizeof(uint64));
      for(k = 0; k < (num_effective>>1); k++)
      {
        //a[k] gives f(r0...rj-1, k). Want to plug in m for x_j
	   y[j][1]=myMod(y[j][1] + a[(k<<1)+1]);
	   y[j][0]=myMod(y[j][0] + a[(k<<1)]);
      }
      update_a_fast(num_effective, r[j], a);
      num_effective = num_effective >> 1;
  }
  clock_t pt=clock()-t; //prover time


  cout << "claimed circuit value is: " << (uint64) myMod(y[0][0] + y[0][1]) << endl;
  t=clock();
  for(uint64 j = 0; j < c->log_level_length[d]; j++) //checks all messages from prover to verifier
  {
    if (j>0)
    {
      if(( (check != myMod(y[j][0] + y[j][1])) 
	&& (check != myMod(y[j][0] + y[j][1]) + PRIME)
          && (check + PRIME != myMod(y[j][0] + y[j][1])) ) || (j == 1)) // check consistency of messages
      {
        cout << "Check failed: j is " << j << "check is: ";
	cout << check << "!=" <<  myMod(y[j][0] + y[j][1]);
        cout << "y[" << j << "][0]: " << y[j][0] << " y[" << j << "][1]: " << y[j][1] << endl;
	cout << "y[" << j << "][0]: " << (signed long long) (y[j][0]-PRIME);
	cout  << " y[" << j << "][1]: " << (signed long long) (y[j][1]-PRIME) << endl; 
	cout << "check " << (signed long long) (check-PRIME) << endl;
      }
    }
    check= myMod(myModMult(y[j][0], 1+PRIME-r[j]) + myModMult(y[j][1], r[j])); //compute next check value at random location r
  }

  for(int i = 0; i < c->log_level_length[d]; i++)
   zi[i]=r[i];

  return check;
}
  

//update vals fast
void update_a_fast(uint64 num_effective, uint64 rjm1, uint64* vals)
{
  uint64 index;
  for(uint64 k=0; k < num_effective; k++)
  {
     index = k >> 1;
     if( k & 1)
       vals[index]= myMod(vals[index] + myModMult(vals[k], rjm1));
     else
       vals[index]=myModMult(vals[k], 1+PRIME-rjm1);
  }
}

int main(int argc, char** argv)
{
  if((argc != 3))
  {
    cout << "argc!= 3. Command line arg should be log(universe size) followed by protocol identifier.";
    cout << "0 for F2, 1 for F0 basic, 2 for matrix multiplication, 3 for pattern matching.\n";
    cout << "This implementation supports V evaluating the polynomials add_i and mult_i in logarithmic time.\n";
    exit(1);
  }
  int d = atoi(argv[1]);
  uint64 n = myPow(2, d);
  //protocol specifies whether to run F2 or F0
  int protocol = atoi(argv[2]);

  if((protocol < 0) || (protocol > 3))
  {
     cout << "protocol parameter should be 0, 1, 2, or 3. 0 is for F2, 1 is for F0, 2 for matmult, 3 for pattern matching\n";
     exit(1);
  }

  /********************************************
  /Begin code to construct circuit of interest*
  /********************************************/
  circ* c;
  clock_t t = clock();
  if(protocol == 0)
    c = construct_F2_circ(d);
  else if(protocol == 1)
    c = construct_F0_circ(d);
  else if(protocol == 2)
	c = construct_mat_circ(d);
  else if(protocol == 3)
	c = construct_pm_circ(d);

   cout << "\n circuit construct time is: " << (double)((double) clock()-t)/CLOCKS_PER_SEC << endl;
  /******************************************
  /End code to construct circuit of interest*
  /******************************************/

  /************************************************************
  /Begin generic code to evaluate circuit in verifiable manner*
  /************************************************************/

  //evaluate the circuit
  t=clock();
  evaluate_circuit(c);

  cout << "circuit eval time is: " << (double)((double) clock()-t)/CLOCKS_PER_SEC << endl;

  cout << " The circuit evaluated to: " << c->gates[0][0].val << endl;
  //exit(0);

  uint64* zi = (uint64*) calloc(c->num_levels, sizeof(uint64));
  uint64 ri=c->gates[0][0].val;

  //run through entire Muggles protocol with prover
  clock_t ct=0;
  clock_t pt=0;
  int com_ct=0;
  int rd_ct=0;

  uint64* r;
  uint64** poly;
  uint64* vals;


  if(PROOF)
  {

	  //allocate an array for storing V's random numbers in each iteration of the protocol
	  if( (protocol <= 1 ))
	  {
  		r = (uint64*) calloc(3*(d+4), sizeof(uint64));
  		//allocate a 2-D array to store P's messages in each iteration of the protocol
  		poly = (uint64**) calloc(3*(d+4), sizeof(uint64*));
  		for(int j=0; j < 3*(d+4); j++)
    			poly[j] = (uint64*) calloc(18, sizeof(uint64));
  		//allocate an array to store the values of V_i+1(omega1) and V_i+1(omega2) the P needs to compute her messages
  		vals = (uint64*) calloc(4*n, sizeof(uint64));
	  }
	  else if(protocol ==2)
	  {
		r = (uint64*) calloc(4*(4*d+4), sizeof(uint64));
  		//allocate a 2-D array to store P's messages in each iteration of the protocol
  		poly = (uint64**) calloc(18*3*4*(d+4), sizeof(uint64*));
  		for(int j=0; j < 18*3*4*(d+4); j++)
    			poly[j] = (uint64*) calloc(18, sizeof(uint64));
  		//allocate an array to store the values of V_i+1(omega1) and V_i+1(omega2) the P needs to compute her messages
  		vals = (uint64*) calloc(2*(n*n*n+n*n+1), sizeof(uint64));
	   }
	  else if(protocol == 3)
	  {
		r = (uint64*) calloc(4*(4*d+4), sizeof(uint64));
  		//allocate a 2-D array to store P's messages in each iteration of the protocol
  		poly = (uint64**) calloc(18*3*4*(d+PAT_LEN+4), sizeof(uint64*));
  		for(int j=0; j < 18*3*4*(d+PAT_LEN+4); j++)
    			poly[j] = (uint64*) calloc(18, sizeof(uint64));
  		//allocate an array to store the values of V_i+1(omega1) and V_i+1(omega2) the P needs to compute her messages
  		vals = (uint64*) calloc(4*n*PAT_LEN, sizeof(uint64));
	   }

	  if(protocol == 0)
	  { //check each level in turn using protocol due to GKR. This is for F2 circuit
		  ri = check_first_level(c, r, zi, d-1);
		for(int i = d-1; i < c->num_levels-1; i++)
		{
		  if(i < c->num_levels-2)
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, F2_add_notd, zero, 0);
		  else 
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, zero, F2_mult_d, 0);
		}
	  }
	  else if(protocol ==1)
	  { 
	  	//check each level in turn using protocol due to GKR. This is for F0 circuit
	  
	  //check each level in turn using protocol due to GKR. This is for F0 circuit
		  
		for(int i = d-1; i < c->num_levels-1; i++)
		{
		  if(i == (d-1))
		  	ri = check_first_level(c, r, zi, d-1);
		  if(i < d)
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, F2_add_notd, zero, 0);
		  if(i == d)
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, zero, F0mult_d, 0);
		  if( (i > d) && (i < 59+d))
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, F0add_dp1to58pd, F0mult_dp1to58pd, 0);
		  if( (i==59+d))
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, F0add_59pd, F0mult_59pd, 0);
		  if(i==60+d)
  			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, zero, F0mult_60pd, 0);
		}
	  }
	  else if(protocol == 2)
	  {
		clock_t test = clock();
		for(int i = 2*d-1; i < c->num_levels-1; i++)
		{
			//std::cout << "here. i is: " << i << "start index is: " << start_index << "start_index_ip1 is: " << start_index_ip1 <<  std::endl;
		  if(i==2*d-1)
		  {
			  ri = check_first_level(c, r,  zi, 2*d);
			  //std::cout << " time spent on first check is: " << (double) (clock() - test)/CLOCKS_PER_SEC << std::endl;
		  }
		  else if(i == 2*d)
		  {
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, zero, F0mult_d, 0);
			//std::cout << " time spent on secnod check is: " << (double) (clock() - test)/CLOCKS_PER_SEC << std::endl;
		  }
		  else if( (i > 2*d) && (i < 59+2*d))
		  {
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, F0add_dp1to58pd, F0mult_dp1to58pd, 0);
		  }
		  else if( (i==59+2*d))
		  {
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, F0add_59pd, F0mult_59pd, 0);
		  }
		  else if(i==60+2*d)
		  {
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, zero, F0mult_60pd, 0);
		  }
		  else if(i==61+2*d)
		  {
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals,  mat_add_61_p2d, zero, 0);
		  }
		  else if(i<=61+3*d)
		  { 
		  	if(i == 61+2*d+1)
		  		ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, mat_add_61_p2dp1, zero, 0);
		  	else
				ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, mat_add_below_63_p3d, zero, 0);
		  }
		  else
		  {
			ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, mat_add_63_p3d, mat_mult_63_p3d, 0);
		  }
		}
	  }
	  else if(protocol == 3)
	  {
		  clock_t test = clock();
		  for(int i = d-1; i < c->num_levels-1; i++)
		  {
			  if(i==d-1)
			  {
				  ri = check_first_level(c,  r,  zi, d);
			  }
			  else if(i == d)
			  {
				ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, zero, F0mult_d, 0);
			  }
			  else if( (i > d) && (i < 59+d))
			  {
				ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, F0add_dp1to58pd, F0mult_dp1to58pd, 0);
			  }
			  else if( (i==59+d))
			  {
				ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals,F0add_59pd, F0mult_59pd, 0);
			  }
			  else if(i==60+d) //squaring level within F0 circuit
			  {
				ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, zero, F0mult_60pd, 0);
			  }
			  else if(i<=60+d+LOG_PAT_LEN) //binary tree of sum gates
			  {
				ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, F2_add_notd, zero, 0);
			  }
			  else if(i==61+d+LOG_PAT_LEN) //squaring level
			  {
				ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, zero, F2_mult_d, 0);
			  }
			  else if(i==62+d+LOG_PAT_LEN)
			  {
				ri=check_level(c, i, zi, ri, &com_ct, &rd_ct, r, poly, vals, zero, zero, 1);
			  }
		  }
	  }
  }
  pt=clock()-t;

  //in practice, this stream observation stage would be occur before the
  //interaction with the prover (with the random location zi at which the LDE of the input is evaluated
  //determined in advance), but in terms of timing things, it doesn't matter so we're doing it after
  //the conversation rather than before.
  t=clock();
  uint64 fr=evaluate_V_i(c->log_level_length[c->num_levels-1], c->level_length[c->num_levels-1], 
                                     c->gates[c->num_levels-1], zi);
  double vt_secs = (double)(clock() - t)/CLOCKS_PER_SEC; 
  double pt_secs =((double) pt)/CLOCKS_PER_SEC;

  if( (fr != ri) && (fr!=ri+PRIME) && (fr != ri-PRIME))
  {
    cout << "Check value derived from stream (fr) is not equal to its claimed value by P (ri).\n";
    cout << "fr is: " << fr << " and ri is: " << ri << endl;
  }
 
  cout << "Done!" << endl;

  cout << "N\tProveT\tVerT\tProofS\tRounds\n";
  cout << myPow(2,d) << "\t" << pt_secs << "\t" << vt_secs << "\t" << com_ct << "\t" << rd_ct << endl;


  //clean up memory before exit
  /*free(r);
  for(int j=0; j < 3*(d+4); j++)
    free(poly[j]);
  free(poly);
  free(vals);
  free(zi);
  destroy_circ(c);*/

  return 1;
}
