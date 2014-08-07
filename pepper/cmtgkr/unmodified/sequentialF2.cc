/*****************************
//Justin Thaler
//May 7, 2011
//FFT-based non-interactive protocol for F2. Implementation described in
//"Practical Verified Computation with Streaming Interactive Proofs"
//by Cormode, Mitzenmacher, and Thaler.
//This program uses a Prime Factor Algorithm due to 
// C. S. Burrus and P. W. Eschenbacher, "An in-place,
// in-order prime factor FFT algorithm"
********************************/

#include <cstdlib>
#include <iostream>
#include <time.h>
#include <cstring>

#define MASK 4294967295 //2^32-1
#define PRIME 2305843009213693951 //2^61-1

typedef unsigned long long uint64;

using namespace std;

uint64 myPow(uint64 x, uint64 b) {
  uint64 i = 1;
  for (int j = 0; j < b; j++)  i *= x;
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
  if((e % 2) == 0)
  {
    result = myModPow(b, e/2);
    return myModMult(result, result);
  }
  else
  {
     return myModMult(myModPow(b, e-1), b);
  }
}

//Performs Extended Euclidean Algorithm
//Used for computing multiplicative inverses mod p
//Computes a solution  to u*u1 + p*u2 = gcd(u,v)=u3
//only works for p=2^61-1
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

void PFA(int N, uint64* x, uint64* A, int* N_i, int* NoverN_i, int M, uint64 Nth_root, uint64 unsc);

//do a length-N1 DFT using numbers
//x[I[0]], ... x[I[N1-1]]
//saves answers in corresponding entries of x
void DFT(int* I, int N1, uint64* x, uint64* prim_to_ij)
{

  uint64 temp1;
  uint64 temp2;
  uint64 temp3;

  //Can achieve significant speedups by handling length 2 and length 3 DFTs as special cases
  if(N1==2)
  {
    temp1 = x[I[0]];
    x[I[0]]=myMod(x[I[0]]+x[I[1]]);
    x[I[1]]=myMod(temp1+ PRIME-x[I[1]]);
    return;
  }
  if(N1 == 3)
  {
    temp1 = myMod(x[I[0]]+x[I[1]]+x[I[2]]);
    temp2 = myMod(x[I[0]] + myModMult(1669582390241348315, x[I[1]]) + myModMult(636260618972345635,x[I[2]]));
    temp3 = myMod(x[I[0]] + myModMult(1669582390241348315, x[I[2]]) + myModMult(636260618972345635,x[I[1]]));
    x[I[0]]=temp1;
    x[I[1]]=temp2;
    x[I[2]]=temp3;
    return;
  } 
    
  uint64 answers[31];
  //uint64* answers = (uint64*) calloc(N1, sizeof(uint64));
  uint64 answer = 0;
  uint64 factor = 1;

  int i;
  int j;
  
  answers[0]=0;
  for(i=0; i < N1; i++)
   answers[0] = myMod(answers[0] + x[I[i]]);

  int ij_modN1 = 0;
  
  for(i = 1; i < N1; i++) 
  {
     ij_modN1 = i;
     answer=x[I[0]];
     for(j = 1; j < N1; j++)
     {
       answer = myMod(answer + myModMult(x[ I[j] ], prim_to_ij[ij_modN1]));
       ij_modN1+=i; 
       if(ij_modN1 >= N1)
         ij_modN1 = ij_modN1-N1;
     }
     answers[i] = answer;
 }

 for(j=0; j < N1; j++)
 {
   x[I[j]] = answers[j];
 }
}

//tranforms array x of length N, saves answer in A
void PFA(int N, uint64* x, uint64* A, int* N_i, int* NoverN_i, int M, uint64 Nth_root, uint64 unsc)
{
  int I[31];
  uint64 N1th_root;
  uint64 prim_to_ij[31];
  
  int N1;
  int N2;
 
  int L;
  for(int k = 0; k < M; k++)
  {
     N1 = N_i[k];
     N2 = NoverN_i[k];
     
     N1th_root=myModPow(Nth_root, N2);
     prim_to_ij[0]=1;
     for(int m = 1; m < N1; m++)
       prim_to_ij[m]=myModMult(prim_to_ij[m-1], N1th_root);
     
     I[0]=0;
   
     for(int j = 0; j < N2; j++)
     {
       for(L = 1; L < N1; L++)
       {
         I[L] = I[L-1]+N2;
         if( I[L] >= N)
           I[L]=I[L]-N;
       }
       //modifies the contents of x
       DFT(I, N1, x, prim_to_ij);
       I[0]=I[0]+N1;
     }
   }
   
   //unscramble x to get output
   L = 0;
   for(int k = 0; k < N; k++)
   {
     A[k] = x[L];
     L = L + unsc;
     if(L >= N)
     	L = L-N;
   }
}


//build a look-up table of multiplicative inverses
//only works for p=2^61-1
uint64* geninv(uint64 n)
{
  uint64* ifact_plus = (uint64*) calloc(n, sizeof(uint64));
  uint64* ifact_minus = (uint64*) calloc(n, sizeof(uint64));
  ifact_plus[0]=1;
  ifact_minus[0]=1;
  uint64 i;
  for(i=1; i < n; i++)
	ifact_plus[i] = myModMult(inv(i), ifact_plus[i-1]);  
  for(i=1; i < n; i++)
	ifact_minus[i] = myModMult(inv(-i+ PRIME), ifact_minus[i-1]); 
  uint64* table= (uint64*) calloc(n, sizeof(uint64));
  for(i=0; i < n; i++)
	table[i] = myModMult(ifact_plus[i], ifact_minus[n-i-1]);
  free(ifact_plus);
  free(ifact_minus);
  return table;
}	


//deprecated (as long as r >=n, which happens with extremely high probability)
//even if r<n there are faster ways to generate the lookup table, but this never
//happens so I'm not bothering to deal with it.
//generate a look up table of factorials needed for lagrange polys
//i'th entry is numerator of chi_x(r)
uint64* genrtabOLD(uint64 n, uint64 r)
{
  uint64* table = (uint64*) calloc(n, sizeof(uint64));
  uint64 mult=1;
  
  uint64* lookup = (uint64*) calloc(n, sizeof(uint64));
  for(uint64 i = 0; i < n; i++)
	lookup[i] = myMod(r-i+ PRIME);
  for(uint64 i = 0; i < n; i++)
  {
    mult=1;
    for(uint64 j=0; j < n; j++)
    {
      if(i != j)
        mult=myModMult(mult, lookup[j]);
    }
    table[i]=mult;
  }
  free(lookup);
  return table;
}

//generate a look up table of factorials needed for lagrange polys
//i'th entry is numerator of chi_x(r)
uint64* genrtab(uint64 n, uint64 r)
{
  if(r < n)
    return genrtabOLD(n, r);
  
  uint64* table = (uint64*) calloc(n, sizeof(uint64));
  uint64 mult=1;
  
  //I will use the fact that rtab[i] = rtab[i-1](r-i+1)/(r-i) if r != i, r!=i-1
  for(uint64 j=1; j < n; j++)
  {
      mult=myModMult(mult, r-j);
  }
  table[0]=mult;
    
  for(uint64 i = 1; i < n; i++)
  {
    table[i]=myModMult(myModMult(table[i-1], r-i+1), inv(r-i));
  }
  return table;
}

//returns array the_hs such that the_hs[j] = (h+j)!/j!
uint64* gen_hs(uint64 h)
{
  uint64* the_hs = (uint64*) malloc(h * sizeof(uint64));
  uint64 mult=1;
  for(uint64 j=2; j <= h; j++)
  {
      mult=myModMult(mult, j);
  }
  the_hs[0]=mult;
    
  for(uint64 i = 1; i < h; i++)
  {
    the_hs[i]=myModMult(myModMult(the_hs[i-1], h+i), inv(i));
  }
  return the_hs;
}

uint64* gen_invs(uint64 h, uint64 N)
{
  uint64* the_invs = (uint64*) calloc(N, sizeof(uint64));
  for(uint64 j = 1; j < 2*h; j++)
    the_invs[j] = inv(j);
  return the_invs;
}

//extrapolate vector vec of length n to location r
//using lookup table of inverses itab and of factorials rtab
uint64 tabextrap(uint64* vec, uint64 n, uint64 r, uint64* itab, uint64* rtab)
{
  //we never use r here?
  uint64 result=0;
  for(uint64 i=0; i< n; i++)
  {
     result=myMod(result+myModMult(myModMult(rtab[i], itab[i]), vec[i]));
  }     
  return result;
}

//randomly generate data of size v*h and compute exact F2
uint64 gendata(uint64 h, uint64 v, uint64** a)
{
  uint64 f2=0;
  for(uint64 i=0; i< v; i++)
  {
    for(uint64 j=0; j< h; j++)
    {
      a[i][j]=rand() % 1000;
      f2+=a[i][j]*a[i][j];
    }
  }
  return f2;
}

//generate the check values for the verifier by extrapolating each row to column r
uint64 vcheck(uint64 h, uint64 v, uint64 r, uint64** a, uint64* itab)
{
  uint64 check=0;
  uint64* rtab=genrtab(h, r);
  uint64 ext;
  for(uint64 i=0; i < v; i++)
  {
    ext=tabextrap(a[i],h,r,itab,rtab); //compute LDE(a)(i, r)
    check=myMod(check+myModMult(ext, ext)); //in end, check = sum_i=1^v LDE(a)(i,r)^2
  }
  free(rtab);
  return check;
}


//build the proof by extrapolating the data to adjacent columns, 
//then summing to get a vector of sums of squares, which can be extrapolated by V
uint64* proof(uint64 h, uint64 v, uint64* itab, uint64** a)
{
  //clock_t t = clock();
  uint64 i;
  uint64 j;
  uint64 k;

  int N=1;
  int M=0;
  int* N_i = (int*) malloc(6 * sizeof(int));
  int* NoverN_i = (int*) malloc(6*sizeof(int));
  uint64 Nth_root;
  uint64 unsc;

  //The following logic figures out what sized DFT to use.
  //(can use \prod_{i in Z} z_i, where Z is any subset of {2, 3, 3, 5, 5, 7, 11, 13, 31} )
  int S[9]={2, 3, 3, 5, 5, 7, 11, 13, 31};
  uint64 mult=1;
  //just assign min_cost to some really large number
  uint64 min_cost = PRIME;
  uint64 cost;
  uint64 sum;

  int best_j;
  int three_found=0;
  int five_found=0;
  //for each Z
  for(j=0; j < 512; j++)
  {
   mult=1;
   sum=0;
   three_found = 0;
   five_found = 0;
   //for each potential element in Z
   for(k=0; k < 9; k++)
   {
     //if item k is in Z
     if((j>>k)&1)
     {
       if(S[k]==3)
       {
         if(three_found==1)
           sum+=6;
         else
         {
          sum+=3;
          three_found=1;  
         }
         mult=mult*S[k];
         continue;
       }
       if(S[k]==5)
       {
         if(five_found==1)
           sum+=20;
         else
         {
          sum+=5;
          five_found=1;  
         }
         mult=mult*S[k];
         continue;
       }
       
       sum=sum+S[k]; 
       mult=mult*S[k];
     }
   }
   //approximate cost of PFAing with this factorization
   cost = v*sum*mult;
   if((mult >= 2*h) && (cost < min_cost))
   {
      min_cost = cost;
      best_j = j;
   }
  }
  //you've identified the best set now, so set up M, N, and N_i
  int should_continue=0;
  for(k=0; k < 9; k++)
  {
    should_continue=0;
    //if item k is in best Z
    if((best_j>>k)&1)
    {
      if(S[k]==3)
      {
        for(i=0; i < M; i++)
        {
          if(N_i[i]==3)
          {
            N_i[i]=9;
            should_continue=1;
          }
        }
      }
      if(S[k]==5)
      {
        for(i=0; i < M; i++)
        {
          if(N_i[i]==5)
          {
            N_i[i]=25;
            should_continue=1;
          }
        }
      }
      N=N*S[k];
      if(should_continue)
         continue;
      N_i[M]=S[k];
      M++;
    }
  }

  //cout << "N is: " << N << " M is: " << M << endl;
  //for(i=0; i < M; i++)
    // cout << "N_i[" << i << "] is: " << N_i[i];
  //cout << endl;

  for(i = 0; i < M; i++)
    NoverN_i[i] = N/N_i[i];
  Nth_root=myModPow(37, (PRIME-1)/N);
  unsc=0;
  for(int i = 0; i < M; i++)
  {
    unsc+=NoverN_i[i];
    if(unsc >= N)
     unsc = unsc-N;
  }  
  uint64 * the_hs = gen_hs(h);
  uint64* the_invs = gen_invs(h, N);
  uint64 inv_N = inv(N);

  uint64* f = (uint64*) calloc(N, sizeof(uint64));
  uint64* transformed_f = (uint64*) malloc(N*sizeof(uint64));
  uint64* transformed_invs = (uint64*) malloc(N* sizeof(uint64)); 
  PFA(N, the_invs, transformed_invs, N_i, NoverN_i, M, Nth_root, unsc);
  //clock_t t=clock();
  for(i = 0; i < v; i++)
  {
    //set up f vector in convolution
    for(k=0; k < h; k++)
    {
      f[k] = myModMult(a[i][k], itab[k]);
    }
    //transform f vector
    PFA(N, f, transformed_f, N_i, NoverN_i, M, Nth_root, unsc);
    //point-wise multiply transformed vectors
    for(k = 0; k < N; k++)
    {
      transformed_f[k] = myModMult(transformed_f[k], transformed_invs[k]);
    }
    //transform back and save in f (need to scale each entry by N^{-1} mod p)
    PFA(N, transformed_f, f, N_i, NoverN_i, M, Nth_root, unsc);
    //at this point, f[N-(h+j)]*the_hs[j]*1/N equals what a[i][h+j] should be for j=0...h-1
    for(k = 0; k < h; k++)
    {
       a[i][h+k] = myModMult(myModMult(f[N-(h+k)], the_hs[k]), inv_N); 
    }
    memset(f, 0, N*sizeof(uint64));
  }
  
  uint64* b=(uint64*) calloc(2*h, sizeof(uint64));   
  uint64 row_f2 = 0;
  for(j=0; j<2*h; j++)
  {
    row_f2=0;
    for(i=0; i<v; i++)
    {
      row_f2=myMod(row_f2+myModMult(a[i][j], a[i][j]));
    }
    b[j]=myMod(row_f2);
  }

  free(f);
  free(transformed_f);
  free(transformed_invs);
  free(the_hs);
  free(the_invs);
  free(N_i);
  free(NoverN_i);  

  return b;
}

int main(int argc, char** argv)
{
    //uint64 skipp;
    if(argc != 3)
    {
      cout << "There should be exactly two command line arguments, specifying h and v";
      exit(1);
    }
    int h= atoi(argv[1]); //read dimensionality from command line
	int v = atoi(argv[2]);

    //prepare space for data
    uint64** a = (uint64**) calloc(v, sizeof(uint64*));
    uint64 i;
    for(i=0; i < v; i++)
       a[i] = (uint64*) calloc(2*h, sizeof(uint64));
    i=0;
    uint64 r=rand(); //pick random location r
    uint64 cf2=gendata(h,v, a);

    //have prover generate his lookup tables. We charge him for this even though they
    //are data independent so he could do this in advance (this cost is quite small now anyway)
    clock_t t=clock();
    uint64* itab=geninv(h); //table of denominators of lagrange polynomials
    clock_t itab_time = clock()-t;

    clock_t ptable_time=clock()-t;

    //have verifier generate his lookup tables (he will use itab also, which we created above). 
    //We will charge for these as well even though they are data-independent (cost is small anyway)
    
    uint64* itab2=geninv(2*h);
    uint64* rtab2=genrtab(2*h, r);
    t=clock();
    uint64 check=vcheck(h,v,r, a, itab); // make verifiers check values
    clock_t vt=clock()-t;

    uint64* b;
    clock_t pt;
    clock_t ct;
    uint64 f2;
    uint64 result;
    t=clock();
    b=proof(h,v,itab, a); //create prover's proof
    pt=clock()-t;
    //itab2=geninv(2*h)
    t=clock();
    result = tabextrap(b, 2*h, r, itab2, rtab2);
    //result=medextrap(b,2*h,r,itab2);
    //result=extrap(b,2*h,r); //extrapolate the proof vector to location r
    ct=clock()-t;
       
    f2=0;

    for(i= 0; i < h; i++)
      f2+=b[i]; //compute claimed F2 value

    if (result!=check || f2!=cf2) 
    {
      cout << "FAIL!\n" << "result: " << result << " check: " << check;
      cout << "f2: " << f2 << " cf2: " << cf2 << endl;
    } 
	else
	{
		cout << "passed. result is: " << result << endl;
	}

    cout << "N\tVerifT\tProveT\tCheckT\tVerifS\tProofS\n";
    // print a header if requested
    cout << h*v << "\t" << (double) (vt+itab_time)/CLOCKS_PER_SEC << "\t" << (double) (pt+ptable_time)/CLOCKS_PER_SEC;
    cout << "\t" <<  (double) ct/CLOCKS_PER_SEC  << "\t" << v << "\t" <<2*h << " " << endl;
    //print output
	
	//clean up memory before exiting
    for(i=0; i < v; i++)
      free(a[i]);
    free(a);
    free(itab2);
    free(rtab2);
    free(itab);
    free(b);
}

