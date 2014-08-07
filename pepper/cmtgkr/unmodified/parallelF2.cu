#include <cstdlib>
#include <iostream>
#include <time.h>
#include <omp.h>

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/for_each.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/host_vector.h> 
#include <thrust/iterator/reverse_iterator.h>
#include <thrust/iterator/counting_iterator.h>


#define MASK         4294967295          // 2^32-1
#define PRIME        2305843009213693951 // 2^61-1
#define NUM_ELEMENTS 1024 * 1024

typedef unsigned long long uint64;

uint64 myPow(uint64 x, uint64 p) {
  uint64 i = 1;
  for (int j = 0; j < p; j++)  i *= x;
  return i;
}


__host__ __device__
uint64 myMod(uint64 x)
{
  return (x >> 61) + (x & PRIME);
}  

//efficient modular multiplication function mod 2^61-1
__host__ __device__
 
uint64 myModMult(uint64 x, uint64 y)
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
__host__ __device__ uint64 myModPow(uint64 b, uint64 e)
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


//build a look-up table of multiplicative inverses
//only works for p=2^61-1
void geninv(uint64 n, thrust::host_vector<uint64>& table)
{
  uint64* ifact_plus = (uint64*) calloc(n, sizeof(uint64));
  uint64* ifact_minus = (uint64*) calloc(n, sizeof(uint64));
  ifact_plus[0]=1;
  ifact_minus[0]=1;
  uint64 i;
  for(i=1; i < n; i++)
	ifact_plus[i] = myModMult(inv(i), ifact_plus[i-1]);  
  for(i=1; i < n; i++)
	ifact_minus[i] = myModMult(inv(-i+PRIME), ifact_minus[i-1]); 

  for(i=0; i < n; i++)
	table[i] = myModMult(ifact_plus[i], ifact_minus[n-i-1]);
  free(ifact_plus);
  free(ifact_minus);
}	


//deprecated (as long as r >=n, which happens with extremely high probability)
//even if r<n there are faster ways to generate the lookup table, but this never
//happens so I'm not bothering to deal with it.
//generate a look up table of factorials needed for lagrange polys
//i'th entry is numerator of chi_x(r)
void genrtabOLD(uint64 n, uint64 r, thrust::host_vector<uint64>& table)
{
  uint64 mult=1;
  
  uint64* lookup = (uint64*) calloc(n, sizeof(uint64));
  for(uint64 i = 0; i < n; i++)
	lookup[i] = myMod(r-i+PRIME);
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
}

//generate a look up table of factorials needed for lagrange polys
//i'th entry is numerator of chi_x(r)
void genrtab(uint64 n, uint64 r, thrust::host_vector<uint64>& table)
{
  if(r < n)
  {
	    genrtabOLD(n, r, table);
		return;
  }
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
}

//returns array the_hs such that the_hs[j] = (h+j)!/j!
void gen_hs(uint64 h, thrust::host_vector<uint64>& the_hs)
{
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
}

void gen_invs(uint64 h, uint64 N, thrust::host_vector<uint64>& the_invs)
{
  for(uint64 j = 1; j < 2*h; j++)
    the_invs[j] = inv(j);
}


struct triple_product
{
	enum TupleLayout
	{
		A,
		B,
		C,
		TRIPLE_PRODUCT_RESULT
	};
	
    triple_product() {}

	template <typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        uint64 a = thrust::get< A >( tuple );                                              
        uint64 b = thrust::get< B >( tuple );                                              
        uint64 c = thrust::get< C >( tuple );                                              

		uint64 triple_product_result = myModMult( myModMult( a, b ), c );
		
        thrust::get< TRIPLE_PRODUCT_RESULT >( tuple ) = triple_product_result;
    }
};

struct modular_sum : public thrust::binary_function<uint64,uint64,uint64>
{
    __host__ __device__
    uint64 operator() (const uint64& lhs, const uint64& rhs)
    {
		return myMod( myMod(lhs) + myMod(rhs) );
	}
};

//extrapolate vector vec of length n to location r
//using lookup table of inverses itab and of factorials rtab
uint64 tabextrap(thrust::device_vector<uint64>& vec, int row, uint64 n, uint64 r, thrust::device_vector<uint64>& itab, 
thrust::device_vector<uint64>& rtab, thrust::device_vector<uint64>& intermediate)
{  //we never use r here, could really get rid of the parameter

 thrust::for_each(

        thrust::make_zip_iterator( thrust::make_tuple( vec.begin()+row*n, itab.begin(), rtab.begin(), intermediate.begin() ) ),

        thrust::make_zip_iterator( thrust::make_tuple( vec.begin() + row*n+n, itab.end(), rtab.end(), intermediate.end() ) ),
              
        triple_product() );

	uint64 answer_parallel = thrust::reduce( intermediate.begin(), intermediate.begin() + n, (uint64)0, modular_sum() );
	return answer_parallel;
}

//generate the check values for the verifier by extrapolating each row to column r
uint64 vcheck(uint64 h, uint64 v, uint64 r, thrust::device_vector<uint64>& a, thrust::device_vector<uint64>& itab, 
thrust::device_vector<uint64>& rtab, thrust::device_vector<uint64>& intermediate)
{
  uint64 check=0;
  uint64 ext;
  for(int i=0; i < v; i++)
  {
    ext=tabextrap(a, i, h,r,itab,rtab, intermediate); //compute LDE(a)(i, r)
    check=myMod(check+myModMult(ext, ext)); //in end, check = sum_i=1^v LDE(a)(i,r)^2
  }
  return check;
}

struct tab_extrap_functor
{	
  uint64 h;
  uint64 v;
  uint4 r;
  uint64* a;
  uint64* itab;
  uint64* rtab;
  uint64* intermediate;
  
    tab_extrap_functor(uint64 p_h, uint64 p_v, uint64 p_r, uint64* p_a, uint64* p_itab, uint64* p_rtab, uint64* p_intermediate)
	{
		h=p_h;
		v=p_v;
		a = p_a;
		itab = p_itab;
		rtab = p_rtab;
		intermediate = p_intermediate;
	}

    __device__ void operator() ( int i )
    {
		uint64 ans = 0;
        for(int j = 0; j < h; j++)
		{
			ans = myMod(ans + myModMult(itab[j], myModMult(rtab[j], a[j*v + i])));
		}
		intermediate[i] = ans;
    }
};

struct F2_functor
{
    F2_functor() {}

    __device__
    unsigned long long operator() ( const unsigned long long& y )
    {
    	return myModMult(y, y);
    }
};


unsigned long long computef2( thrust::device_vector<unsigned long long>& a_dev, int row, int numcols, thrust::device_vector<unsigned long long>& a_2dev )
{
	// copy a to a_dev, then a_dev to a_2dev
    thrust::copy( a_dev.begin()+row*numcols, a_dev.begin() + row*numcols + numcols, a_2dev.begin() );

 	//a_2dev <- a * a
    thrust::transform( a_2dev.begin(), a_2dev.begin() + numcols, a_2dev.begin(), F2_functor() );
    return thrust::reduce( a_2dev.begin(), a_2dev.begin() + numcols, (uint64)0, modular_sum() );
}

uint64 vcheck_by_cols(uint64 h, uint64 v, uint64 r, thrust::device_vector<uint64>& a, thrust::device_vector<uint64>& itab, 
thrust::device_vector<uint64>& rtab, thrust::device_vector<uint64>& intermediate)
{
	thrust::counting_iterator<int> first(0);
	thrust::for_each(first, first+v, tab_extrap_functor(h, v, r, thrust::raw_pointer_cast(&a[0]), 
		thrust::raw_pointer_cast(&itab[0]), thrust::raw_pointer_cast(&rtab[0]), thrust::raw_pointer_cast(&intermediate[0])));
	return computef2(intermediate, 0, v, a);
}
	

uint64 myRand()
{
	return rand() % 1000;
}


void compute_N_M_N_i_and_N_overNi(int* outerN, int* outerM, int* outerN_i, int* outerNoverN_i, unsigned long long h, unsigned long long v)
{
  int N=1;
  int M=0;
  
  int N_i[6];

  //The following logic figures out what sized DFT to use.
  //(can use \prod_{i in Z} z_i, where Z is any subset of {2, 3, 3, 5, 5, 7, 11, 13, 31} )
  int S[9];
  S[0]=2; S[1]=3; S[2]=3; S[3]=5; S[4]=5; S[5]=7; S[6]=11; S[7]=13; S[8]=31;
  unsigned long long mult=1;
  //just assign min_cost to some really large number
  unsigned long long min_cost = PRIME;
  unsigned long long cost;
  unsigned long long sum;

  int best_j;
  int three_found=0;
  int five_found=0;
  //for each Z
  for(unsigned long long j=0; j < 512; j++)
  {
   mult=1;
   sum=0;
   three_found = 0;
   five_found = 0;
   //for each potential element in Z
   for(int k=0; k < 9; k++)
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
  for(int k=0; k < 9; k++)
  {
    should_continue=0;
    //if item k is in best Z
    if((best_j>>k)&1)
    {
      if(S[k]==3)
      {
        for(int i=0; i < M; i++)
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
        for(int i=0; i < M; i++)
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
  *outerN=N;
  *outerM=M;
  for(int j = 0; j < M; j++)
    outerN_i[j]=N_i[j];
  for(int i = 0; i < M; i++)
    outerNoverN_i[i] = N/N_i[i];
  
}

__device__ void PFA( int N, unsigned long long* x, unsigned long long* A, int* N_i, int* NoverN_i, int M, 
		unsigned long long Nth_root, unsigned long long unsc, int* I, uint64* prim_to_ij, uint64* answers, int row, int v );



struct pfa_functor
{	
    int m_N;
	uint64* m_x; 
	uint64* m_A;
	int* m_N_i;
	int* m_NoverN_i;
	int m_M;
	uint64 m_Nth_root;
	uint64 m_unsc;
	int* m_I;
	uint64* m_prim_to_ij;
	uint64* m_answers;
	int v;

    pfa_functor( int N, unsigned long long* x, unsigned long long* A, int* N_i, int* NoverN_i, int M, 
		unsigned long long Nth_root, 
		unsigned long long unsc, int* I, uint64* prim_to_ij, uint64* answers, int p_v )
	{
		m_N = N;
		m_x =x; 
		m_A=A; //A is transformed_f (meant to hold result of the transformation)
		m_N_i=N_i;
		m_NoverN_i=NoverN_i;
		m_M = M;
		m_Nth_root=Nth_root;
		m_unsc = unsc;
		m_I=I;
		m_prim_to_ij = prim_to_ij;
		m_answers = answers;
		v = p_v;
	}

    __device__ void operator() ( int col )
    {
    	//col is the index of the column of x you should be transforming (x has N rows, v cols)
        PFA(m_N, m_x, m_A, m_N_i, m_NoverN_i, m_M, m_Nth_root, m_unsc, m_I, m_prim_to_ij, m_answers, col, v);
    }
};

/*struct colf2_functor
{	
    int col;
	int row_len;
	uint64* squared_entries;
	uint64* a;

    colf2_functor( int p_col, int p_row_len, uint64* p_squared_entries, uint64* p_a)
	{
		col = p_col;
		row_len = p_row_len;
		squared_entries = p_squared_entries;
		a=p_a;
	}

    __device__ void operator() ( int row )
    {
        squared_entries[row] = myModMult(a[row_len*row + col], a[row_len*row+col]);
    }
};*/

struct add_squared
{
	enum TupleLayout
	{
		A,
		B,
		RESULT
	};
	
    add_squared() {}

	template <typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        uint64 a = thrust::get< A >( tuple );                                              
        uint64 b = thrust::get< B >( tuple );                                              
                                                    
		uint64 result = myMod( b + myModMult( a, a ) );
		
        thrust::get< RESULT >( tuple ) = result;
    }
};

void colf2(thrust::device_vector<uint64>& a, int row, int h, thrust::device_vector<uint64>& b, int b_offset)
{  //we never use r here, could really get rid of the parameter

 thrust::for_each(

        thrust::make_zip_iterator( thrust::make_tuple( a.begin()+row*h, b.begin()+ b_offset, b.begin() + b_offset ) ),

        thrust::make_zip_iterator( thrust::make_tuple( a.begin() + row*h+h, b.begin() + b_offset + h, b.begin() + b_offset + h ) ),
              
        add_squared() );
}



//do a length-N1 DFT using numbers
//x[I[0]], ... x[I[N1-1]]
//saves answers in corresponding entries of x
__device__ void DFT(int* I, int N1, unsigned long long* x, unsigned long long* prim_to_ij, uint64* answers, int col, int v)
{

  unsigned long long temp1;
  unsigned long long temp2;
  unsigned long long temp3;

  //Can achieve significant speedups by handling length 2 and length 3 DFTs as special cases
  if(N1==2)
  {
    temp1 = x[I[col] * v + col];
    x[I[col]*v + col]=myMod(x[I[col] * v + col]+x[I[v+col] * v + col]);
    x[I[v+col]*v + col]=myMod(temp1+PRIME-x[I[v+col]* v + col]);
    return;
  }
  if(N1 == 3)
  {
    temp1 = myMod(x[I[col]*v + col]+x[I[v+col]*v + col]+x[I[2*v+col] * v + col]);
    temp2 = myMod(x[I[col]*v + col] + myModMult(1669582390241348315, x[I[v+col]*v + col]) + myModMult(636260618972345635,x[I[2*v+col]*v + col]));
    temp3 = myMod(x[I[col]*v + col] + myModMult(1669582390241348315, x[I[2*v+col]*v + col]) + myModMult(636260618972345635,x[I[v+col]*v + col]));
    x[I[col]*v + col]=temp1;
    x[I[v+col]*v + col]=temp2;
    x[I[2*v+col]*v + col]=temp3;
    return;
  } 
    
  //unsigned long long* answers = (unsigned long long*) calloc(N1, sizeof(unsigned long long));
  unsigned long long answer = 0;

  int i;
  int j;
  
  answers[col]=0;
  for(i=0; i < N1; i++)
   answers[col] = myMod(answers[col] + x[I[i*v+col]*v + col]);

  int ij_modN1 = 0;
  
  for(i = 1; i < N1; i++) 
  {
     ij_modN1 = i;
     answer=x[I[col]*v + col];
     for(j = 1; j < N1; j++)
     {
       answer = myMod(answer + myModMult(x[ I[j*v+col]*v + col ], prim_to_ij[ij_modN1*v+col]));
       ij_modN1+=i; 
       if(ij_modN1 >= N1)
         ij_modN1 = ij_modN1-N1;
     }
     answers[i*v+col] = answer;
 }

 for(j=0; j < N1; j++)
 {
   x[I[j*v + col]*v + col] = answers[j*v+col];
 }
}

//tranforms array x of length N, saves answer in A
__device__ void PFA(int N, uint64* x, uint64* A, int* N_i, int* NoverN_i, int M, uint64 Nth_root, 
					uint64 unsc, int* I, uint64* prim_to_ij, uint64* answers, int col, int v)
{
  //transform the col'th column of x. N is number of rows, v is number of cols
  unsigned long long N1th_root;
  
  int N1;
  int N2;
 
  int L;
  for(int k = 0; k < M; k++)
  {
     N1 = N_i[k];
     N2 = NoverN_i[k];
     
     N1th_root=myModPow(Nth_root, N2);
     prim_to_ij[col]=1;
     for(int m = 1; m < N1; m++)
       prim_to_ij[m*v+col]=myModMult(prim_to_ij[(m-1)*v+col], N1th_root);
     
     I[col]=0;
   
     for(int j = 0; j < N2; j++)
     {
       for(L = 1; L < N1; L++)
       {
         I[L*v + col] = I[(L-1) * v + col]+N2;
         if( I[L*v + col] >= N)
           I[L*v+col]=I[L*v + col]-N;
       }
       //modifies the contents of x
       DFT(I, N1, x, prim_to_ij, answers, col, v);
       I[col]=I[col]+N1;
     }
   }
   
   //unscramble x to get output
   L = 0;
   for(int k = 0; k < N; k++)
   {
     A[k*v + col] = x[L*v + col];
     L = L + unsc;
     if(L >= N)
     	L = L-N;
   }
}



struct modMultFunctor : public thrust::binary_function<uint64,uint64,uint64>
{
    __host__ __device__
    uint64 operator() (const uint64& lhs, const uint64& rhs)
    {
		return myModMult( lhs, rhs );
	}
};

void entrywiseModMult( thrust::device_vector<uint64>& mat1, int row1, int numcols1, 
					  thrust::device_vector<uint64>& vec2, 
					  thrust::device_vector<uint64>& mat3, int row3, int numcols3  )
{
    thrust::transform( mat1.begin()+(row1*numcols1), mat1.begin()+((row1*numcols1)+numcols1), vec2.begin(), mat3.begin()+(row3*numcols3), modMultFunctor() );
}

struct newentrywisefunctor
{	
  uint64* a;
  uint64* itab;
  uint64* f;
  int N;
  int v;
    newentrywisefunctor( uint64* p_a, uint64* p_itab, uint64* p_f, int p_N, int p_v  )
	{
		a = p_a;
		itab = p_itab;
		f = p_f;
		N = p_N;
		v = p_v;
	}

    __device__ void operator() ( int i )
    {
    	//pointwise multiply vector itab of length N 
    	//with each column of a v-column matrix a (a must have N rows)
        for(int j = 0; j < N; j++)
		{
			f[j*v + i] = myModMult(a[j*v + i], itab[j]);
		}
    }
};

struct entrywise_rev_mod_mult_functor
{	
  uint64* f;
  uint64* the_hs;
  int N;
  int h;
  uint64 inv_N;
  uint64* a_extended;
  int v;

    entrywise_rev_mod_mult_functor( uint64* p_f, uint64* p_the_hs, uint64* p_a_extended, int p_N, int p_h, int p_v, uint64 p_inv_N  )
	{
		f = p_f;
		the_hs = p_the_hs;
		a_extended = p_a_extended;
		N = p_N;
		h = p_h;
		v = p_v;
		inv_N = p_inv_N;
	}

    __device__ void operator() ( int i )
    {
		for(int k = 0; k < h; k++)
		{
			a_extended[k*v+i] = myModMult(myModMult(f[(N-(h+k))*v+i], the_hs[k]), inv_N); 
		}
	}
};
	

//build the proof by extrapolating the data to adjacent columns, 
//then summing to get a vector of sums of squares, which can be extrapolated by V
void proof(thrust::device_vector<uint64>& b, int h, int v, thrust::device_vector<uint64>& itab, thrust::device_vector<uint64>& a,
		   thrust::device_vector<uint64>& f, thrust::device_vector<uint64>& transformed_f, thrust::device_vector<uint64>& transformed_invs,
		   thrust::device_vector<uint64>& the_hs, thrust::device_vector<uint64>& the_invs, thrust::device_vector<uint64>& a_extended, 
		   int N, int M, thrust::device_vector<int>& N_i, thrust::device_vector<int>& NoverN_i,
		   thrust::device_vector<int>& I1, thrust::device_vector<uint64>& answers1, thrust::device_vector<uint64>& prim_to_ij1,
		    thrust::device_vector<int>& I, thrust::device_vector<uint64>& answers, thrust::device_vector<uint64>& prim_to_ij)
{
	clock_t t = clock();

	uint64 Nth_root=myModPow(37, (PRIME-1)/N);
	uint64 unsc=0;
	for(int i = 0; i < M; i++)
	{
		unsc+=NoverN_i[i];
		if(unsc >= N)
		 unsc = unsc-N;
	}  

	uint64 inv_N = inv(N);

	thrust::counting_iterator<int> first(0);
	
	thrust::for_each(first, first+1, 
		pfa_functor(N, thrust::raw_pointer_cast( &the_invs[ 0 ] ),  thrust::raw_pointer_cast( &transformed_invs[ 0 ] ), 
		thrust::raw_pointer_cast( &N_i[ 0 ]), thrust::raw_pointer_cast( &NoverN_i[ 0 ] ), M, Nth_root, unsc, 
		thrust::raw_pointer_cast( &I1[ 0 ] ), 
		thrust::raw_pointer_cast( &prim_to_ij1[ 0 ] ),
		thrust::raw_pointer_cast( &answers1[ 0 ] ), 1));
	//cudaThreadSynchronize();

	
	thrust::for_each(first, first+v, 
		newentrywisefunctor(thrust::raw_pointer_cast(&a[0]), thrust::raw_pointer_cast(&itab[0]), thrust::raw_pointer_cast(&f[0]), h, v ));
//cudaThreadSynchronize();




	//transform f vector and save it in transformed_f
	thrust::for_each(first, first+v, 
		pfa_functor(N, thrust::raw_pointer_cast( &f[ 0 ] ),  thrust::raw_pointer_cast( &transformed_f[ 0 ] ), 
		thrust::raw_pointer_cast( &N_i[ 0 ]), thrust::raw_pointer_cast( &NoverN_i[ 0 ] ), M, Nth_root, unsc, 
		thrust::raw_pointer_cast( &I[ 0 ] ), 
		thrust::raw_pointer_cast( &prim_to_ij[ 0 ] ),
		thrust::raw_pointer_cast( &answers[ 0 ] ), (int) v ));
		

	
	//point-wise multiply transformed vectors

		thrust::for_each(first, first+v, 
		newentrywisefunctor(thrust::raw_pointer_cast(&transformed_f[0]), 
		thrust::raw_pointer_cast(&transformed_invs[0]), thrust::raw_pointer_cast(&transformed_f[0]), N, v ));
		
//cudaThreadSynchronize();

	//transform back and save in f (need to scale each entry by N^{-1} mod p)
	//cudaThreadSynchronize();
	thrust::for_each(first, first+v, 
		pfa_functor(N, thrust::raw_pointer_cast( &transformed_f[ 0 ] ),  thrust::raw_pointer_cast( &f[ 0 ] ), 
		thrust::raw_pointer_cast( &N_i[ 0 ]), thrust::raw_pointer_cast( &NoverN_i[ 0 ] ), M, Nth_root, unsc, 
		thrust::raw_pointer_cast( &I[ 0 ] ), 
		thrust::raw_pointer_cast( &prim_to_ij[ 0 ] ),
		thrust::raw_pointer_cast( &answers[ 0 ] ), v ));

	

    //at this point, f[N-(h+j)]*the_hs[j]*1/N equals what a[i][h+j] should be for j=0...h-1
	
	
		//cudaThreadSynchronize();

	
	thrust::for_each(first, first+v, entrywise_rev_mod_mult_functor(thrust::raw_pointer_cast(&f[0]),
	thrust::raw_pointer_cast(&the_hs[0]), thrust::raw_pointer_cast(&a_extended[0]), N, h, v, inv_N));

	cudaThreadSynchronize();

	std::cout << "total time in proof before transpose thing was: " << (double) (clock() - t)/CLOCKS_PER_SEC  <<  std::endl;
	
	for(int j=0; j<h; j++)
	{
	  b[j] = myMod(computef2(a, j, v, transformed_f));
	}
	for(int j = h; j < 2*h; j++)
	{
	  b[j] = myMod(computef2(a_extended, (int)j-(int)h, v, transformed_f));
	}

	return;
}


int main(int argc, char** argv)
{

    if(argc != 3)
    {
      std::cout << "There should be exactly two command line argument, specifying h and vn";
      exit(1);
    }
    int h=(uint64) atoi(argv[1]);
    int v=(uint64) atoi(argv[2]); 
	//std::cout << "made it here\n";
	if(h > v)
	{
		std::cout << "Please choose h <= v\n";
		exit(1);
	}

	uint64 r=myMod(rand()); //pick random location r

	int* host_N_i = (int*) calloc(6, sizeof(int));
    int* host_NoverN_i = (int*) calloc(6, sizeof(int));
    int N;
    int M;
    compute_N_M_N_i_and_N_overNi(&N, &M, host_N_i, host_NoverN_i, h, v);


	clock_t t = clock();
	thrust::host_vector<uint64> h_a( h*v );
	thrust::device_vector<uint64> d_a(h*v);
	thrust::host_vector<uint64> h_itab(h);
	thrust::device_vector<uint64> d_itab(h);
	thrust::host_vector<uint64> h_rtab(h);
	thrust::device_vector<uint64> d_rtab(h);
	thrust::device_vector<uint64> intermediate(v);
	thrust::host_vector<uint64> h_itab2(2*h);

	thrust::device_vector<uint64> d_itab2(2*h, 0);
	thrust::host_vector<uint64> h_rtab2(2*h);
	thrust::device_vector<uint64> d_rtab2(2*h, 0);
	thrust::device_vector<uint64> b(2*h, 0);
	thrust::host_vector<uint64> h_the_hs(h, 0);
	thrust::device_vector<uint64> d_the_hs(h);

	thrust::host_vector<uint64> h_the_invs(N, 0);
	thrust::device_vector<uint64> d_the_invs(N);
	thrust::device_vector<uint64>  f(N*v, 0);
	thrust::device_vector<uint64>  transformed_f(N*v, 0);
	thrust::device_vector<uint64>  transformed_invs(N, 0);
    thrust::device_vector<uint64> intermediate2(2*h, 0);
	thrust::device_vector<uint64> d_a_extended(h*v, 0);

	thrust::device_vector<int> I(34*v, 0);
	thrust::device_vector<uint64> answers(34*v, 0);
	thrust::device_vector<uint64> prim_to_ij(34*v, 0);

	thrust::device_vector<int> I1(34, 0);
	thrust::device_vector<uint64> answers1(34, 0);
	thrust::device_vector<uint64> prim_to_ij1(34, 0);

	thrust::generate(h_a.begin(), h_a.end(), myRand);

    d_a=h_a;


    //have prover generate his lookup tables. We charge him for this even though they
    //are data independent so he could do this in advance (this cost is quite small now anyway)

	std::cout << "time allocating memory was: " << (double) (clock() - t)/CLOCKS_PER_SEC  <<  std::endl;

	t = clock();
	
    geninv(h, h_itab); //table of denominators of lagrange polynomials

    d_itab = h_itab;
		
	
	genrtab(h, r, h_rtab);
	d_rtab=h_rtab;
	

    geninv(2*h, h_itab2); //table of denominators of lagrange polynomials

    d_itab2 = h_itab2;


	genrtab(2*h, r, h_rtab2);

	d_rtab2=h_rtab2;

	//handle all the memory allocation tasks for the proof generation function
    thrust::device_vector<int> N_i(6);
    thrust::device_vector<int>  NoverN_i(6);
    
    
    for(int i = 0; i < M; i++)
    {
      N_i[i] =host_N_i[i];
      NoverN_i[i]=host_NoverN_i[i];
    }
	
	gen_hs(h, h_the_hs);

    d_the_hs=h_the_hs;
	
	gen_invs(h, N, h_the_invs);
	d_the_invs=h_the_invs;


	std::cout << "time initializing stuff before proof was: " << (double) (clock() - t)/CLOCKS_PER_SEC  <<  std::endl;
	cudaThreadSynchronize();
	clock_t unique=clock();
	proof(b, h,  v, d_itab, d_a, f, transformed_f, transformed_invs,
		  d_the_hs, d_the_invs, d_a_extended, N, M, N_i, NoverN_i, 
		  I1, answers1, prim_to_ij1, I, answers, prim_to_ij );
		  cudaThreadSynchronize();
	double pt = (double) ((double)clock() - (double)unique)/CLOCKS_PER_SEC;
	std::cout << "time spent in proof was: " << pt << std::endl;
	
	t=clock();

	uint64 result = tabextrap(b, 0, 2*h, r, d_itab2, d_rtab2, intermediate2);
	double ct = (double) (clock() - t)/CLOCKS_PER_SEC;
   std::cout << "time checking proof was: " << ct  <<  std::endl;
	
	//whether it's faster for V to process the data column-wise or row-wise depends on
	//the dimensions of the input grid (because there is substantial overhead in processing
	//row-wise if the rows are small)
	uint64 check;
	double vt;
	cudaThreadSynchronize();
	t=clock();
	check = vcheck_by_cols(h, v, r, d_a, d_itab, d_rtab, intermediate);

	//for(int i = 0; i < 499; i++)
		//vcheck_by_cols(h, v, r, d_a, d_itab, d_rtab, intermediate);
	cudaThreadSynchronize();
	vt = (double) (clock() - t)/CLOCKS_PER_SEC;
	std::cout << "time spent in vcheck (over 500 trials) was: " << vt  <<  std::endl;

	if (result!=check)
    {
		std::cout << "FAIL!\n" << "result: " << result << " check: " << check << std::endl;
    }  
	 else
	 {
		 std::cout << "passed. result is: " << result << " check is: " << check << std::endl;
	 }

	 std::cout << "N\tVerifT\tProveT\tCheckT\tVerifS\tProofS\n";
    // print a header if requested
	 std::cout << h*v << "\t" << (double) vt << "\t" << (double) pt;
	 std::cout << "\t" <<  (double) ct  << "\t" << v << "\t" <<2*h << " " << std::endl;

	}