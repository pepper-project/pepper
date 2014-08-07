/*************************************
//Justin Thaler
//May 6, 2011. 
//Implementation of protocol described in 
//"Delegating Computation: Interactive Proofs for Muggles" by Goldwasser, Kalai, and Rothblum.
//Currently this implementation supports, F2, F0, pattern matching, and matrix multiplication. 
//The starting point for this implementation was a sequential program originally written for the 
//paper "Practical Verified Computation with Streaming Interactive Proofs" by Cormode, Mitzenmacher, 
//and Thaler. This implementation utilizes GPUs, and was written for the
//paper "Verifying Computations with Massively Parallel Interactive Proofs" by Thaler,
//Roberts, Mitzenmacher, and Pfister.
**************************************/

#include <cstdlib>
#include <iostream>
#include <time.h>
#include <omp.h>
#include <math.h>

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
#define CUTOFF 5
#define LOG_PAT_LEN 3
#define PAT_MASK 7
#define PAT_LEN 8

#define TEST 1

__host__ __device__ int TYPE_TWO(int j, int level_length, int level_length_ip1)
{
	return 1;
}

__host__ __device__ int IN1_TWO(int j, int level_length, int level_length_ip1)
{
	return j;
}

__host__ __device__ int IN2_TWO(int j, int level_length, int level_length_ip1)
{
	if(j & 1)
		return j;
	else
		return j+1;
}

__host__ __device__ int TYPE_THREE(int j, int level_length, int level_length_ip1)
{
	return (j&1);
}

__host__ __device__ int IN1_THREE(int j, int level_length, int level_length_ip1)
{
		if( (j & 1) == 0)
			return level_length_ip1-1;
		else	
			return (j>>1);
}
   
__host__ __device__ int IN2_THREE(int j, int level_length, int level_length_ip1)
{
	return (j>>1);
}

__host__ __device__ int TYPE_FOUR(int index, int level_length, int level_length_ip1, int n_squared)
{
	if(index >= level_length-n_squared-1)
		return 0;
	return 1;
}

__host__ __device__ int IN1_FOUR(int index, int level_length, int level_length_ip1, int d, int two_pow_d, int n_squared)
{
	if(index >= level_length-n_squared-1)
	{
		int gap = level_length - index;
		return level_length_ip1 - gap;
	}
	else
	{
		int k = (index & (two_pow_d-1));
		int i = ((index >> (2*d)) & (two_pow_d-1));
		return ((i << d) + k);
	}
}

__host__ __device__ int IN2_FOUR(int index, int level_length, int level_length_ip1,  int d, int two_pow_d, int n_squared)
{
	if(index >= level_length-n_squared-1)
	{
		return level_length_ip1 - 1;
	}
	else
	{
		int k = (index & (two_pow_d-1));
		int j = ((index >> d) & (two_pow_d-1));
		int ret = ((1 << 2*d) + (k << d) + j);
		return ret;
	}
}

__host__ __device__ int IN1_FIVE(int index, int level_length, int level_length_ip1,  int d, int two_pow_d, int n_squared)
{
	if(index >= level_length-n_squared-1)
	{
		int gap = level_length - index;
		return level_length_ip1 - gap;
	}
	else
	{
		return 2*index;
	}
}

__host__ __device__ int IN2_FIVE(int index, int level_length, int level_length_ip1,  int d, int two_pow_d, int n_squared)
{
	if(index >= level_length-n_squared-1)
	{
		return level_length_ip1 - 1;
	}
	else
	{
		return 2*index+1;
	}
}

__host__ __device__ int IN1_SIX(int index, int level_length, int level_length_ip1)
{
	if(index == level_length-1)
		return level_length_ip1-1;
	return index;
}

__host__ __device__ int IN2_SIX(int index, int level_length, int level_length_ip1, int n_squared)
{
	if(index == level_length)
		return level_length_ip1-1;
	return index + n_squared;
}


__host__ __device__ int IN1_SEVEN(int index, int n)
{
	int i = index >> LOG_PAT_LEN;
	int j = index & PAT_MASK;
	if(i != n-1)
		return i+j;
	return n-1;
}

__host__ __device__ int IN2_SEVEN(int index, int n)
{
	int j = index & PAT_MASK;
	int i = index >> LOG_PAT_LEN;
	if(i != n-1)
		return n+j;
	return n-1+PAT_LEN;
}


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

__host__ __device__
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

/*__host__ __device__
uint64 myModMult(uint64 x, uint64 y)
{
  uint64 hi_x = x >> 32;
  uint64 hi_y = y >> 32;

  uint64 low_x = x & MASK;
  uint64 low_y = y & MASK;


  //since myMod might return something slightly large than 2^61-1,
  //we need to multiply by 8 in two pieces to avoid overflow. Gross.
  uint64 piece1 = myMod( ( myMod( ( myMod( hi_x * hi_y ) << 2 ) ) << 1 ) );

  uint64 z     = myMod( hi_x * low_y );
  uint64 hi_z  = z >> 32;
  uint64 low_z = z & MASK;

  //Note 2^64 mod (2^61-1) is 8
  uint64 piece2 = myMod(( hi_z << 3 ) + myMod( low_z << 32 ) );

  z     = myMod( hi_y * low_x );
  hi_z  = z >> 32;
  low_z = z & MASK;

  uint64 piece3 = myMod( ( hi_z << 3 ) + myMod( ( low_z << 32 ) ) );

  uint64 piece4 = myMod( low_x * low_y );
  uint64 result = myMod( piece1 + piece2 + piece3 + piece4 );

  return result;
} */ 

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
__host__ __device__ void extEuclideanAlg(uint64 u, uint64* u1, uint64* u2, uint64* u3)
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
__host__ __device__ uint64 inv(uint64 a)
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

//circuit data structure
typedef struct circ
{
  int* log_level_length; //needs to be ceiling of log_level_length (controls number of vars in various polynomials)
  uint64* level_length; //number of gates at each level of the circuit
  thrust::host_vector<int> type; //two-d array of gate types
  thrust::host_vector<int> in1;
  thrust::host_vector<int> in2;
  thrust::host_vector<uint64> val;
  thrust::host_vector<uint64> in1_vals;
  thrust::host_vector<uint64> in2_vals;
  int num_levels; //number of levels of circuit
} circ;

void destroy_circ(circ* c)
{
  free(c->log_level_length);
  free(c->level_length);
  free(c);
}

struct modular_sum : public thrust::binary_function<uint64,uint64,uint64>
{
    __host__ __device__
    uint64 operator() (const uint64& lhs, const uint64& rhs)
    {
		return myMod( myMod(lhs) + myMod(rhs) );
	}
};

struct modular_mult : public thrust::binary_function<uint64,uint64,uint64>
{
    __host__ __device__
    uint64 operator() (const uint64& lhs, const uint64& rhs)
    {
		return myModMult( lhs, rhs );
	}
};


struct chi_func
{
	enum TupleLayout
	{
		A,
		B,
		C,
		RESULT
	};
	
    chi_func() {}

	template <typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        uint64 i = thrust::get< A >( tuple );                                              
        uint64 r_i = thrust::get< B >( tuple );                                              
        uint64 x = thrust::get< C >( tuple );                                              

		uint64 result;
		if( (x >> i) & 1)
			result = r_i;
		else
			result = 1 + PRIME - r_i;		
        thrust::get< RESULT >( tuple ) = result;
    }
};

//computes chi_v(r), where chi is the Lagrange polynomial that takes
//boolean vector v to 1 and all other boolean vectors to 0. (we view v's bits as defining
//a boolean vector. n is dimension of this vector.
//all arithmetic done mod p
uint64 chi(uint64 v, thrust::device_vector<uint64>& r, uint64 n, int rskip)
{
	thrust::device_vector<uint64> scratch(n, 0);
	thrust::device_vector<uint64> v_array(n);
	thrust::fill(v_array.begin(), v_array.end(), v);
	thrust::counting_iterator<int> count(0);

	  thrust::for_each(

        thrust::make_zip_iterator( thrust::make_tuple( count, r.begin() + rskip, v_array.begin(), scratch.begin()) ),

        thrust::make_zip_iterator( thrust::make_tuple( count +n, r.begin()+ rskip + n, v_array.begin() + n, scratch.begin() + n )),

        chi_func() );
	   return thrust::reduce(scratch.begin(), scratch.end(), (uint64) 1, modular_mult());
}

__device__ uint64 d_chi(uint64 v, uint64* r, uint64 n, int rskip)
{ 
  uint64 x=v;
  uint64 c = 1;
  for(uint64 i = 0; i <n; i++)
  {
    if( x&1 )
      c=myModMult(c, r[rskip + i]);
    else
      c=myModMult(c, 1 + PRIME - r[rskip + i]);
    x=x>>1;
  }
  return c;
}


//extrapolate the polynomial implied by vector vec of length n to location r
uint64 extrap(thrust::device_vector<uint64>& vec, int start_index, uint64 n, uint64 r)
{
  uint64 result=0;
  uint64 mult=1;
  for(uint64 i = 0; i < n; i++)
  {
    mult=1;
    for(uint64 j=0; j < n; j++)
    {
      if (i>j)
        mult=myModMult(myModMult(mult, myMod(r-j+PRIME)), inv(i-j));
      if (i<j)
        mult=myModMult(myModMult(mult, myMod(r-j+PRIME)), inv(myMod(i+PRIME-j)));
    }
    result=myMod(result+myModMult(mult, vec[start_index + i]));
  }
  return result;
}

//extrapolate the polynomial implied by vector vec of length n to location r
uint64 host_extrap(uint64* vec, uint64 n, uint64 r)
{
  uint64 result=0;
  uint64 mult=1;
  for(uint64 i = 0; i < n; i++)
  {
    mult=1;
    for(uint64 j=0; j < n; j++)
    {
      if (i>j)
        mult=myModMult(myModMult(mult, myMod(r-j+PRIME)), inv(i-j));
      if (i<j)
        mult=myModMult(myModMult(mult, myMod(r-j+PRIME)), inv(myMod(i+PRIME-j)));
    }
    result=myMod(result+myModMult(mult, vec[i]));
  }
  return result;
}


struct eval_V_i_functor
{	
	uint64* level_i;
	uint64* r;
	uint64* scratch;
	int mi;
	int rskip;

    eval_V_i_functor(uint64* p_level_i, uint64* p_r, int p_mi, int p_rskip, uint64* p_scratch) 
	{	
		level_i = p_level_i;
		r = p_r;
		mi = p_mi;
		rskip = p_rskip;
		scratch = p_scratch;
	}
		
    __device__ void operator() ( int k)
    {
       scratch[k] = myModMult(level_i[k], d_chi(k, r, mi, rskip));
	}
};

//evaluates V_i polynomial at location r.
//Here V_i is described in GKR08; it is the multi-linear extension
//of the vector of gate values at level i of the circuit
uint64 evaluate_V_i(int mi, int ni, thrust::device_vector<uint64>& level_i,
					 thrust::device_vector<uint64>& r, int rskip, thrust::device_vector<uint64>& scratch)
{
  thrust::counting_iterator<int> count(0);
  thrust::for_each(count, count + ni, eval_V_i_functor(thrust::raw_pointer_cast(&(level_i[0])), 
	  thrust::raw_pointer_cast(&(r[0])), mi, rskip, thrust::raw_pointer_cast(&(scratch[0]))));

 return thrust::reduce(scratch.begin(), scratch.begin() + ni, (uint64) 0, modular_sum());
}


//assuming the input level of circuit c contains the values of the input gates
//evaluate the circuit c, filling in the values of all internal gates as well.
void evaluate_circuit_seq(circ* c, thrust::device_vector<uint64>& cur_in1_vals, thrust::device_vector<uint64>& cur_in2_vals, 
thrust::device_vector<uint64>& vals_above, int real_d, int two_pow_d, int n_squared)
{
  int d = c->num_levels-1;
  int start_index_ip1 = 0;
  int start_index = c->level_length[d];
  thrust::counting_iterator<int> first(0);
  int m_in1_func;
  int m_in2_func;
  int m_type_func;
  int m_level_length;
	  int m_level_length_ip1;
  for(int i = d-1; i >= 0; i--)
  {	
	   m_in1_func = c->in1[i];
		m_in2_func = c->in2[i];
		m_type_func = c->type[i];
		m_level_length = c->level_length[i];
		m_level_length_ip1 = c->level_length[i+1];
	   for(int j = 0; j < c->level_length[i]; j++)
	   {
		   int in1=0;
		   int in2=0;
		   int type;
		   if(m_in1_func == 0)
			in1 = j;
		   else if(m_in1_func == 1)
			in1 = 2*j;
		   else if(m_in1_func == 2)
			   in1 = IN1_TWO(j, m_level_length, m_level_length_ip1);
		   else if(m_in1_func == 3)
			   in1 = IN1_THREE(j, m_level_length, m_level_length_ip1);
		   else if(m_in1_func == 4)
			   in1 = IN1_FOUR(j, m_level_length, m_level_length_ip1, real_d, two_pow_d, n_squared);
		   else if(m_in1_func == 5)
			   in1 = IN1_FIVE(j, m_level_length, m_level_length_ip1, real_d, two_pow_d, n_squared);
		   else if(m_in1_func == 6)
			   in1 = IN1_SIX(j, m_level_length, m_level_length_ip1);
		   else if(m_in1_func == 7)
				in1 = IN1_SEVEN(j, two_pow_d);

		   if(m_in2_func == 0)
			in2 = j;
		   else if(m_in2_func == 1)
			in2 = 2*j+1;
		   else if(m_in2_func == 2)
			   in2 = IN2_TWO(j, m_level_length, m_level_length_ip1);
		   else if(m_in2_func == 3)
			   in2 = IN2_THREE(j, m_level_length, m_level_length_ip1);
		   else if(m_in2_func == 4)
			   in2 = IN2_FOUR(j, m_level_length, m_level_length_ip1, real_d, two_pow_d, n_squared);
		   else if(m_in2_func == 5)
			   in2 = IN2_FIVE(j, m_level_length, m_level_length_ip1, real_d, two_pow_d, n_squared);
			else if(m_in2_func == 6)
			   in2 = IN2_SIX(j, m_level_length, m_level_length_ip1, n_squared);
 		   else if(m_in2_func == 7)
				in2 = IN2_SEVEN(j, two_pow_d);

		  if(m_type_func == 0)
			type = 0;
		  else if(m_type_func == 1)
			type = 1;
		  else if(m_type_func == 2)
			type = TYPE_TWO(j, m_level_length, m_level_length_ip1);
		  else if(m_type_func == 3)
			type = TYPE_THREE(j, m_level_length, m_level_length_ip1);
		  else if(m_type_func == 4)
			type = TYPE_FOUR(j, m_level_length, m_level_length_ip1, n_squared);
		

		   uint64 val1= c->val[start_index_ip1+in1];
		   uint64 val2= c->val[start_index_ip1+in2];
	   
		   c->in1_vals[start_index + j] = val1;
		   c->in2_vals[start_index + j] = val2;
		   if(type == 0)
			 c->val[start_index + j]=myMod(val1+val2);
		   else 
			 c->val[start_index + j]=myModMult(val1, val2);
	   }
	start_index_ip1 = start_index;
	start_index = start_index + c->level_length[i];
  }
}    

//evaluates beta_z polynomial (described in GKR08)
//at location k (the bits of k are interpreted as a boolean vector). mi is dimension of k and r.
__device__ __host__ uint64 evaluate_beta_z_ULL(int mi, uint64* z, uint64 k)
{
  uint64 ans=1;
  uint64 x=k;
  uint64 zi;
  for(int i = 0; i < mi; i++)
  {
	zi = z[i];
    if(x&1)
      ans = myModMult(ans, zi);
    else
      ans = myModMult(ans, 1+PRIME-zi);
    x = x >> 1;
  }
  return ans;
}


struct init_betar_functor
{	
    uint64* m_betar;
	uint64 m_mi;
	uint64* m_z;
	
    init_betar_functor(uint64* betar, uint64 mi, uint64* z)
	{	
		m_mi = mi;
		m_z = z;
		m_betar=betar;
	}
		
    __device__ void operator() ( int k)
    {
       m_betar[k] = evaluate_beta_z_ULL(m_mi, m_z, k);
	}
};

struct vstream_functor
{
  int level_length;
  int level_length_ip1;
  uint64 betar;
  int n_squared;
  uint64* cur_addormultr;
  uint64* temp1;
  uint64 plus;
  uint64 mult;
  int type_func;
  
  vstream_functor(int p_level_length, int p_level_length_ip1, uint64 p_betar, int p_n_squared,
  uint64* p_cur_addormultr, uint64* p_temp1, uint64 p_plus, uint64 p_mult, int p_type_func)
  {
  	level_length = p_level_length;
  	level_length_ip1 = p_level_length_ip1;
  	betar = p_betar;
  	n_squared = p_n_squared;
  	cur_addormultr = p_cur_addormultr;
  	temp1 = p_temp1;
  	plus = p_plus;
  	mult = p_mult;
  	type_func = p_type_func;
  }
  
  __device__ void operator() ( int k)
	{
	  int type;
	  if(type_func == 0)
	    type = 0;
	  else if(type_func == 1)
	    type = 1;
	  else if(type_func == 2)
	    type = TYPE_TWO(k, level_length, level_length_ip1);
	  else if(type_func == 3)
	    type = TYPE_THREE(k, level_length, level_length_ip1);
	  else if(type_func == 4)
		type = TYPE_FOUR(k, level_length, level_length_ip1, n_squared);
		
	  if(type==0)
		  temp1[k] = myModMult(betar, myModMult(cur_addormultr[k], plus));
	  else
		  temp1[k] = myModMult(betar, myModMult(cur_addormultr[ k], mult));
	  }
};

struct first_j_functor
{	
	int m_start_index;
	int m_start_index_ip1;
    uint64* m_betar;
	uint64* m_z;
	uint64* m_addormultr;
	int m_j;
	uint64* m_r;
	uint64* m_temp1;
	uint64* m_temp2;
	uint64* m_temp3;
	uint64* m_in1_vals; 
	uint64* m_in2_vals;
	uint64 mi;
	int m_type_func;
	int level_length;
	int level_length_ip1;
	int n_squared;

    first_j_functor(int p_level_length, int p_level_length_ip1, int start_index, int start_index_ip1, int j, uint64* betar,
	 uint64* z, uint64* addormultr, uint64* r, uint64* temp1, uint64* temp2, uint64* temp3,
		uint64* in1_vals, uint64* in2_vals, int p_mi, int type_func, int p_n_squared)
	{	
		m_start_index = start_index;
		m_j = j;
		m_start_index_ip1 = start_index_ip1;
		m_betar = betar;
		m_z = z;
		m_addormultr = addormultr;
		m_r = r;
		m_temp1 = temp1;
		m_temp2 = temp2;
		m_temp3 = temp3;
		m_in1_vals = in1_vals;
		m_in2_vals = in2_vals;
		mi = p_mi;
		m_type_func = type_func;
		level_length=p_level_length;
		level_length_ip1 = p_level_length_ip1;
		n_squared = p_n_squared;
	}
		
    __device__ void operator() ( int k)
    {
      //in cases where V1 or V2 are trivial to compute, compute them now
        uint64 V1 = m_in1_vals[ k];
		uint64 V2 = m_in2_vals[k];

	    uint64 aorm;
		uint64 betar;
        //prep betar fields because in this round variable j will take on values 0, 1, or 2 rather than just 0,1
        uint64 kshiftj = k>>m_j;
        if( kshiftj & 1)
           m_betar[ k] = myModMult(m_betar[k], inv(m_z[m_j])); 
        else
           m_betar[ k] = myModMult(m_betar[+ k], inv(1+PRIME-m_z[m_j])); 

		uint64 old_aorm = 1;
		if(m_j > CUTOFF)
		 old_aorm = m_addormultr[ k ];
		else
		{
			for(int i = 0; i < m_j; i++)
			{
				if((k >> i) & 1)
					old_aorm = myModMult(old_aorm, m_r[i]);
				else
					old_aorm = myModMult(old_aorm, 1+PRIME-m_r[i]);
			}
		}
		//now we iterate over the points at which we evaluate the polynomial for this round
        for(int m = 0; m <= 2; m++)
        {
           //compute betar for this gate for this round, and update betar field if we're done with it for this round (m==2)
           if(m==0)
             betar = myModMult(m_betar[k], myModMult(1+PRIME-m_z[m_j], 1+PRIME-m));
           else if(m==1)
             betar=myModMult(m_betar[ k], myModMult(m_z[m_j], m));
           else
			   betar = myModMult(m_betar[k], myMod(myModMult(1+PRIME-m_z[m_j], 1+PRIME-m) + myModMult(m_z[m_j], m)));
           if(m==2)  
			m_betar[ k] = myModMult(m_betar[ k],
                  		myMod(myModMult(1+PRIME-m_z[m_j], 1+PRIME-m_r[m_j]) + myModMult(m_z[m_j], m_r[m_j])));
		   
          //compute addormult for this gate for this round, and update the field if we're done with it for this round (m==2)
          if(kshiftj & 1)
          {
            if(m==0) 
              continue;
            else
              aorm=myModMult(old_aorm, m);
            if((m==2) && ((m_j >= CUTOFF-1) || m_j == (mi-1)))
              m_addormultr[  k ] = myModMult(old_aorm, m_r[m_j]);
          }
          else
          {
            if(m==1)
              continue;
            else 
              aorm=myModMult(old_aorm, 1+PRIME-m);
            if((m==2) && ((m_j >= CUTOFF) || m_j == (mi-1)))
              m_addormultr[ k ] = myModMult(old_aorm, 1+PRIME-m_r[m_j]);
          }
          //finally, update the evaluation of this round's polynomial at m based on this gate's contribution
        //to beta, add_i or multi_i and V_{i+1}(omega1) and V_{i+1}(omega2)

		int type;

	if(m_type_func == 0)
	    type = 0;
	  else if(m_type_func == 1)
	    type = 1;
	  else if(m_type_func == 2)
	    type = TYPE_TWO(k, level_length, level_length_ip1);
	  else if(m_type_func == 3)
	    type = TYPE_THREE(k, level_length, level_length_ip1);
	  else if(m_type_func == 4)
		type = TYPE_FOUR(k, level_length, level_length_ip1, n_squared);

        if(type == 0)
        {
			if(m == 0)
				m_temp1[ k ]=myModMult(betar, myModMult(aorm, myMod(V1 + V2)));
			else if(m == 1)
				m_temp2[ k ]=myModMult(betar, myModMult(aorm, myMod(V1 + V2)));
			else
				m_temp3[ k ]=myModMult(betar, myModMult(aorm, myMod(V1 + V2)));
        }
        else 
        {
			if(m == 0)
				m_temp1[ k ]=myModMult(betar, myModMult(aorm, myModMult(V1, V2)));
			else if(m == 1)
				m_temp2[ k ]=myModMult(betar, myModMult(aorm, myModMult(V1, V2)));
			else
				m_temp3[ k ]=myModMult(betar, myModMult(aorm, myModMult(V1, V2)));
        }
       } //end m loop
     } //end k loop (gates)
};

struct third_j_functor
{	
	int start_index;
	int start_index_ip1;
    uint64 betartimesV1;
	uint64* addormultr;
	int j;
	uint64* r;
	uint64* temp1;
	uint64* temp2;
	uint64* temp3;
	uint64* vals;
	int mi;
	int mip1;
	uint64 betar;
	uint64 V1;
	int type_func;
	int in1_func;
	int in2_func;
	int level_length;
	int level_length_ip1;
	int d;
	int two_pow_d;
	int n_squared;

    third_j_functor(int p_level_length, int p_level_length_ip1, int p_start_index, int p_start_index_ip1, int p_j, uint64 p_betar, uint64 p_betartimesV1, 
	uint64* p_addormultr, uint64* p_r, uint64* p_temp1, uint64* p_temp2, 
		uint64* p_temp3, uint64* p_vals, int p_mi, int p_mip1, uint64 p_V1, int p_type_func, int p_in1_func, int p_in2_func, 
		int p_d, int p_two_pow_d, int p_n_squared)
	{	
		 start_index = p_start_index;
		start_index_ip1 = p_start_index_ip1;
		betartimesV1 = p_betartimesV1;
		addormultr = p_addormultr;
		 j = p_j;
		 r = p_r;
		 temp1 = p_temp1;
		 temp2 = p_temp2;
		 temp3 = p_temp3;
		vals = p_vals;
		mi = p_mi;
		mip1 = p_mip1;
		betar=p_betar;
		V1 = p_V1;
		type_func = p_type_func;
		in1_func = p_in1_func;
		in2_func = p_in2_func;
		level_length = p_level_length;
		level_length_ip1 = p_level_length_ip1;
		d = p_d;
		two_pow_d = p_two_pow_d;
		n_squared = p_n_squared;
	}
    __device__ void operator() ( int k)
    {
	  uint64 index;
	  uint64 aorm;
	  uint64 V2;
	  int in2;

      //now we iterate over the points at which we evaluate the polynomial for this round
      for(int m = 0; m <= 2; m++)
      {
	    if(in2_func == 0)
		in2 = k;
	   else if(in2_func == 1)
		in2 = 2*k+1;
	   else if(in2_func == 2)
		   in2 = IN2_TWO(k, level_length, level_length_ip1);
	   else if(in2_func == 3)
		   in2 = IN2_THREE(k, level_length, level_length_ip1);
	   else if(in2_func == 4)
		   in2 = IN2_FOUR(k, level_length, level_length_ip1, d, two_pow_d, n_squared);
	   else if(in2_func == 5)
		   in2 = IN2_FIVE(k, level_length, level_length_ip1, d, two_pow_d, n_squared);
	    else if(in2_func == 6)
		   in2 = IN2_SIX(k, level_length, level_length_ip1, n_squared);
	   else if(in2_func == 7)
		   in2 = IN2_SEVEN(k, two_pow_d);


          //compute contribution to V_{i+1}(omega2) for this gate
          index=in2 >> (j-mi-mip1);
          if(index & 1)
          {
            if(m==0)
              continue;
            if(m==1)
              V2=vals[index];
            else 
              V2=myMod(myModMult(vals[index], m)+myModMult(vals[index-1], PRIME+1-m));
            aorm=myModMult(addormultr[ k], m);
            if(m==2)
              addormultr[k] = myModMult(addormultr[k], r[j]);
           } 
           else
           {
             if(m==1)
               continue;
             if(m==0)
               V2=vals[index];
             else   
               V2=myMod(myModMult(vals[index+1], m)+myModMult(vals[index], PRIME+1-m));
             aorm=myModMult(addormultr[ k], 1+PRIME-m);
             if(m==2)
               addormultr[k]= myModMult(addormultr[ k], 1+PRIME-r[j]);
           }
        
        //finally, update the evaluation of this round's polynomial at m based on this gate's contribution
        //to beta, add_i or multi_i and V_{i+1}(omega1) and V_{i+1}(omega2)
		int type;
		if(type_func == 0)
			type = 0;
		else if(type_func == 1)
			type = 1;
		else if(type_func == 2)
			type = TYPE_TWO(k, level_length, level_length_ip1);
		else if(type_func == 3)
			type = TYPE_THREE(k, level_length, level_length_ip1);
		else if(type_func == 4)
				type = TYPE_FOUR(k, level_length, level_length_ip1, n_squared);
		
	    if(type == 0)
        {
			if(m == 0)
				temp1[ k ]=myModMult(betar, myModMult(aorm, myMod(V1 + V2)));
			else if(m == 1)
				temp2[ k ]=myModMult(betar, myModMult(aorm, myMod(V1 + V2)));
			else
				temp3[ k ]=myModMult(betar, myModMult(aorm, myMod(V1 + V2)));
        }
        else 
        {
			if(m == 0)
				temp1[ k ]=myModMult(betartimesV1, myModMult(aorm, V2));
			else if(m == 1)
				temp2[ k ]=myModMult(betartimesV1, myModMult(aorm, V2));
			else
				temp3[ k ]=myModMult(betartimesV1, myModMult(aorm, V2));
        }

      } // end m loop    
	}
};

struct second_j_functor
{	
	int start_index;
	int start_index_ip1;
    uint64 betar;
	uint64* addormultr;
	int j;
	uint64* r;
	uint64* temp1;
	uint64* temp2;
	uint64* temp3;
	uint64* vals;
	int mi;
	int mip1;
	uint64* in2_vals;
	int type_func;
	int in1_func;
	int level_length;
	int level_length_ip1;
	int d; 
	int two_pow_d;
	int n_squared;


    second_j_functor(int p_level_length, int p_level_length_ip1, int p_start_index, int p_start_index_ip1, int p_j, uint64 p_betar,
	 uint64* p_addormultr, uint64* p_r, uint64* p_temp1, uint64* p_temp2, uint64* p_temp3, uint64* p_vals, int p_mi, int p_mip1,
		uint64* p_in2_vals, int p_type_func, int p_in1_func, int p_d, int p_two_pow_d, int p_n_squared)
	{	
		 start_index = p_start_index;
		 start_index_ip1 = p_start_index_ip1;
		addormultr = p_addormultr;
		 j = p_j;
		 r = p_r;
		 temp1 = p_temp1;
		 temp2 = p_temp2;
		 temp3 = p_temp3;
		vals = p_vals;
		mi = p_mi;
		mip1 = p_mip1;
		betar=p_betar;
		in2_vals = p_in2_vals;
		in1_func = p_in1_func;
		type_func = p_type_func;
		level_length = p_level_length;
		level_length_ip1 = p_level_length_ip1;
		d=p_d; 
		two_pow_d = p_two_pow_d;
		n_squared = p_n_squared;
	}
    __device__ void operator() ( int k)
    {
	  uint64 index;
	  uint64 aorm;
	  uint64 V1;
	  uint64 V2;
	  int in1;

		V2 = in2_vals[k];
		  //now we iterate over the points at which we evaluate the polynomial for this round
      		 for(int m = 0; m <= 2; m++)
      		{
		  //now compute contribution to V_i+1(omega1) for this gate
		   
		   if(in1_func == 0)
			in1 = k;
		   else if(in1_func == 1)
			in1 = 2*k;
		   else if(in1_func == 2)
			   in1 = IN1_TWO(k, level_length, level_length_ip1);
		   else if(in1_func == 3)
			   in1 = IN1_THREE(k, level_length, level_length_ip1);
		   else if(in1_func == 4)
			   in1 = IN1_FOUR(k, level_length, level_length_ip1, d, two_pow_d, n_squared);
		   else if(in1_func == 5)
			   in1 = IN1_FIVE(k, level_length, level_length_ip1, d, two_pow_d, n_squared);
		   else if(in1_func == 6)
			   in1 = IN1_SIX(k, level_length, level_length_ip1);
		   else if(in1_func == 7)
			   in1 = IN1_SEVEN(k, two_pow_d);


           	  index=in1 >> (j-mi);
          	 if(index & 1)
           	{
  	           if(m==0)
  	            continue; 
  	           else if(m==1)
  	  	          V1=vals[index];
  	           else
         		     V1=myMod(myModMult(vals[index], m)+myModMult(vals[index-1], PRIME+1-m));
              	//now compute contribution to tilde{add}_i or tilde{multi}_i for this gate
             	aorm=myModMult(addormultr[k], m);
             	if(m==2)
               		addormultr[ k] = myModMult(addormultr[ k], r[j]);
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
             		aorm=myModMult(addormultr[ k], 1+PRIME-m);
             		if(m==2)
               			addormultr[ k] = myModMult(addormultr[ k], 1+PRIME-r[j]);
		 	}  
		
        		//finally, update the evaluation of this round's polynomial at m based on this gate's contribution
       	 	//to beta, add_i or multi_i and V_{i+1}(omega1) and V_{i+1}(omega2)

			int type;
			if(type_func == 0)
				type = 0;
			else if(type_func == 1)
				type = 1;
			else if(type_func == 2)
				type = TYPE_TWO(k, level_length, level_length_ip1);
			else if(type_func == 3)
				type = TYPE_THREE(k, level_length, level_length_ip1);
			else if(type_func == 4)
				type = TYPE_FOUR(k, level_length, level_length_ip1, n_squared);
      		if(type ==0)
			{
     			if(m == 0)
					temp1[ k ]=myModMult(betar, myModMult(aorm, myMod(V1 + V2)));
				else if(m == 1)
					temp2[ k ]=myModMult(betar, myModMult(aorm, myMod(V1 + V2)));
				else
					temp3[ k ]=myModMult(betar, myModMult(aorm, myMod(V1 + V2)));
			}
			else 
			{
				if(m == 0)
					temp1[ k ]=myModMult(betar, myModMult(aorm, myModMult(V1, V2)));
				else if(m == 1)
					temp2[ k ]=myModMult(betar, myModMult(aorm, myModMult(V1, V2)));
				else
					temp3[ k ]=myModMult(betar, myModMult(aorm, myModMult(V1, V2)));
			}

	  } //end m loop
	}
};

struct vals_update_functor
{	
	uint64* vals;
	uint64* r;
	int j;
	uint64* temp;
	
    vals_update_functor(uint64* p_vals, uint64* p_r, int p_j, uint64* p_temp)
	{	
		vals = p_vals;
		r = p_r;
		j=p_j;
		temp = p_temp;
	}

    __device__ void operator() ( int index)
    {
       uint64 k = index << 1;
	   temp[index] = myMod(myModMult( vals[k], 1+PRIME-r[j-1] ) + myModMult(vals[k+1], r[j-1]));
	}
};

struct first_vals_update_functor
{	
	uint64* vals;
	uint64* r;
	int j;
	uint64* temp1;
	uint64* temp2;
	uint64* temp_vals;
	
    first_vals_update_functor(uint64* p_vals, uint64* p_r, int p_j, uint64* p_temp1, uint64* p_temp2, uint64* p_temp_vals)
	{	
		vals = p_vals;
		r = p_r;
		j=p_j;
		temp1 = p_temp1;
		temp2 = p_temp2;
		temp_vals = p_temp_vals;
	}

    __device__ void operator() ( int index)
    {
	   int k = (index << 1);
	   temp1[index]=vals[k];
	   temp2[index]=vals[k+1];
       temp_vals[index]=myMod(myModMult(vals[k], 1+PRIME-r[j]) + myModMult(vals[k+1], r[j]));
	}
};

unsigned long long check_first_level(circ* c, thrust::device_vector<uint64>& r, thrust::device_vector<uint64>& zi, 
									 unsigned long long d, thrust::device_vector<uint64>& vals,  thrust::device_vector<uint64>& temp_vals, 
									 thrust::device_vector<uint64>& temp1, thrust::device_vector<uint64>& temp2,
									 thrust::host_vector<uint64>& poly, int start_index)
{
  thrust::copy(c->val.begin() + start_index, c->val.begin() + start_index + c->level_length[d], vals.begin());

  unsigned long long check=0;

  for(int i =0; i < c->log_level_length[d]; i++)
	r[i] = rand();

  clock_t t=clock();
  thrust::counting_iterator<int> first(0);


  int num_effective = myPow(2, c->log_level_length[d]);
  for(int j=0; j < c->log_level_length[d]; j++) // computes all the messages from prover to verifier
  {
      thrust::for_each(first, first+(num_effective >> 1),
              
	    first_vals_update_functor(thrust::raw_pointer_cast(&(vals[0])), thrust::raw_pointer_cast(&(r[0])), j,
		thrust::raw_pointer_cast(&(temp1[0])), thrust::raw_pointer_cast(&(temp2[0])), thrust::raw_pointer_cast(&(temp_vals[0])) ));

		thrust::copy(temp_vals.begin(), temp_vals.begin() + (num_effective >> 1), vals.begin());

		poly[2*j] = thrust::reduce(temp1.begin(), temp1.begin() + (num_effective >> 1), ((uint64) 0), modular_sum() );
		poly[2*j+1] = thrust::reduce(temp2.begin(), temp2.begin() + (num_effective >> 1), ((uint64) 0), modular_sum());
		//num_effective tracks the number of points at which we need to evaluate Vi+1
		//for this round of the protocol (it halves every round, as one more variable becomes 'bound')
		num_effective = num_effective >> 1;
  }
  clock_t pt=clock()-t; //prover time


  std::cout << "claimed circuit value is: " << (unsigned long long) myMod(poly[0] + poly[1]) << std::endl;
  t=clock();
  for(unsigned long long j = 0; j < c->log_level_length[d]; j++) //checks all messages from prover to verifier
  {
    if (j>0)
    {
      if( (check != myMod(poly[2*j] + poly[2*j+1])) && (check != myMod(poly[2*j] + poly[2*j+1]) + PRIME)
          && (check + PRIME != myMod(poly[2*j] + poly[2*j+1])) ) // check consistency of messages
      {
		  std:: cout << "first-level check failed: j is " << j << "check is: ";
		  std::cout << check << "!=" <<  myMod(poly[2*j] + poly[2*j+1]);
		  std::cout << "poly[" << j << "][0]: " << poly[2*j] << " poly[" << j << "][1]: " << poly[2*j+1] << std::endl;
      }
    }
    check= myMod(myModMult(poly[2*j], 1+PRIME-r[j]) + myModMult(poly[2*j+1], r[j])); //compute next check value at random location r
  }
  thrust::copy(r.begin(), r.begin() + c->log_level_length[d], zi.begin());

  return check;
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


uint64 check_level(circ* c, int i, int start_index, int start_index_ip1, thrust::device_vector<uint64>& z, thrust::host_vector<uint64>& h_z, uint64 ri, 
				   int* com_ct, int* rd_ct, thrust::device_vector<uint64>& r, thrust::host_vector<uint64>& poly, thrust::device_vector<uint64>& vals, 
				   thrust::device_vector<uint64>& temp_vals, 
uint64 (*add_ifnc)(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1), 
uint64 (*mult_ifnc)(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1), thrust::device_vector<uint64>& temp1, 
thrust::device_vector<uint64>& temp2, thrust::device_vector<uint64>& temp3, 
thrust::device_vector<uint64>& cur_in1_vals, thrust::device_vector<uint64>& cur_in2_vals, 
thrust::device_vector<uint64>& cur_betar, thrust::device_vector<uint64>& cur_addormultr, int d, int two_pow_d, int n_squared, int vstream)
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




  thrust::copy(c->in1_vals.begin() + start_index, c->in1_vals.begin() + start_index + ni, cur_in1_vals.begin());
  thrust::copy(c->in2_vals.begin() + start_index, c->in2_vals.begin() + start_index + ni, cur_in2_vals.begin());
//std::cout << "made it through first copies" << std::endl;
  *com_ct = *com_ct + 3*nvars;
 
  //r is V's random coin tosses for this iteration.
  //really should choose r <--F_p, but this chooses r <-- [2^32]
  for(int j = 0; j < nvars; j++)
  {
    r[j] = rand();
  }


  *rd_ct = *rd_ct + mi+2*mip1;


  int type_func = c->type[i];
  int in1_func = c->in1[i];
  int in2_func = c->in2[i];
 
  //initialize betar and addormultr values for this iteration of protocol
  //these values are used to track each gate's contribution to beta_z, add_i, or mult_i polynomials


  thrust::counting_iterator<int> first(0);
  thrust::for_each(first, first+ni, 
	  init_betar_functor(thrust::raw_pointer_cast(&( cur_betar[ 0 ] )), mi, thrust::raw_pointer_cast(&( z[ 0 ] ))));

  uint64 betar=0; //represents contribution to beta(z, p) by a given gate
  uint64 V1=0; //represents contribution to V_{i+1}(omega1) by a given gate
  uint64 V2=0; //represents contribution to V_{i+1}(omega2) by a given gate


  //num_effective is used for computing the necessary V(omega1) or V(omega2) values at any round of the protocol
  int num_effective = myPow(2, c->log_level_length[i+1]);

  //initialize vals array, which will store evaluate of Vi+1 at various points
  //intially, the protocol needs to know Vi+1 at all points in boolean hypercube
  //so that's what we initialize to.
  thrust::fill(vals.begin(), vals.begin() + num_effective, 0);
  thrust::fill(poly.begin(), poly.begin() + 3 * (mi + 2*mip1 + 1), 0);

  thrust::copy(c->val.begin() + start_index_ip1, c->val.begin() + start_index_ip1 + c->level_length[i+1], vals.begin());

  uint64 betartimesV1;

  //run through each round in this iteration, one round for each of the mi+2mip1 variables

  for(int j = 0; j < mi; j++)
  {
	 // std::cout << "on layer " << i << " and round " << j << std::endl;
	  thrust::fill(temp1.begin(), temp1.begin() + ni, 0);
	  thrust::fill(temp2.begin(), temp2.begin() + ni, 0);
	  thrust::fill(temp3.begin(), temp3.begin() + ni, 0);

	  thrust::for_each(first, first+ni, 
	  first_j_functor(c->level_length[i], c->level_length[i+1], start_index, start_index_ip1, j, thrust::raw_pointer_cast(&( cur_betar[ 0 ] )),
	  thrust::raw_pointer_cast(&( z[ 0 ] )), thrust::raw_pointer_cast(&( cur_addormultr[ 0 ] )), 
	  thrust::raw_pointer_cast(&( r[ 0 ] )), thrust::raw_pointer_cast(&( temp1[ 0 ] )),
	  thrust::raw_pointer_cast(&( temp2[ 0 ] )), thrust::raw_pointer_cast(&( temp3[ 0 ] )), thrust::raw_pointer_cast(&( cur_in1_vals[ 0 ] )), 
	  thrust::raw_pointer_cast(&( cur_in2_vals[ 0 ] )), mi, type_func, n_squared  ));
	  
	  poly[3*j] = thrust::reduce(temp1.begin(), temp1.begin() + ni, (uint64) 0, modular_sum());
	  poly[3*j+1] = thrust::reduce(temp2.begin(), temp2.begin() + ni, (uint64) 0, modular_sum());
	  poly[3*j + 2] = thrust::reduce(temp3.begin(), temp3.begin() + ni, (uint64) 0, modular_sum());
  }

   
    betar=cur_betar[0];	
	
    for(int j = mi; j < mi+mip1; j++)
    {
	//	std::cout << "on layer " << i << " and round " << j << std::endl;
     if(j > mi)
     {
		 thrust::for_each(first, first+(num_effective >> 1),
              
	    vals_update_functor(thrust::raw_pointer_cast(&(vals[0])), thrust::raw_pointer_cast(&(r[0])), j, thrust::raw_pointer_cast(&(temp_vals[0]))));
		thrust::copy(temp_vals.begin(), temp_vals.begin() + (num_effective >> 1), vals.begin());
		//num_effective tracks the number of points at which we need to evaluate Vi+1
		//for this round of the protocol (it halves every round, as one more variable becomes 'bound')
		num_effective=num_effective>>1;
      }
      

	  thrust::fill(temp1.begin(), temp1.begin() + ni, 0);
	  thrust::fill(temp2.begin(), temp2.begin() + ni, 0);
	  thrust::fill(temp3.begin(), temp3.begin() + ni, 0);


		  thrust::for_each(first, first+ni, 
		  second_j_functor(c->level_length[i], c->level_length[i+1], start_index, start_index_ip1, j, betar,
		  thrust::raw_pointer_cast(&( cur_addormultr[ 0 ] )), 
		  thrust::raw_pointer_cast(&( r[ 0 ] )), thrust::raw_pointer_cast(&( temp1[ 0 ] )),
		  thrust::raw_pointer_cast(&( temp2[ 0 ] )), thrust::raw_pointer_cast(&( temp3[ 0 ] )), thrust::raw_pointer_cast(&( vals[ 0 ] )), mi, mip1,
		  thrust::raw_pointer_cast(&( cur_in2_vals[ 0 ])), type_func, in1_func, d, two_pow_d, n_squared ));

	  poly[3*j] = thrust::reduce(temp1.begin(), temp1.begin() + ni, (uint64) 0, modular_sum());
	  poly[3*j+1] = thrust::reduce(temp2.begin(), temp2.begin() + ni, (uint64) 0, modular_sum());
	  poly[3*j + 2] = thrust::reduce(temp3.begin(), temp3.begin() + ni, (uint64) 0, modular_sum());
	}

  for(int j = mi+mip1; j < nvars; j++)
  {
    //if we're in a round > mi, we need to know evaluations of Vi+1 at some non-boolean locations
    //The following logic computes these values
    if(j != mi+mip1)
    { 
      thrust::for_each(first, first+(num_effective >> 1),
              
	    vals_update_functor(thrust::raw_pointer_cast(&(vals[0])), thrust::raw_pointer_cast(&(r[0])), j, thrust::raw_pointer_cast(&(temp_vals[0]))));
		thrust::copy(temp_vals.begin(), temp_vals.begin() + (num_effective >> 1), vals.begin());
		//num_effective tracks the number of points at which we need to evaluate Vi+1
		//for this round of the protocol (it halves every round, as one more variable becomes 'bound')
		num_effective=num_effective>>1;
    }
    else
    {
        num_effective = myPow(2, c->log_level_length[i+1]);
		thrust::fill(vals.begin(), vals.begin()+num_effective,0);
		thrust::copy(c->val.begin() + start_index_ip1, c->val.begin() + start_index_ip1 + c->level_length[i+1], vals.begin());
    	
		//thrust::device_vector<uint64> scratch(nip1, 0);
		V1=evaluate_V_i(mip1, nip1, vals, r, mi, temp1);
    	betartimesV1=myModMult(betar, V1);
    }

	  thrust::fill(temp1.begin(), temp1.begin() + ni, 0);
	  thrust::fill(temp2.begin(), temp2.begin() + ni, 0);
	  thrust::fill(temp3.begin(), temp3.begin() + ni, 0);

		thrust::for_each(first, first+ni, 
	  third_j_functor(c->level_length[i], c->level_length[i+1], start_index, start_index_ip1, j, betar, betartimesV1,
	  thrust::raw_pointer_cast(&( cur_addormultr[ 0 ] )), 
	  thrust::raw_pointer_cast(&( r[ 0 ] )), thrust::raw_pointer_cast(&( temp1[ 0 ] )),
	  thrust::raw_pointer_cast(&( temp2[ 0 ] )), thrust::raw_pointer_cast(&( temp3[ 0 ] )),
	  thrust::raw_pointer_cast(&( vals[ 0 ] )), mi, mip1, V1, type_func, in1_func, in2_func, d, two_pow_d, n_squared )); 
	  
	  poly[3*j] = thrust::reduce(temp1.begin(), temp1.begin() + ni, (uint64) 0, modular_sum());
	  poly[3*j+1] = thrust::reduce(temp2.begin(), temp2.begin() + ni, (uint64) 0, modular_sum());
	  poly[3*j + 2] = thrust::reduce(temp3.begin(), temp3.begin() + ni, (uint64) 0, modular_sum());
  }//end j loop
 
  //have verifier check that all of P's messages in the sum-check protocol for this level of the circuit are valid
  //t1=clock();
  if( (myMod(poly[0]+poly[1]) != ri) && (myMod(poly[0]+poly[1]) != ri-PRIME) 
       && (myMod(poly[0]+poly[1]) != ri+PRIME))
  {
    std::cout << "poly[0][0]+poly[0][1] != ri. poly[0][0] is: " << poly[0] << " poly[0][1] is: ";
    std::cout << poly[1] << " ri is: " << ri << "i is: " << i << std::endl;
  }
  uint64 check=0;

  for(int j =1; j < nvars; j++)
  {
	  check=host_extrap(thrust::raw_pointer_cast(& (poly[0])) + 3*(j-1), 3, r[j-1]);  
    if( (check != myMod(poly[3*j]+poly[3*j + 1])) && (check != (myMod(poly[3*j]+poly[3*j + 1]) + PRIME))
        && (check != myMod(poly[3*j]+poly[3*j + 1])-PRIME))
    {
      std::cout << "poly[j][0]+poly[j][1] != extrap. poly[j][0] is: " << poly[3*j] << " poly[j][1] is: ";
      std::cout << poly[3*j+1] << " extrap is: " << check << " j is: " << j << " i is: " << i << std::endl; //exit(1); 
    }
  }
  

  //*ct+=clock()-t1;
  //finally check whether the last message extrapolates to f_z(r). In practice, verifier would
  //compute beta(z, r), add_i(r), and mult_i(r) on his own, and P would tell him what
  //V_{i+1}(omega1) and Vi+1(omega2) are. (Below, V1 is the true value of V_{i+1}(omega1) and V2
  //is the true value of Vi+1(omega2))
 
  thrust::copy(c->val.begin() + start_index_ip1, c->val.begin() + start_index_ip1 + nip1, vals.begin());
  V2 = evaluate_V_i(mip1, nip1, vals, r, mi+mip1, temp1);

  uint64 fz=0;
  uint64 plus=myMod(V1+V2);
  uint64 mult=myModMult(V1, V2);

  uint64 test_add;
  uint64 test_mult;
  
  //Have the verifier evaluate the add_i and mult_i polynomials at the necessary location
  test_add = (*add_ifnc)(r, mi, mip1, ni, nip1);
  test_mult = (*mult_ifnc)(r, mi, mip1, ni, nip1);
  fz=myModMult(betar, myMod(myModMult(test_add, plus) + myModMult(test_mult, mult)));
  
 
  if(vstream == 1)
  {  	
	  //cudaThreadSynchronize();
	  //clock_t vstream_time = clock();
  	thrust::for_each(first, first+ni, 
  	
	  vstream_functor((int) c->level_length[i], (int) c->level_length[i+1], betar, n_squared,
	  thrust::raw_pointer_cast(&( cur_addormultr[ 0 ] )), 
	 thrust::raw_pointer_cast(&( temp1[ 0 ] )), plus, mult, type_func)); 
    
     fz =  thrust::reduce(temp1.begin(), temp1.end(), (uint64) 0, modular_sum());
	//cudaThreadSynchronize();
	 //std::cout << "vstream time was: " << (double) (clock() - vstream_time)/CLOCKS_PER_SEC;
  }

  //fz now equals the value f_z(r), assuming P truthfully provided V_{i+1}(omega1) and Vi+1(omega2) are.

  //compute the *claimed* value of fz(r) implied by P's last message, and see if it matches

  check=host_extrap(thrust::raw_pointer_cast(&(poly[0])) + 3*(nvars-1), 3, r[nvars-1]);  
  if( (check != fz) && (check != fz + PRIME) && (check != fz- PRIME))
  {
      std::cout << "fzaz != extrap. poly[nvars-1][0] is: " << poly[3*(nvars-1)];
      std::cout << " poly[nvars-1][1] is: " << poly[3*(nvars-1) + 1] << " extrap is: " << check << " i is: " << i << std::endl; 
      std::cout << "fz is: " << fz << std::endl;
  }
  
  //now reduce claim that V_{i+1}(r1)=V1 and V_{i+1}(r2)=V2 to V_{i+1}(r3)=V3.
  //Let gamma be the line such that gamma(0)=r1, gamma(1)=r2
  //P computes V_{i+1)(gamma(0))... V_{i+1}(gamma(mip1))
  //t1=clock();
  uint64* lpoly = (uint64*) calloc(mip1+1, sizeof(uint64));
  thrust::host_vector<uint64> point1(mip1, 0);
  thrust::device_vector<uint64> point;
  static uint64 vec[2];

  //thrust::device_vector<uint64> scratch3(nip1, 0);

  for(int k = 0; k < mip1+1; k++)
  {
    for(int j = 0; j < mip1; j++)
    {
      vec[0]=r[mi+j];
      vec[1]=r[mi+mip1+j];
      point1[j] = host_extrap(vec, 2, k);
    }
    point = point1;
    lpoly[k]=evaluate_V_i(mip1, nip1, vals, point, 0, temp1); 
  }

  if( (V1 != lpoly[0]) && (V1 != lpoly[0] + PRIME) && (V1 != lpoly[0]-PRIME))
    std::cout << "V1 != lpoly[0]. V1 is: " << V1 << " and lpoly[0] is: " << lpoly[0] << std::endl;
  if( (V2 != lpoly[1]) && (V2 != lpoly[1] + PRIME) && (V2 != lpoly[1]-PRIME))
    std::cout << "V2 != lpoly[1]. V2 is: " << V2 << " and lpoly[1] is: " << lpoly[1] << std::endl;

  uint64 t = rand(); //std::cout << "\n\n\n\nt is: " << t << "\n\n";
  for(int j = 0; j < mip1; j++)
  {
    vec[0]=r[mi+j];
    vec[1]=r[mi+mip1+j];
    h_z[j] = host_extrap(vec, 2, t);
  }
   z=h_z;
   uint64 answer= host_extrap(lpoly, mip1+1, t);

   free(lpoly);
   //std::cout << "z[0] is: " << z[0] << std::endl;
   return answer;
}

uint64 myRand()
{
	return rand() % 1000;
}

uint64 myRand10()
{
	return rand() % 10;
}

struct f2_layer_d_setup
{	
	int* type;
	int* in1;
	int* in2;
	
    f2_layer_d_setup(int* p_type, int* p_in1, int* p_in2)
	{	
		type=p_type;
		in1 = p_in1;
		in2 = p_in2;
	}

    __device__ void operator() ( int j)
    {
       type[j] = 1;
	   in1[ j] = j;
	   in2[ j] = j;
	}
};

struct f2_layer_notd_setup
{	
	int* type;
	int* in1;
	int* in2;
	
    f2_layer_notd_setup(int* p_type, int* p_in1, int* p_in2)
	{	
		type=p_type;
		in1 = p_in1;
		in2 = p_in2;
	}

    __device__ void operator() ( int j)
    {
       type[j] = 0;
	   in1[j] = 2*j;
	   in2[j] = 2*j+1;
	}
};

//constructs the circuit computing F2, initializes the stream items too
//d is log of universe size
circ* construct_F2_circ(int d, int* num_gates)
{            

  uint64 n = myPow(2, d);
  circ* c = new circ; 
  c->level_length = (uint64*) calloc(d+2, sizeof(uint64));
  c->log_level_length = (int*) calloc(d+2, sizeof(int));
  c->in1 = thrust::host_vector<int>(d+2, 1);
  c->in2 = thrust::host_vector<int>(d+2, 1);
  c->type = thrust::host_vector<int>(d+2, 0);

  c->in1[d]=0;
  c->in2[d]=0;
  c->type[d]=1;

  c->num_levels=d+2;

  c->val = thrust::host_vector<uint64>(4*n);
  c->in1_vals = thrust::host_vector<uint64>(4*n);
  c->in2_vals = thrust::host_vector<uint64>(4*n);

  uint64 size = n;

  c->level_length[d+1]=n;
  c->log_level_length[d+1]=d;

  uint64 ans = 0;

thrust::counting_iterator<int> count(0);

  thrust::generate(c->val.begin(), c->val.end(), myRand);
  
  clock_t tim = clock();
  for(uint64 j = 0; j < n; j++)
  {
     //c->type[j]=-1;
     ans+=c->val[j] * c->val[j];
  }
  std::cout << "The correct second frequency moment is: " << ans << std::endl;
  std::cout << "Time to compute second frequency moment unverifiably is: " << (double)(clock()-tim)/(double)CLOCKS_PER_SEC;
  
  int gate = n;
  for(int i = d; i >= 0; i--)
  { 
    c->level_length[i]=size;
    c->log_level_length[i]=i;

    if(i == d)
    {
		gate += size;
    }
    else
    {
		gate +=size;
    }
    size=size >> 1;
  }
  *num_gates = gate;
  return c;
}

//constuct circuit for computing matrix multiplication
circ* construct_mat_circ(int d, int* num_gates)
{ 
  clock_t t = clock();
  clock_t start = clock();
  int n = myPow(2, d);
  int n_squared = n*n;
  int n_cubed = n*n*n;


  circ* c = new circ; 
  c->level_length = (uint64*) calloc(64+3*d, sizeof(uint64));
  c->log_level_length = (int*) calloc(64+3*d, sizeof(int));
  c->num_levels=64+3*d;

  c->in1 = thrust::host_vector<int>(64+3*d);
  c->in2 = thrust::host_vector<int>(64+3*d);
  c->type = thrust::host_vector<int>(64+3*d);
 
  c->val = thrust::host_vector<uint64>(125*n_squared+n+1+3*n_squared+3*d*n_squared+3*n_cubed + 3*n_squared + 3000, 0);

    std::cout << "time to allocate memory in construction function is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;
  //std::cout << "125*n is: " << 125*n << std::endl;

  c->level_length[63+3*d] = 3*n_squared+1;
  c->log_level_length[63+3*d] = ceil(log((double)c->level_length[63+3*d])/log((double) 2));


  t = clock();
  //set up input level
  c->val[3*n_squared]=0;//need a constant 0 gate. So universe really has size 2^d-1, not 2^d
  thrust::generate(c->val.begin(), c->val.begin() + 2*n_squared, myRand10);
   std::cout << "time to randomly generate data is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;
  /*for(int i = 0; i < n; i++)
  {
	  for(int j = 0; j < n; j++)
	  {
		  std::cout << "A_" << i << j << " is: " << c->val[i*n+j] << std::endl;
	  }
  }

  for(int i = 0; i < n; i++)
  {
	  for(int j = 0; j < n; j++)
	  {
		  std::cout << "B_" << i << j << " is: " << c->val[n_squared + i*n+j] << std::endl;
	  }
  }*/

  t=clock();
  for(int i = 0; i < n; i++)
  {
	  for(int j = 0; j < n; j++)
	  {
		  for(int k = 0; k < n; k++)
		  {
			  c->val[2*n_squared+i*n+j] += c->val[i*n+k] * c->val[n_squared+k*n+j];
		  }
		   //std::cout << "Setting C_" << i << j << "to: -" << c->val[2*n_squared + i*n + j];
		   c->val[2*n_squared+i*n+j] = myModMult(c->val[2*n_squared+i*n+j], PRIME-1);
	  }
  }
  std::cout << "time to compute matvec unverifiably is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;

  c->level_length[62+3*d] = n_cubed + n_squared + 1;
  c->log_level_length[62+3*d] = 3*d + 1;
  c->in1[62+3*d] = c->in2[62+3*d] = c->type[62+3*d] = 4;

  int step=1;
  for(int j = 61+3*d; j >=62+2*d; j--)
  {
	  c->level_length[j] = n_cubed/myPow(2, step) + n_squared + 1;
	  c->log_level_length[j] = ceil(log((double) c->level_length[j])/log((double)2));
	  c->in1[j]=c->in2[j]=5;
	  c->type[j]=0;
	  step++;
  }

  c->level_length[61+2*d]=n_squared+1;

  c->in1[61+2*d]=6;
  c->in2[61+2*d]=6;
  c->type[61+2*d] = 0;

  c->level_length[60+2*d]=(n_squared + 1);


  c->level_length[59+2*d]=2*(n_squared + 1);


  for(int i = 2*d+1; i < 59+2*d; i++)
  {
    c->level_length[i]=2*(n_squared + 1);

  }

  //std::cout << "real number of distinct items is: " << ans;
  //input level now set up

  //set up level just below input
  c->type[60+2*d] = 1;
  c->in1[60+2*d] = c->in2[60+2*d] = 0;

  //set up level just below that
  c->type[59+2*d] = c->in1[59+2*d] = c->in2[59+2*d] = 3;

  //implement gates to compute x^(p-1) for each input x
  //level_length and log_level length already set up for these levels
  for(uint64 i=2*d+1; i < 59+2*d; i++)
  {
	  c->type[i] = 2;
	  c->in1[i] = 2;
	  c->in2[i] = 2;
  }
 
    //set up level 2*d
    c->level_length[2*d]=n*n;

	c->in1[2*d] = c->in2[2*d] = c->type[2*d] = 1;

  
  //set up levels 0 to d-1
  uint64 size = n_squared/2;
  
  for(long long i = 2*d-1; i >= 0; i--)
  {
    c->level_length[i]=size;
    c->log_level_length[i]=i;
    
	c->in1[i] = c->in2[i] = 1;
	c->type[i] = 0;
	size=size >> 1;
  }
  for(int i = 0; i < c->num_levels; i++)
  {
	  *num_gates = *num_gates + c->level_length[i];
	  c->log_level_length[i]=ceil((double) log((double)c->level_length[i])/log((double)2));
  }
  c->in1_vals = thrust::host_vector<uint64>(*num_gates);
  c->in2_vals = thrust::host_vector<uint64>(*num_gates);

   std::cout << "total time in construction is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC << std::endl;
  return c;
}


//constuct circuit computing F0 using only + and * gates

//constuct circuit computing F0 using only + and * gates
circ* construct_F0_circ(int d, int* num_gates)
{ 
  clock_t t = clock();
  clock_t start = clock();
  uint64 n = myPow(2, d);
  circ* c = new circ; 
  c->level_length = (uint64*) calloc(62+d, sizeof(uint64));
  c->log_level_length = (int*) calloc(62+d, sizeof(int));
  c->num_levels=62+d;

  c->in1 = thrust::host_vector<int>(62+d);
  c->in2 = thrust::host_vector<int>(62+d);
  c->type = thrust::host_vector<int>(62+d);


  c->val = thrust::host_vector<uint64>(125*n);
  c->in1_vals = thrust::host_vector<uint64>(125*n);
  c->in2_vals = thrust::host_vector<uint64>(125*n);

    std::cout << "time to allocate memory in construction function is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;
  //std::cout << "125*n is: " << 125*n << std::endl;

  //set up all the level_lengths and log_level_lengths for all levels d+1 and up
  c->level_length[61+d]=n;
  c->log_level_length[61+d]=d;
  c->level_length[60+d]=n;
  c->log_level_length[60+d]=d;

  c->level_length[59+d]=2*n;
  c->log_level_length[59+d]=d+1;

  for(int i = d+1; i < 59+d; i++)
  {
    c->level_length[i]=2*n;
    c->log_level_length[i]=d+1;
  }

  t = clock();
  //set up input level
   c->val[n-1]=0;//need a constant 0 gate. So universe really has size 2^d-1, not 2^d
  thrust::generate(c->val.begin(), c->val.begin() + n-1, myRand10);
  std::cout << "time to randomly generate data is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;
  t=clock();
  int ans=0;
  for(uint64 j = 0; j < n-1; j++)
  {
     if(c->val[j] > 0)
       ans++;
  }
   std::cout << "time to compute F0 unverifiably is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;
  std::cout << "real number of distinct items is: " << ans;
  //input level now set up

  t=clock(); 
  //set up level just below input
  c->type[60+d] = 1;
  c->in1[60+d] = c->in2[60+d] = 0;

  //set up level just below that
  c->type[59+d] = c->in1[59+d] = c->in2[59+d] = 3;

  //implement gates to compute x^(p-1) for each input x
  //level_length and log_level length already set up for these levels
  for(uint64 i=d+1; i < 59+d; i++)
  {
	  c->type[i] = 2;
	  c->in1[i] = 2;
	  c->in2[i] = 2;
  }
 
    //set up level d
    c->level_length[d]=n;
    c->log_level_length[d]=d;

	c->in1[d] = c->in2[d] = c->type[d] = 1;

  
  //set up levels 0 to d-1
  uint64 size = n/2;
  
  for(long long i =d-1; i >= 0; i--)
  {
    c->level_length[i]=size;
    c->log_level_length[i]=i;
    
	c->in1[i] = c->in2[i] = 1;
	c->type[i] = 0;
	size=size >> 1;
  }
  for(int i = 0; i < c->num_levels; i++)
	  *num_gates = *num_gates + c->level_length[i];

   std::cout << "total time in construction is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC << std::endl;
  return c;
} // end construct_F0_circ

//constuct circuit computing Pattern Matching
//Note: the output of this circuit will be # of places pattern does *not* occur at, minus 2
//Since I chose the pattern to match the last PAT_LEN symbols of the input, and wired the circuit to
//ensure that the n'th location evalutes to 0 too
circ* construct_pm_circ(int d, int* num_gates)
{ 
  clock_t start = clock();
  int n = myPow(2, d);
  int t = myPow(2, LOG_PAT_LEN); 
  
  circ* c = new circ; 
  c->level_length = (uint64*) calloc(64+d+LOG_PAT_LEN, sizeof(uint64));
  c->log_level_length = (int*) calloc(64+d+LOG_PAT_LEN, sizeof(int));
  c->num_levels=64+d+LOG_PAT_LEN;

  c->in1 = thrust::host_vector<int>(64+d+LOG_PAT_LEN);
  c->in2 = thrust::host_vector<int>(64+d+LOG_PAT_LEN);
  c->type = thrust::host_vector<int>(64+d+LOG_PAT_LEN);

  c->val = thrust::host_vector<uint64>(125*n+4*n*t);
  c->in1_vals = thrust::host_vector<uint64>(125*n+4*n*t);
  c->in2_vals = thrust::host_vector<uint64>(125*n+4*n*t);
  
  c->level_length[63+d+LOG_PAT_LEN]=n+t;
  
  for(int i = 0; i <  n; i++)
  	c->val[i] = rand() % 10;
  for(int i = n; i < n+PAT_LEN; i++)
  {
  	c->val[i] = myModMult(PRIME-1, c->val[i - PAT_LEN]);
  }
  //done generating input. pattern should appear at end of input

    std::cout << "time to allocate memory in construction function is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC  << std::endl;
  //std::cout << "125*n is: " << 125*n << std::endl;

  //set up addition layer connecting (i, j) to t_j
  c->level_length[62+d+LOG_PAT_LEN]=n*t;
  c->type[62+d+LOG_PAT_LEN] = 0;
  c->in1[62+d+LOG_PAT_LEN] =  c->in2[62+d+LOG_PAT_LEN] = 7;
  
  //set up squaring layer
  c->level_length[61+d+LOG_PAT_LEN]=n*t;
   c->type[61+d+LOG_PAT_LEN]=1;
    c->in1[61+d+LOG_PAT_LEN]= c->in2[61+d+LOG_PAT_LEN]=0;
  
  //create binary tree of sum gates to finish compute I_i
  //bottom of tree used to have length n+1, but changed to n, 
  //since last gate will be 0 by design of sticking pattern at end of input
  int blowup = 1;
  for(int i = 0; i < LOG_PAT_LEN; i++)
  {
  	c->level_length[61+d+i] = blowup*n;
	c->type[61+d+i] = 0;
	c->in1[61+d+i] =c->in2[61+d+i] = 1;
	blowup = blowup * 2;
  }


//used to be n+1, changed to n. 60+d is the squaring level for F0 computation
  c->level_length[60+d]=n;
    //set up level just below input
  c->type[60+d] = 1;
  c->in1[60+d] = c->in2[60+d] = 0;

//used to be 2*(n+1), changed to 2*n
  c->level_length[59+d]=2*n;

//used to be 2*(n+1), changed to 2*n
  for(int i = d+1; i < 59+d; i++)
  {
    c->level_length[i]=2*n;
  }


  //set up level just below that
  c->type[59+d] = c->in1[59+d] = c->in2[59+d] = 3;

  //implement gates to compute x^(p-1) for each input x
  //level_length and log_level length already set up for these levels
  for(uint64 i=d+1; i < 59+d; i++)
  {
	  c->type[i] = 2;
	  c->in1[i] = 2;
	  c->in2[i] = 2;
  }
 
    //set up level d
    c->level_length[d]=n;


	c->in1[d] = c->in2[d] = c->type[d] = 1;

  
  //set up levels 0 to d-1
  uint64 size = n/2;
  
  for(long long i = d-1; i >= 0; i--)
  {
    c->level_length[i]=size;
    c->log_level_length[i]=i;
    
	c->in1[i] = c->in2[i] = 1;
	c->type[i] = 0;
	size=size >> 1;
  }
  for(int i = 0; i < c->num_levels; i++)
  {
	  *num_gates = *num_gates + c->level_length[i];
	  c->log_level_length[i]=ceil((double) log((double)c->level_length[i])/log((double)2));
  }
  c->in1_vals = thrust::host_vector<uint64>(*num_gates);
  c->in2_vals = thrust::host_vector<uint64>(*num_gates);

   std::cout << "total time in construction is: " << (double) ((double)clock()-(double)t)/(double) CLOCKS_PER_SEC << std::endl;
  return c;
} // end construct_PM_circ



struct f2_mult_d_func
{
	enum TupleLayout
	{
		A,
		B,
		C,
		RESULT
	};
	
    f2_mult_d_func() {}

	template <typename TTuple >
    __device__ void operator() ( TTuple tuple )
    {
        uint64 a = thrust::get< A >( tuple );                                              
        uint64 b = thrust::get< B >( tuple );                                              
        uint64 c = thrust::get< C >( tuple );                                              

		uint64 result = myModMult( myModMult( a, b ), c );
		result = myMod(result + myModMult(myModMult(1+PRIME-a, 1+PRIME-b), 1+PRIME-c)); 
		
        thrust::get< RESULT >( tuple ) = result;
    }
};

//evaluates the polynomial mult_d for the F2 circuit at point r
uint64 F2_mult_d(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
{
  thrust::device_vector<uint64> scratch(mi);

  thrust::for_each(

        thrust::make_zip_iterator( thrust::make_tuple( r.begin(), r.begin()+mi, r.begin() + mi + mip1, scratch.begin() )),

        thrust::make_zip_iterator( thrust::make_tuple(r.begin()+mi, r.begin() + (2* mi), r.begin() + (2*mi + mip1), scratch.begin() + mi )),

        f2_mult_d_func() );
	   return thrust::reduce(scratch.begin(), scratch.end(), (uint64) 1, modular_mult());
}

uint64 F2_mult_d_old(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
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
uint64 zero(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
{
  return 0;
}

//evaluates the polynomial add_i for any layer of the F2 circuit other than the d'th layer
uint64 F2_add_notd_old(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
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

//evaluates the polynomial add_i for any layer of the F2 circuit other than the d'th layer
uint64 F2_add_notd(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
{
  thrust::device_vector<uint64> scratch(mi);

  thrust::for_each(

        thrust::make_zip_iterator( thrust::make_tuple( r.begin(), r.begin()+mi+1, r.begin() + mi + mip1+1, scratch.begin() )),

        thrust::make_zip_iterator( thrust::make_tuple(r.begin()+mi, r.begin() + (2* mi)+1, r.begin() + (2*mi + mip1+1), scratch.begin() + mi )),

        f2_mult_d_func() );
 uint64 ans = thrust::reduce(scratch.begin(), scratch.end(), (uint64) 1, modular_mult());
   ans = myModMult(ans, 1+PRIME-r[mi]);
  return myModMult(ans, r[mi+mip1]); 
}

uint64 F0add_dp1to58pd(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
{
  //uint64 temp1 = chi(ni-1, r, mi, 0);
  //uint64 temp2 = chi(nip1-1, r, mip1, mi);
  //uint64 temp3 = chi(nip1-1, r, mip1, mi+mip1);
  
  return 0;
  //return myModMult(temp1, myModMult(temp2, temp3));
}

//evaluates mult_i for the F0 circuit for any i between d+1 and 58+d
uint64 F0mult_dp1to58pd(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
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
  
  return myMod(ans1 + ans2);
}

//evaluates mult_i polynomial for layer 59+d of the F0 circuit
uint64 F0mult_59pd(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
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
uint64 F0add_59pd(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
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
   ans1 = myModMult(ans1, chi(nip1-1, r, mip1, mi));
   return ans1;
}

//evaluates the mult_i polynomial for the d'th layer of the F0 circuit
uint64 F0mult_d(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
{
  //all gates p have in1=2p, in2=2p+1
  uint64 ans = F2_add_notd(r, mi, mip1, ni, nip1);
  
  return ans;
}

//evaluates the mult_i polynomial for layer 60+d of the F0 circuit
uint64 F0mult_60pd(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
{
  //all gates p but n-2 have in1=in2=p.
  uint64 ans = F2_mult_d(r, mi, mip1, ni, nip1);
  
  return ans;
}

uint64 mat_add_63_p3d(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
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

uint64 mat_mult_63_p3d(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
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


uint64 mat_add_below_63_p3d(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
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



uint64 mat_add_61_p2dp1(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
{
	int n_squared = ni-myPow(2, mi-2)-1;
	int d = floor(Log2((double) n_squared)/2 + 0.5);
		
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


uint64 mat_add_61_p2d(thrust::device_vector<uint64>& r, int mi, int mip1, int ni, int nip1)
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
  
int main(int argc, char** argv)
{
	srand(0);
  if((argc != 3))
  {
    std::cout << "argc!= 3. Command line arg should be log(universe size) followed by protocol identifier.";
    std::cout << "0 for F2, 1 for F0 basic.\n";
    std::cout << "This implementation supports V evaluating the polynomials add_i and mult_i in logarithmic time.\n";
    exit(1);
  }
  int d = atoi(argv[1]);
  int n = myPow(2, d);
  int n_squared = n*n;
  int two_pow_d = myPow(2, d);

  //protocol specifies whether to run F2 or F0
  int protocol = atoi(argv[2]);

  if((protocol < 0) || (protocol > 3))
  {
     std::cout << "protocol parameter should be 0, 1, 2, or 3. 0 is for F2, 1 is for F0, 2 for matvec, 3 for pattern matching\n";
     exit(1);
  }

  
  clock_t t = clock();

   thrust::device_vector<uint64> temp1;
   thrust::device_vector<uint64> temp2;
	thrust::device_vector<uint64> temp3;
	thrust::device_vector<uint64> cur_betar;
	thrust::device_vector<uint64> cur_addormultr;
	thrust::device_vector<uint64> cur_in1_vals;
	thrust::device_vector<uint64> cur_in2_vals;
	thrust::device_vector<uint64> temp_vals;
	thrust::device_vector<uint64> vals;
	 thrust::device_vector<uint64> r;
	  thrust::host_vector<uint64> poly;


		 if(protocol == 0)
		 {
			 temp1 = thrust::device_vector<uint64> (2*n);
			 temp2 = thrust::device_vector<uint64>(n);
			 temp3 = thrust::device_vector<uint64> (n);

			 cur_betar= thrust::device_vector<uint64>(n);
			 cur_addormultr=thrust::device_vector<uint64>(n, 1);
			 cur_in1_vals=thrust::device_vector<uint64>(n);
			 cur_in2_vals =thrust::device_vector<uint64>(n);
			  vals =  thrust::device_vector<uint64>(2*n, 0);
			 temp_vals = thrust::device_vector<uint64> (2*n);
			 r = thrust::device_vector<uint64> (3*(d+4));
			 poly = thrust::host_vector<uint64> (18*3*(d+4));
		 }

		 if(protocol ==1)
		 {
			 temp1 = thrust::device_vector<uint64> (4*n);
			 temp2 = thrust::device_vector<uint64>(2*n);
			 temp3 = thrust::device_vector<uint64> (2*n);

			 cur_betar= thrust::device_vector<uint64>(2*n);
			 cur_addormultr=thrust::device_vector<uint64>(2*n, 1);
			 cur_in1_vals=thrust::device_vector<uint64>(2*n);
			 cur_in2_vals =thrust::device_vector<uint64>(2*n);
			 vals =  thrust::device_vector<uint64>(4*n, 0);
			 temp_vals = thrust::device_vector<uint64> (4*n);
			 r = thrust::device_vector<uint64> (3*(d+4));
			 poly = thrust::host_vector<uint64> (18*3*(d+4));
		 }
		if(protocol ==2)
		 {
			 int max_level_length = n*n*n+n*n+1;
			 temp1 = thrust::device_vector<uint64> (2*max_level_length);
			 temp2 = thrust::device_vector<uint64>(max_level_length);
			 temp3 = thrust::device_vector<uint64> (max_level_length);

			 cur_betar= thrust::device_vector<uint64>(max_level_length);
			 cur_addormultr=thrust::device_vector<uint64>(max_level_length, 1);
			 cur_in1_vals=thrust::device_vector<uint64>(max_level_length);
			 cur_in2_vals =thrust::device_vector<uint64>(max_level_length);
			 vals =  thrust::device_vector<uint64>(2*max_level_length, 0);
			 temp_vals = thrust::device_vector<uint64> (2*max_level_length);

			 r = thrust::device_vector<uint64> (3*(4*d+4));
			 poly = thrust::host_vector<uint64> (18*3*4*(d+4));
		 }
		 if(protocol == 3)
		 {
			 temp1 = thrust::device_vector<uint64> (4*n*PAT_LEN);
			 temp2 = thrust::device_vector<uint64>(2*n*PAT_LEN);
			 temp3 = thrust::device_vector<uint64> (2*n*PAT_LEN);

			 cur_betar= thrust::device_vector<uint64>(2*n*PAT_LEN);
			 cur_addormultr=thrust::device_vector<uint64>(2*n*PAT_LEN, 1);
			 cur_in1_vals=thrust::device_vector<uint64>(2*n*PAT_LEN);
			 cur_in2_vals =thrust::device_vector<uint64>(2*n*PAT_LEN);
			 vals =  thrust::device_vector<uint64>(4*n*PAT_LEN, 0);
			 temp_vals = thrust::device_vector<uint64> (4*n*PAT_LEN);
			 r = thrust::device_vector<uint64> (3*(d+4+PAT_LEN));
			 poly = thrust::host_vector<uint64> (18*3*(d+4+PAT_LEN));
		 }



	//allocate an array to store the values of V_i+1(omega1) and V_i+1(omega2) the P needs to compute her messages
   std::cout << "time to allocate memory is: " << ((double) (clock() - t))/CLOCKS_PER_SEC << std::endl;
  /********************************************
  *Begin code to construct circuit of interest*
  ********************************************/
  circ* c;
  int num_gates = 0;
  t=clock();

  if(protocol == 0)
    c = construct_F2_circ(d, &num_gates);
  else if(protocol == 1)
    c = construct_F0_circ(d, &num_gates);
  else if(protocol == 2)
    c = construct_mat_circ(d, &num_gates);
  else if(protocol == 3)
    c = construct_pm_circ(d, &num_gates);

  int sum = 0;
  for(int i = 0; i < c->num_levels; i++)
	  sum+= c->level_length[i];

  std::cout << "\ntime to construct circuit is: " << ((double) (clock()-t))/CLOCKS_PER_SEC << " num_gates is: " << num_gates << std::endl;

  /******************************************
  *End code to construct circuit of interest*
  *******************************************/

  /************************************************************
  *Begin generic code to evaluate circuit in verifiable manner*
  *************************************************************/
  cudaThreadSynchronize();
  t = clock();

  //evaluate the circuit
  evaluate_circuit_seq(c, cur_in1_vals, cur_in2_vals, temp1, d, two_pow_d, n*n);
    std::cout << "time to evaluate circuit is: " << ((double) (clock() - t))/CLOCKS_PER_SEC << std::endl;

	
 std::cout << " The circuit evaluated to: " << c->val[num_gates-1] << std::endl;
 cudaThreadSynchronize();

  thrust::device_vector<uint64> zi(c->num_levels);
  thrust::host_vector<uint64> h_zi(c->num_levels);
  uint64 ri=c->val[num_gates-1];
  
  //run through entire Muggles protocol with prover
  clock_t pt=0;
  int com_ct=0;
  int rd_ct=0;


  int start_index = num_gates-1;
  int start_index_ip1 = start_index - c->level_length[1];

//type_func: 0 = zero, 1 = one.
//in1_func: 0 = identity, 1=2*j
//in2_func: 0 = identity, 1 = 2*j+1

  t=clock();
  clock_t test= clock();
  if(protocol == 0)
  { //check each level in turn using protocol due to GKR. This is for F2 circuit
    /*for(int i = 0; i < c->num_levels-1; i++)
    {
      if(i < c->num_levels-2)
        ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, F2_add_notd, zero, 
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared);
      else 
        ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, F2_mult_d, 
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared);
	  if(i < c->num_levels-2)
	  {
		start_index = start_index_ip1;
		start_index_ip1 = start_index_ip1 - c->level_length[i+2];
	  }
	}*/
	 start_index = num_gates-(2*n-1);
	start_index_ip1 = start_index - c->level_length[d+1];
	for(int i = d-1; i < c->num_levels-1; i++)
    {
	  if(i==d-1)
	  {
		  start_index = num_gates-(2*n-1);
		  start_index_ip1 = start_index - c->level_length[d+1];
		  ri = check_first_level(c,  r,  zi, d,  vals,   temp_vals, temp1, temp2, poly, start_index);
	  }
	   else 
        ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, F2_mult_d, 
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	}
  }
   else if(protocol == 1)
  { //check each level in turn using protocol due to GKR. This is for F0 circuit
     start_index = num_gates-(2*n-1);
		  start_index_ip1 = start_index - c->level_length[d+1];
	for(int i = d-1; i < c->num_levels-1; i++)
    {
	  if(i==d-1)
	  {
		  start_index = num_gates-(2*n-1);
		  start_index_ip1 = start_index - c->level_length[d+1];
		  ri = check_first_level(c,  r,  zi, d,  vals,   temp_vals, temp1, temp2, poly, start_index);
	  }
	 
     /* else if(i < d)
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, F2_add_notd, zero, 
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr);
	  }*/
	  else if(i == d)
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, F0mult_d, 
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if( (i > d) && (i < 59+d))
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, F0add_dp1to58pd, F0mult_dp1to58pd,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if( (i==59+d))
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals,F0add_59pd, F0mult_59pd,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if(i==60+d)
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, F0mult_60pd,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }

	  if( (i >= d) && (i < c->num_levels-2))
	  {
		start_index = start_index_ip1;
		start_index_ip1 = start_index_ip1 - c->level_length[i+2];
	  }
    }
  }
  else if(protocol == 2)
  { //check each level in turn using protocol due to GKR. This is for matvec circuit
	  start_index = num_gates-(2*n_squared-1);
	  start_index_ip1 = start_index - c->level_length[2*d+1];
    for(int i = 2*d-1; i < c->num_levels-1; i++)
    {
		test=clock();
		//cudaThreadSynchronize();
	  if(i==2*d-1)
	  {
		  start_index = num_gates-(2*n_squared-1);
		  start_index_ip1 = start_index - c->level_length[2*d+1];
		  ri = check_first_level(c,  r,  zi, 2*d,  vals,   temp_vals, temp1, temp2, poly, start_index);
		  cudaThreadSynchronize();
		  //std::cout << " time spent on first check is: " << (double) (clock() - test)/CLOCKS_PER_SEC << std::endl;
	  }
	  else if(i == 2*d)
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, F0mult_d, 
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
		cudaThreadSynchronize();
		//std::cout << " time spent on secnod check is: " << (double) (clock() - test)/CLOCKS_PER_SEC << std::endl;
	  }
	  else if( (i > 2*d) && (i < 59+2*d))
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, F0add_dp1to58pd, F0mult_dp1to58pd,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if( (i==59+2*d))
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals,F0add_59pd, F0mult_59pd,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if(i==60+2*d)
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, F0mult_60pd,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if(i==61+2*d)
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, mat_add_61_p2d, zero,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if(i<=61+3*d)
	  {
	    if(i==61+2*d+1)
		{
			ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, mat_add_61_p2dp1, zero,
			temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
		}
		else
		{
			ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, mat_add_below_63_p3d, zero,
			temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
		}
	  }
	  else
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, mat_add_63_p3d, mat_mult_63_p3d,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
		cudaThreadSynchronize();
		//std::cout << " time spent on last check is: " << (double) (clock() - test)/CLOCKS_PER_SEC << std::endl;
	  }

	  if( (i >= 2*d) && (i < c->num_levels-2))
	  {
		start_index = start_index_ip1;
		start_index_ip1 = start_index_ip1 - c->level_length[i+2];
	  }
    }
  }
   else if(protocol == 3)
  { //check each level in turn using protocol due to GKR. This is for pm circuit
     start_index = num_gates-(2*n-1);
	 start_index_ip1 = start_index - c->level_length[d+1];
	for(int i = d-1; i < c->num_levels-1; i++)
    {

	  if(i==d-1)
	  {
		  start_index = num_gates-(2*n-1);
		  start_index_ip1 = start_index - c->level_length[d+1];
		  ri = check_first_level(c,  r,  zi, d,  vals,   temp_vals, temp1, temp2, poly, start_index);
	  }
	  else if(i == d)
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, F0mult_d, 
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if( (i > d) && (i < 59+d))
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, F0add_dp1to58pd, F0mult_dp1to58pd,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if( (i==59+d))
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals,F0add_59pd, F0mult_59pd,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if(i==60+d) //squaring level within F0 circuit
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, F0mult_60pd,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if(i<=60+d+LOG_PAT_LEN) //binary tree of sum gates
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, F2_add_notd, zero,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if(i==61+d+LOG_PAT_LEN) //squaring level
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, F2_mult_d,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 0);
	  }
	  else if(i==62+d+LOG_PAT_LEN)
	  {
		ri=check_level(c, i, start_index, start_index_ip1, zi, h_zi, ri, &com_ct, &rd_ct, r, poly, vals, temp_vals, zero, zero,
		temp1, temp2, temp3, cur_in1_vals, cur_in2_vals, cur_betar, cur_addormultr, d, two_pow_d, n_squared, 1);
	  }

	  if( (i >= d) && (i < c->num_levels-2))
	  {
		start_index = start_index_ip1;
		start_index_ip1 = start_index_ip1 - c->level_length[i+2];
	  }
    }
  }
  
  
  cudaThreadSynchronize();
  pt=clock()-t;

  //in practice, this stream observation stage would be occur before the
  //interaction with the prover (with the random location zi at which the LDE of the input is evaluated
  //determined in advance), but in terms of timing things, it doesn't matter so we're doing it after
  //the conversation rather than before.
   t = clock();
  thrust::copy(c->val.begin(), c->val.begin() + c->level_length[c->num_levels-1], temp2.begin());
  cudaThreadSynchronize();
  //std::cout<< "copy time for V was: " << (double)(clock() - t)/CLOCKS_PER_SEC << std::endl;
   //t = clock();
  uint64 fr=evaluate_V_i(c->log_level_length[c->num_levels-1], c->level_length[c->num_levels-1], 
                                     temp2, zi, 0, temp1);

  cudaThreadSynchronize();
  
  double vt_secs = (double)(clock() - t)/CLOCKS_PER_SEC; 


  double pt_secs =((double) pt)/CLOCKS_PER_SEC;

  if( (fr != ri) && (fr!=ri+PRIME) && (fr != ri-PRIME))
  {
    std::cout << "Check value derived from stream (fr) is not equal to its claimed value by P (ri).\n";
    std::cout << "fr is: " << fr << " and ri is: " << ri << std::endl;
  }
 
  std::cout << "Done!" << std::endl;

  std::cout << "N\tProveT\tVerT\tProofS\tRounds\n";
  std::cout << myPow(2,d) << "\t" << pt_secs << "\t" << vt_secs << "\t" << com_ct << "\t" << rd_ct << std::endl;

  //clean up memory before exit
  destroy_circ(c);

  return 1;
}
