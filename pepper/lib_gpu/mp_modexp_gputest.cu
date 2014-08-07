#include <cassert>
#include <iostream>

#include <sys/time.h>

#include <cuda_runtime.h>

#include "mp_modexp.h"

/**
 * Returns the limb index of an particular thread. This is the index of the
 * limb this thread is responsible for in the mp number.
 * Assumes that
 *   - num_limbs is a power of 2 that is less than the warp size (32)
 *   - the grid is one dimentional.
 *   - blockDim.x is a multiple of num_limbs.
 */
#define LIMB_IDX(S) (threadIdx.x & ((S) - 1))

/**
 * Returns the element index of an particular thread. This is the index of the
 * element that this thread is responsible for in a vector.
 * Assumes that
 *   - num_limbs is a power of 2 that is less than the warp size (32)
 *   - the grid is one dimentional.
 *   - blockDim.x is a multiple of num_limbs.
 */
#define BLOCK_ELEM_IDX(S) ((threadIdx.y * blockDim.x + threadIdx.x) >> (__ffs((S)) - 1))
#define ELEM_IDX(S) (((blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) >> (__ffs((S)) - 1))


static __device__ WORD mp_umul_hi(WORD a, WORD b)
{
#if MP_USE_64BIT
	return __umul64hi(a, b);
#else
	return __umulhi(a, b);
#endif
}

static __device__ WORD mp_umul_lo(WORD a, WORD b)
{
	return a * b;
}

template<int S>
static __device__ int vote_any(int predicate)
{
	return __any(predicate);
}

#if !MP_USE_64BIT
template<>
static __device__ int vote_any<64>(int predicate)
{
	volatile __shared__ int tmp;

	tmp = 0;
	__syncthreads();
	if (predicate)
		tmp = 1;
	__syncthreads();

	return tmp;
}
#endif

/* assumes ar and br are 'montgomeritized' */
template<int S>
static __device__ void mp_montmul_dev2(WORD *ret, const WORD *ar, const WORD *br,
			       const WORD *n, const WORD *np, const int block_elem_idx,
                               const int limb_idx)
{
#if __CUDA_ARCH__ >= 200 && MP_USE_64BIT
	__shared__ WORD _t[MAX_ELEMS_PER_BLOCK(S) * S * 3];
	__shared__ uint32_t _c[MAX_ELEMS_PER_BLOCK(S) * S * 3];

	volatile WORD *t = _t + (S * 3 * block_elem_idx);
	volatile uint32_t *c = _c + (S * 3 * block_elem_idx);
#else
	__shared__ WORD _t[MAX_ELEMS_PER_BLOCK(S) * S * 2];
	__shared__ uint32_t _c[MAX_ELEMS_PER_BLOCK(S) * S * 2];

	volatile WORD *t = _t + (S * 2 * block_elem_idx);
	volatile uint32_t *c = _c + (S * 2 * block_elem_idx);
#endif

	c[limb_idx] = 0;
	c[limb_idx + S] = 0;
	t[limb_idx] = 0;
	t[limb_idx + S] = 0;
	sync_if_needed();

#if MP_USE_64BIT
#pragma unroll 4
#else
#pragma unroll 8
#endif
	/* step 1: calculate t + mn */
	for (int i = 0; i < S; i++) {
#if 0
		/* readability version */

		WORD hi = mp_umul_hi(ar[i], br[limb_idx]);
		WORD lo = mp_umul_lo(ar[i], br[limb_idx]);

		ADD_CARRY(c[i + limb_idx + 1], t[i + limb_idx + 1], t[i + limb_idx + 1], hi);
		ADD_CARRY(c[i + limb_idx], t[i + limb_idx], t[i + limb_idx], lo);

		WORD m = t[i] * np[0];
		hi = mp_umul_hi(m, n[limb_idx]);
		lo = mp_umul_lo(m, n[limb_idx]);

		ADD_CARRY(c[limb_idx + i + 1], t[limb_idx + i + 1], t[limb_idx + i + 1], hi);
		ADD_CARRY(c[limb_idx + i], t[limb_idx + i], t[limb_idx + i], lo);
		ADD_CARRY_CLEAR(c[limb_idx + i + 1], t[limb_idx + i + 1], t[limb_idx + i + 1], c[limb_idx + i]);

#else
		/* hand-optimized version */

		WORD t_hi = mp_umul_hi(ar[i], br[limb_idx]);
		WORD t_lo = mp_umul_lo(ar[i], br[limb_idx]);

		ADD_CARRY(c[i + limb_idx], t[i + limb_idx], t[i + limb_idx], t_lo);
		WORD m = t[i] * np[0];
		ADD_CARRY(c[i + limb_idx + 1], t[i + limb_idx + 1], t[i + limb_idx + 1], t_hi);

		WORD mn_hi = mp_umul_hi(m, n[limb_idx]);
		WORD mn_lo = mp_umul_lo(m, n[limb_idx]);

		WORD _t0 = t[limb_idx + i] + mn_lo;
		uint32_t carry = c[limb_idx + i] + (_t0 < mn_lo);
		t[limb_idx + i] = _t0;
		sync_if_needed();

		WORD _t1 = t[limb_idx + i + 1] + mn_hi;
		WORD _t2 = _t1 + carry;
		c[limb_idx + i] = 0;
		sync_if_needed();
		t[limb_idx + i + 1] = _t2;
		c[limb_idx + i + 1] += (_t2 < _t1) + (_t1 < mn_hi);
		sync_if_needed();
#endif
	}
	/* here all t[0] ~ t[S - 1] should be zero. c too */
        t = &t[S];
        c = &c[S];

	while (vote_any<S>(c[limb_idx - 1]))
		ADD_CARRY_CLEAR(c[limb_idx], t[limb_idx], t[limb_idx], c[limb_idx - 1]);

	/* step 2: return t or t - n */
	if (c[S - 1])		// c may be 0 or 1, but not 2
		goto u_is_bigger;

	/* Ugly, but practical.
	 * Can we do this much better with Fermi's ballot()? */
	for (int i = S - 1; i >= 0; i--) {
		if (t[i] > n[i])
			goto u_is_bigger;
		if (t[i] < n[i])
			goto n_is_bigger;
	}

u_is_bigger:
	/* Calculate t - n. Here, c is used for borrow */
	SUB_BORROW(c[limb_idx], t[limb_idx], t[limb_idx], n[limb_idx]);

	if (limb_idx < S - 1) {
		while (vote_any<S>(c[limb_idx] != 0)) {
			SUB_BORROW_CLEAR(c[limb_idx + 1], t[limb_idx + 1],
					 t[limb_idx + 1], c[limb_idx]);
		}
	}

        /* Fall through */

n_is_bigger:
	/* return t */
	ret[limb_idx] = t[limb_idx];
}

//static __device__ WORD ar_pow[MP_SW_MAX_FRAGMENT / 2][MAX_NUM_THREADS];

static __device__ WORD base_pow[MP_MAX_MULTI_EXP][MP_SW_MAX_FRAGMENT / 2][MAX_NUM_THREADS];
#define SW_TABLE_LIMB(table, fragment)  ((table)[(fragment) * MAX_NUM_THREADS + (vec_limb_idx)])
#define SW_TABLE_LIMB2(table, fragment, vec_limb_idx)  (table[fragment * MAX_NUM_THREADS + vec_limb_idx])

template<int S>
static __device__ void
compute_base_pow(WORD* table, const WORD* base, const WORD* n, const WORD* np, int max_fragment)
{
  const int limb_idx = LIMB_IDX(S);
  const int vec_limb_idx = ELEM_IDX(S) * S + limb_idx;
  const int block_elem_idx = BLOCK_ELEM_IDX(S);

  __shared__ WORD s_tmp[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_base_sq[MAX_ELEMS_PER_BLOCK(S) * S];

  WORD *tmp = &s_tmp[block_elem_idx * S];
  WORD *base_sq = &s_base_sq[block_elem_idx * S];

  SW_TABLE_LIMB(table, 0) = base[limb_idx];
  mp_montmul_dev2<S>(base_sq, base, base, n, np, block_elem_idx, limb_idx);

  for (int i = 3; i <= max_fragment; i += 2) {
    tmp[limb_idx] = SW_TABLE_LIMB(table, (i >> 1) - 1);
    mp_montmul_dev2<S>(tmp, tmp, base_sq, n, np, block_elem_idx, limb_idx);
    SW_TABLE_LIMB(table, i >> 1) = tmp[limb_idx];
  }
}

/**
 * assumes ar is 'montgomeritized'
 * assumes sw represents an exponent != 0
 */
#define POW_LIMB(sw, frag_idx) \
  base_pow[(sw)->base[frag_idx]][(sw)->fragment[frag_idx] >> 1][vec_limb_idx]

template<int S>
static __device__ void mp_multi_modexp_dev(
    WORD *ret, const WORD *ar, const struct mp_multi_sw *multi_sw, 
    const WORD *n, const WORD *np)
{
  const int limb_idx = LIMB_IDX(S);
  const int vec_limb_idx = ELEM_IDX(S) * S + limb_idx;
  const int block_elem_idx = BLOCK_ELEM_IDX(S);

  __shared__ WORD s_tmp[MAX_ELEMS_PER_BLOCK(S) * S];
  WORD *tmp = &s_tmp[block_elem_idx * S];

  ret[limb_idx] = POW_LIMB(multi_sw, multi_sw->num_fragments - 1);

  for (int i = multi_sw->num_fragments - 2; i >= 0; i--) {
    for (int k = 0; k < multi_sw->length[i]; k++)
      mp_montmul_dev2<S>(ret, ret, ret, n, np, block_elem_idx, limb_idx);

    if (multi_sw->fragment[i]) {
      tmp[limb_idx] = POW_LIMB(multi_sw, i);
      mp_montmul_dev2<S>(ret, ret, tmp, n, np, block_elem_idx, limb_idx);
    }
  }
}

/**
 * assumes ar is 'montgomeritized'
 * assumes sw represents an exponent != 0
 */
template<int S>
static __device__ void mp_modexp_dev(WORD *ret, const WORD *ar, const struct mp_sw *sw, 
		const WORD *n, const WORD *np)
{
  const int limb_idx = LIMB_IDX(S);
  const int vec_limb_idx = ELEM_IDX(S) * S + limb_idx;
  const int block_elem_idx = BLOCK_ELEM_IDX(S);

  __shared__ WORD s_tmp[MAX_ELEMS_PER_BLOCK(S) * S];
  WORD *tmp = &s_tmp[block_elem_idx * S];
  WORD *ar_pow = &base_pow[0][0][0];

  compute_base_pow<S>(ar_pow, ar, n, np, sw->max_fragment);

  ret[limb_idx] = SW_TABLE_LIMB(ar_pow, sw->fragment[sw->num_fragments - 1] >> 1);

  for (int i = sw->num_fragments - 2; i >= 0; i--) {
    for (int k = 0; k < sw->length[i]; k++)
      mp_montmul_dev2<S>(ret, ret, ret, n, np, block_elem_idx, limb_idx);

    if (sw->fragment[i]) {
      tmp[limb_idx] = SW_TABLE_LIMB(ar_pow, sw->fragment[i] >> 1);
      mp_montmul_dev2<S>(ret, ret, tmp, n, np, block_elem_idx, limb_idx);
    }
  }
}

// A macro to access the nth chunk of a word.
#define WINDOW(word, n) ((word) >> (WINDOW_SIZE * (n)) & ((1 << WINDOW_SIZE) - 1))

/**
 * assumes ar is 'montgomeritized'
 * assumes exp != 0.
 */
template<int S>
static __device__ void
mp_modexp_cached_dev(
    WORD *ret, const WORD *exp, const WORD *powm_cache,
    const WORD *n, const WORD *np, const WORD *r_sq)
{
  const int limb_idx = LIMB_IDX(S);
  const int block_elem_idx = BLOCK_ELEM_IDX(S);

  __shared__ WORD s_tmp[MAX_ELEMS_PER_BLOCK(S) * S];
  WORD *tmp = &s_tmp[block_elem_idx * S];

  // Ret = mont(1)
  tmp[limb_idx] = (limb_idx == 0);
  mp_montmul_dev2<S>(ret, tmp, r_sq, n, np, block_elem_idx, limb_idx);

  for (int i = 0; i < S; i++) {
    for (int j = 0; j < BITS_PER_WORD / WINDOW_SIZE; j++) {
      if (WINDOW(exp[i], j) > 0) {
        tmp[limb_idx] = powm_cache[(WINDOW(exp[i], j) - 1) * S + limb_idx];

        mp_montmul_dev2<S>(ret, ret, tmp, n, np, block_elem_idx, limb_idx);
      }

      powm_cache += ((1 << WINDOW_SIZE) - 1) * S;
    }
  }
}

// Montgomerize input first.
template<int S, int NUM_BASES>
static __global__ void
mp_multi_modexp_mont_kernel(
    WORD *RET, const WORD *BASE, const struct mp_multi_sw *sw,
    const WORD *R_SQ, const WORD *N, const WORD *NP)
{
  __shared__ WORD s_tmp[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_r_sq[S];
  __shared__ WORD s_n[S];
  __shared__ WORD s_np[1];
  __shared__ WORD s_ret[MAX_ELEMS_PER_BLOCK(S) * S];
	
  const int limb_idx = LIMB_IDX(S);
  const int elem_idx = ELEM_IDX(S);
  const int block_elem_idx = BLOCK_ELEM_IDX(S);

  WORD *tmp = &s_tmp[block_elem_idx * S];
  WORD *ret = &s_ret[block_elem_idx * S];

  //if (!elem_idx && !limb_idx) printf("[%d] %p %x\n", blockIdx.x, a, a[limb_idx]);
  // Account for the case when sw represents the exponent 0.
  if (sw[elem_idx].num_fragments == 0) {
    ret[limb_idx] = (limb_idx == 0);
  } else {
    s_r_sq[limb_idx] = R_SQ[limb_idx];
    s_n[limb_idx] = N[limb_idx];
    s_np[0] = NP[0];

    for (int i = 0; i < sw->num_bases; i++)
    {
      tmp[limb_idx] = BASE[(NUM_BASES * elem_idx + i) * S + limb_idx];
      mp_montmul_dev2<S>(tmp, tmp, s_r_sq, s_n, s_np, block_elem_idx, limb_idx);
      compute_base_pow<S>(&base_pow[i][0][0], tmp, s_n, s_np, sw[elem_idx].max_fragment[i]);
      //a += S;
    }
    //a = &s_a[NUM_BASES * block_elem_idx * S];

    mp_multi_modexp_dev<S>(ret, tmp, &sw[elem_idx], s_n, s_np);

    tmp[limb_idx] = (limb_idx == 0); // tmp = 1;
    mp_montmul_dev2<S>(ret, ret, tmp, s_n, s_np, block_elem_idx, limb_idx);
  }

  RET[elem_idx * S + limb_idx] = ret[limb_idx];
}

// Montgomerize input first.
template<int S>
static __global__ void
mp_modexp_mont_kernel(
    WORD *RET, const WORD *A, const struct mp_sw *sw,
    const WORD *R_SQ, const WORD *N, const WORD *NP)
{
  __shared__ WORD s_a[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_r_sq[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_n[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_np[1];
  __shared__ WORD s_ret[MAX_ELEMS_PER_BLOCK(S) * S];
	
  const int limb_idx = LIMB_IDX(S);
  const int elem_idx = ELEM_IDX(S);
  const int block_elem_idx = BLOCK_ELEM_IDX(S);

  WORD *a = &s_a[block_elem_idx * S];
  WORD *r_sq = &s_r_sq[block_elem_idx * S];
  WORD *n = &s_n[block_elem_idx * S];
  WORD *ret = &s_ret[block_elem_idx * S];

  //if (!elem_idx && !limb_idx) printf("[%d] %p %x\n", blockIdx.x, a, a[limb_idx]);
  // Account for the case when sw represents the exponent 0.
  if (sw[elem_idx].num_fragments == 0) {
    ret[limb_idx] = (limb_idx == 0);
  } else {
    a[limb_idx] = A[elem_idx * S + limb_idx];
    r_sq[limb_idx] = R_SQ[limb_idx];
    n[limb_idx] = N[limb_idx];
    s_np[0] = NP[0];

    mp_montmul_dev2<S>(a, a, r_sq, n, s_np, block_elem_idx, limb_idx);

    mp_modexp_dev<S>(ret, a, &sw[elem_idx], n, s_np);

    r_sq[limb_idx] = (limb_idx == 0); // r_sq = 1;
    mp_montmul_dev2<S>(ret, ret, r_sq, n, s_np, block_elem_idx, limb_idx);
  }

  RET[elem_idx * S + limb_idx] = ret[limb_idx];
}

template<int S>
static __global__ void
mp_modexp_cached_kernel(
    WORD *RET, const WORD *EXP, const WORD *powm_cache,
    const WORD *R_SQ, const WORD *N, const WORD *NP)
{
  __shared__ WORD s_exp[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_r_sq[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_n[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_np[1];
  __shared__ WORD s_ret[MAX_ELEMS_PER_BLOCK(S) * S];
	
  const int limb_idx = LIMB_IDX(S);
  const int elem_idx = ELEM_IDX(S);
  const int block_elem_idx = BLOCK_ELEM_IDX(S);

  WORD *exp = &s_exp[block_elem_idx * S];
  WORD *r_sq = &s_r_sq[block_elem_idx * S];
  WORD *n = &s_n[block_elem_idx * S];
  WORD *ret = &s_ret[block_elem_idx * S];

  //if (limb_idx == 0) printf("WORD: %lx: \n", EXP[0]);

  //if (!elem_idx && !limb_idx) printf("[%d] %p %x\n", blockIdx.x, a, a[limb_idx]);
  exp[limb_idx] = EXP[elem_idx * S + limb_idx];
  r_sq[limb_idx] = R_SQ[limb_idx];
  n[limb_idx] = N[limb_idx];
  s_np[0] = NP[0];

  mp_modexp_cached_dev<S>(ret, exp, powm_cache, n, s_np, r_sq);

  r_sq[limb_idx] = (limb_idx == 0); // r_sq = 1;
  mp_montmul_dev2<S>(ret, ret, r_sq, n, s_np, block_elem_idx, limb_idx);

  RET[elem_idx * S + limb_idx] = ret[limb_idx];
}

template<int S>
static __global__ void mp_montgomerize_kernel(
    WORD *RET, const WORD *A,
    const WORD *R_SQ, const WORD *N, const WORD *NP)
{
  __shared__ WORD s_r_sq[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_n[MAX_ELEMS_PER_BLOCK(S) * S];
  __shared__ WORD s_np[1];
  __shared__ WORD s_ret[MAX_ELEMS_PER_BLOCK(S) * S];
	
  const int limb_idx = LIMB_IDX(S);
  const int elem_idx = ELEM_IDX(S);
  const int block_elem_idx = BLOCK_ELEM_IDX(S);

  WORD *r_sq = &s_r_sq[block_elem_idx * S];
  WORD *n = &s_n[block_elem_idx * S];
  WORD *ret = &s_ret[block_elem_idx * S];

  ret[limb_idx] = A[elem_idx * S + limb_idx];
  r_sq[limb_idx] = R_SQ[limb_idx];
  n[limb_idx] = N[limb_idx];
  s_np[0] = NP[0];

  mp_montmul_dev2<S>(ret, ret, r_sq, n, s_np, block_elem_idx, limb_idx);

  RET[elem_idx * S + limb_idx] = ret[limb_idx];
}

static void
compute_launch_config(
    int &grid_size, dim3 &block_size, int &vec_size,
    int num_elems, int S) {

  vec_size = min(num_elems, MAX_ELEMS_PER_KERNEL(S));

  // Make sure vec_size either divides MAX_ELEMS_PER_BLOCK for it
  // is less than that.
  if ((vec_size > MAX_ELEMS_PER_BLOCK(S)) &&
      (vec_size % MAX_ELEMS_PER_BLOCK(S) != 0)) {
    vec_size -= vec_size % MAX_ELEMS_PER_BLOCK(S);
  }

  block_size = dim3(S, min(vec_size, MAX_ELEMS_PER_BLOCK(S)));
  grid_size = (vec_size + block_size.y - 1) / block_size.y;

  //std::cout << vec_size << " " << len_vector << std::endl;
  //std::cout << grid_size << " " << block_size.y << " " << i << std::endl;
  assert(block_size.y * grid_size == vec_size);
  assert(block_size.x * block_size.y <= MAX_BLOCK_SIZE);
  assert(block_size.x * block_size.y * grid_size <= MAX_NUM_THREADS);
  //std::cout << std::endl << grid_size << " " << block_size.x << " " << block_size.y << std::endl;
}

void mp_montgomerize_gpu_nocopy(
    int len_vector, WORD *ret, WORD *a,
    const WORD *r_sq, const WORD *n, const WORD *np, int S)
{
  //mp_modexp_mont_kernel<<<len_vector/8, dim3(S, 8)>>>(S, ret, a, sw, r_sq, n, np);
  //return;

  int vec_size, grid_size;
  dim3 block_size;
  for (int i = 0; i < len_vector; i += vec_size) {
    compute_launch_config(grid_size, block_size, vec_size, len_vector - i, S);

    switch (S) {
      case S_256: mp_montgomerize_kernel<S_256><<<grid_size, block_size>>>(
                          &ret[i * S], &a[i * S],
                          r_sq, n, np); break;
      case S_512: mp_montgomerize_kernel<S_512><<<grid_size, block_size>>>(
                          &ret[i * S], &a[i * S],
                          r_sq, n, np); break;
      case S_1024: mp_montgomerize_kernel<S_1024><<<grid_size, block_size>>>(
                          &ret[i * S], &a[i * S],
                          r_sq, n, np); break;
      case S_2048: mp_montgomerize_kernel<S_2048><<<grid_size, block_size>>>(
                          &ret[i * S], &a[i * S],
                          r_sq, n, np); break;
      default: assert(false);
    }

    assert(cudaGetLastError() == cudaSuccess);
  }
}

void mp_modexp_cached_gpu_nocopy(
    int len_vector, WORD *ret, const WORD *exp, const WORD *powm_cache,
    const WORD *r_sq, const WORD *n, const WORD *np, int S)
{
  //mp_modexp_mont_kernel<<<len_vector/8, dim3(S, 8)>>>(S, ret, a, sw, r_sq, n, np);
  //return;

  int vec_size, grid_size;
  dim3 block_size;
  for (int i = 0; i < len_vector; i += vec_size) {
    compute_launch_config(grid_size, block_size, vec_size, len_vector - i, S);

    switch (S) {
      case S_256: mp_modexp_cached_kernel<S_256><<<grid_size, block_size>>>(
                          &ret[i * S],
                          &exp[i * S], powm_cache,
                          r_sq, n, np); break;
      case S_512: mp_modexp_cached_kernel<S_512><<<grid_size, block_size>>>(
                          &ret[i * S],
                          &exp[i * S], powm_cache,
                          r_sq, n, np); break;
      case S_1024: mp_modexp_cached_kernel<S_1024><<<grid_size, block_size>>>(
                          &ret[i * S],
                          &exp[i * S], powm_cache,
                          r_sq, n, np); break;
      case S_2048: mp_modexp_cached_kernel<S_2048><<<grid_size, block_size>>>(
                          &ret[i * S],
                          &exp[i * S], powm_cache,
                          r_sq, n, np); break;
      default: assert(false);
    }

    assert(cudaGetLastError() == cudaSuccess);
  }
}


void mp_vec_multi_modexp(
    int vector_len, WORD *ret, WORD *base, struct mp_multi_sw *multi_sw,
    const WORD *r_sq, const WORD *n, const WORD *np, int S)
{
  //mp_modexp_mont_kernel<<<len_vector/8, dim3(S, 8)>>>(S, ret, a, sw, r_sq, n, np);
  //return;

  int vec_size, grid_size;
  dim3 block_size;

  WORD *elem_ret = ret;
  WORD *elem_base = base;
  struct mp_multi_sw *elem_sw = multi_sw;

  for (int i = 0; i < vector_len; i += vec_size) {
    compute_launch_config(grid_size, block_size, vec_size, vector_len - i, S);

    switch (S) {
      case S_256:
        mp_multi_modexp_mont_kernel<S_256, MP_MAX_MULTI_EXP><<<grid_size, block_size>>>(
                   elem_ret, elem_base, elem_sw,
                   r_sq, n, np); break;
      case S_512:
        mp_multi_modexp_mont_kernel<S_512, MP_MAX_MULTI_EXP><<<grid_size, block_size>>>(
                   elem_ret, elem_base, elem_sw,
                   r_sq, n, np); break;
      case S_1024:
        mp_multi_modexp_mont_kernel<S_1024, MP_MAX_MULTI_EXP><<<grid_size, block_size>>>(
                   elem_ret, elem_base, elem_sw,
                   r_sq, n, np); break;
      case S_2048:
        mp_multi_modexp_mont_kernel<S_2048, MP_MAX_MULTI_EXP><<<grid_size, block_size>>>(
                   elem_ret, elem_base, elem_sw,
                   r_sq, n, np); break;
      default: assert(false);
    }

    assert(cudaGetLastError() == cudaSuccess);
    elem_ret += vec_size * S;
    elem_base += MP_MAX_MULTI_EXP * vec_size * S;
    elem_sw += vec_size;
  }
}

void mp_many_modexp_mont_gpu_nocopy(
    int len_vector, WORD *ret, WORD *a, struct mp_sw *sw,
    const WORD *r_sq, const WORD *n, const WORD *np, int S)
{
  //mp_modexp_mont_kernel<<<len_vector/8, dim3(S, 8)>>>(S, ret, a, sw, r_sq, n, np);
  //return;

  int vec_size, grid_size;
  dim3 block_size;
  for (int i = 0; i < len_vector; i += vec_size) {
    compute_launch_config(grid_size, block_size, vec_size, len_vector - i, S);

    switch (S) {
      case S_256: mp_modexp_mont_kernel<S_256><<<grid_size, block_size>>>(
                          &ret[i * S],
                          &a[i * S], &sw[i],
                          r_sq, n, np); break;
      case S_512: mp_modexp_mont_kernel<S_512><<<grid_size, block_size>>>(
                          &ret[i * S],
                          &a[i * S], &sw[i],
                          r_sq, n, np); break;
      case S_1024: mp_modexp_mont_kernel<S_1024><<<grid_size, block_size>>>(
                          &ret[i * S],
                          &a[i * S], &sw[i],
                          r_sq, n, np); break;
      case S_2048: mp_modexp_mont_kernel<S_2048><<<grid_size, block_size>>>(
                          &ret[i * S],
                          &a[i * S], &sw[i],
                          r_sq, n, np); break;
      default: assert(false);
    }

    assert(cudaGetLastError() == cudaSuccess);
  }
}

