#include <cassert>
#include <sys/time.h>
#include <helper_cuda.h>

#define __GPU__
#include "mp_modexp.h"
#include "mp_modexp_gpu.h"

__device__ WORD mp_umul_hi(WORD a, WORD b)
{
#if MP_USE_64BIT
	return __umul64hi(a, b);
#else
	return __umulhi(a, b);
#endif
}

__device__ WORD mp_umul_lo(WORD a, WORD b)
{
	return a * b;
}

template<int S>
__device__ int vote_any(int predicate)
{
	return __any(predicate);
}

template<>
__device__ int vote_any<64>(int predicate)
{
	volatile __shared__ int tmp;

	tmp = 0;
	__syncthreads();
	if (predicate)
		tmp = 1;
	__syncthreads();

	return tmp;
}

/* assumes ar and br are 'montgomeritized' */
template<int S>
__device__ void mp_montmul_dev(WORD *ret, const WORD *ar, const WORD *br,
			       const WORD *n, WORD np, int limb_idx, int idx)
{
#if __CUDA_ARCH__ >= 200 && MP_USE_64BIT
	__shared__ WORD _t[S * 3 * MP_MSGS_PER_BLOCK];
	__shared__ uint32_t _c[S * 3 * MP_MSGS_PER_BLOCK];

	volatile WORD *t = _t + (S * 3 * limb_idx);
	volatile uint32_t *c = _c + (S * 3 * limb_idx);
#else
	__shared__ WORD _t[S * 2 * MP_MSGS_PER_BLOCK];
	__shared__ uint32_t _c[S * 2 * MP_MSGS_PER_BLOCK];

	volatile WORD *t = _t + (S * 2 * limb_idx);
	volatile uint32_t *c = _c + (S * 2 * limb_idx);
#endif

	c[idx] = 0;
	c[idx + S] = 0;
	t[idx] = 0;
	t[idx + S] = 0;
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

		WORD hi = mp_umul_hi(ar[i], br[idx]);
		WORD lo = mp_umul_lo(ar[i], br[idx]);

		ADD_CARRY(c[i + idx + 1], t[i + idx + 1], t[i + idx + 1], hi);
		ADD_CARRY(c[i + idx], t[i + idx], t[i + idx], lo);

		WORD m = t[i] * np;
		hi = mp_umul_hi(m, n[idx]);
		lo = mp_umul_lo(m, n[idx]);

		ADD_CARRY(c[idx + i + 1], t[idx + i + 1], t[idx + i + 1], hi);
		ADD_CARRY(c[idx + i], t[idx + i], t[idx + i], lo);
		ADD_CARRY_CLEAR(c[idx + i + 1], t[idx + i + 1], t[idx + i + 1], c[idx + i]);

#else
		/* hand-optimized version */

		WORD t_hi = mp_umul_hi(ar[i], br[idx]);
		WORD t_lo = mp_umul_lo(ar[i], br[idx]);

		ADD_CARRY(c[i + idx], t[i + idx], t[i + idx], t_lo);
		WORD m = t[i] * np;
		ADD_CARRY(c[i + idx + 1], t[i + idx + 1], t[i + idx + 1], t_hi);

		WORD mn_hi = mp_umul_hi(m, n[idx]);
		WORD mn_lo = mp_umul_lo(m, n[idx]);

		WORD _t0 = t[idx + i] + mn_lo;
		uint32_t carry = c[idx + i] + (_t0 < mn_lo);
		t[idx + i] = _t0;
		sync_if_needed();

		WORD _t1 = t[idx + i + 1] + mn_hi;
		WORD _t2 = _t1 + carry;
		c[idx + i] = 0;
		sync_if_needed();
		t[idx + i + 1] = _t2;
		c[idx + i + 1] += (_t2 < _t1) + (_t1 < mn_hi);
		sync_if_needed();

#endif
	}

	/* here all t[0] ~ t[S - 1] should be zero. c too */

	while (vote_any<S>(c[idx + S - 1] != 0))
		ADD_CARRY_CLEAR(c[idx + S], t[idx + S], t[idx + S], c[idx + S - 1]);

	/* step 2: return t or t - n */
	if (c[S * 2 - 1])		// c may be 0 or 1, but not 2
		goto u_is_bigger;

	/* Ugly, but practical.
	 * Can we do this much better with Fermi's ballot()? */
	for (int i = S - 1; i >= 0; i--) {
		if (t[i + S] > n[i])
			goto u_is_bigger;
		if (t[i + S] < n[i])
			goto n_is_bigger;
	}

u_is_bigger:
	/* return t - n. Here, c is used for borrow */
	SUB_BORROW(c[idx], ret[idx], t[idx + S], n[idx]);

	if (idx < S - 1) {
		while (vote_any<S>(c[idx] != 0)) {
			SUB_BORROW_CLEAR(c[idx + 1], ret[idx + 1],
					ret[idx + 1], c[idx]);
		}
	}
	return;

n_is_bigger:
	/* return t */
	ret[idx] = t[idx + S];
	return;
}

__device__ WORD ar_pow[MAX_STREAMS + 1][MP_SW_MAX_FRAGMENT / 2][2][MP_MAX_NUM_PAIRS][MAX_S];

template<int S>
__global__ void mp_modexp_kernel(int num_pairs,
				 WORD *RET, const WORD *A,
				 const struct mp_sw *_sw,
				 const WORD *N, const WORD *NP, const WORD *R_SQR,
				 unsigned int stream_id,
				 uint8_t *checkbits = 0
)
{
	__shared__ WORD n[S];
	__shared__ WORD _ret[S * MP_MSGS_PER_BLOCK];
	__shared__ WORD _tmp[S * MP_MSGS_PER_BLOCK];

	WORD np;

	const int limb_idx = threadIdx.x / S;
	const int msg_idx = (blockIdx.x / 2) * MP_MSGS_PER_BLOCK + limb_idx;
	const int pair_idx = blockIdx.x % 2;
	const int idx = threadIdx.x % S;

	if (msg_idx >= num_pairs)
		return;

	WORD *ret = _ret + limb_idx * S;
	WORD *tmp = _tmp + limb_idx * S;

	const struct mp_sw *sw = &_sw[pair_idx];
	int num_frags = sw->num_fragments;

	n[idx] = N[pair_idx * MAX_S + idx];
	np = NP[pair_idx * MAX_S + 0];

	tmp[idx] = A[msg_idx * (2 * MAX_S) + pair_idx * MAX_S + idx];
	ret[idx] = R_SQR[pair_idx * MAX_S + idx];

	mp_montmul_dev<S>(tmp, tmp, ret, n, np, limb_idx, idx); /* tmp = ar */
	ar_pow[stream_id][0][pair_idx][msg_idx][idx] = tmp[idx];

	mp_montmul_dev<S>(ret, tmp, tmp, n, np, limb_idx, idx);	/* ret = (a^2)*r */

	for (int i = 3; i <= sw->max_fragment; i+= 2) {
		mp_montmul_dev<S>(tmp, tmp, ret, n, np, limb_idx, idx); /* tmp = (a^i)*r */
		ar_pow[stream_id][i >> 1][pair_idx][msg_idx][idx] = tmp[idx];
	}

	ret[idx] = ar_pow[stream_id][sw->fragment[num_frags - 1] >> 1][pair_idx][msg_idx][idx];

	for (int i = num_frags - 2; i >= 0; i--) {
		for (int k = sw->length[i]; k >= 1; k--)
			mp_montmul_dev<S>(ret, ret, ret, n, np, limb_idx, idx);

		if (sw->fragment[i]) {
			tmp[idx] = ar_pow[stream_id][sw->fragment[i] >> 1][pair_idx][msg_idx][idx];
			mp_montmul_dev<S>(ret, ret, tmp, n, np, limb_idx, idx);
		}
	}

#if MP_MODEXP_OFFLOAD_POST
	/*
	 * now ret = (a^e)*r
	 * calculate montmul(ret, 1) = (a^e)*r*(r^-1) = (a^e)
	 */

	tmp[idx] = (idx == 0);
	mp_montmul_dev<S>(ret, ret, tmp, n, np, limb_idx, idx);
#endif

	RET[msg_idx * (2 * MAX_S) + pair_idx * MAX_S + idx] = ret[idx];

	sync_if_needed();
	if (threadIdx.x == 0 && checkbits != 0)
		*(checkbits + blockIdx.x) = 1;
}

#if MP_MODEXP_OFFLOAD_POST

template<int S>
__global__ void mp_modexp_post_kernel(int num_pairs,
				      WORD *RET, WORD *N, WORD *NP, WORD *R_SQR, WORD *IQMP,
				      uint8_t *checkbits = 0)
{
	__shared__ WORD _p[S * MP_MSGS_PER_BLOCK];
	__shared__ WORD _q[S * MP_MSGS_PER_BLOCK];

	const int limb_idx = threadIdx.x / S;
	const int msg_idx = blockIdx.x * MP_MSGS_PER_BLOCK + limb_idx;
	const int idx = threadIdx.x % S;

#if __CUDA_ARCH__ >= 200 && MP_USE_64BIT
	__shared__ WORD _t[S * 3 * MP_MSGS_PER_BLOCK];
	__shared__ uint32_t _c[S * 3 * MP_MSGS_PER_BLOCK];

	volatile WORD *t = _t + (S * 3 * limb_idx);
	volatile uint32_t *c = _c + (S * 3 * limb_idx);
#else
	__shared__ WORD _t[S * 2 * MP_MSGS_PER_BLOCK];
	__shared__ uint32_t _c[S * 2 * MP_MSGS_PER_BLOCK];

	volatile WORD *t = _t + (S * 2 * limb_idx);
	volatile uint32_t *c = _c + (S * 2 * limb_idx);
#endif

//	__shared__ WORD _tmp[S * MP_MSGS_PER_BLOCK];

	//WORD *ret = _ret + limb_idx * S;
	WORD *tmp = _t + limb_idx * S;

	WORD np = NP[0 * MAX_S + 0];
	__shared__ WORD n[S];

	if (msg_idx >= num_pairs)
		return;

	WORD *p = _p + limb_idx * S;
	WORD *q = _q + limb_idx * S;

	p[idx] = RET[msg_idx * (2 * MAX_S) + 0     + idx];
	q[idx] = RET[msg_idx * (2 * MAX_S) + MAX_S + idx];

	/* calculate M1 - M2 (or M1 - M2 + n, if M1 < M2). Here, c is used for borrow */
	c[idx] = 0;
	c[idx + S] = 0;

	SUB_BORROW(c[idx], p[idx], p[idx], q[idx]);

	if (idx < S - 1) {
		while (vote_any<S>(c[idx] != 0)) {
			SUB_BORROW_CLEAR(c[idx + 1], p[idx + 1],
					p[idx + 1], c[idx]);
		}
	}

	n[idx] = N[0 * MAX_S + idx];

	if (c[S - 1] == 1) {
		c[S - 1] = 0;

		ADD_CARRY(c[idx], p[idx], p[idx], n[idx]);

		if (idx < S - 1) {
			while (vote_any<S>(c[idx] != 0)) {
				ADD_CARRY_CLEAR(c[idx + 1], p[idx + 1],
						p[idx + 1], c[idx]);
			}
		}
	}

	tmp[idx] = R_SQR[0 * MAX_S + idx];
	mp_montmul_dev<S>(p, p, tmp, n, np, limb_idx, idx); /* p = out_p * r */

	tmp[idx] = IQMP[idx];
	mp_montmul_dev<S>(p, p, tmp, n, np, limb_idx, idx); /* p = out_p * iqmp */

	__syncthreads(); /* sync because n is overwritten below */

	/* calculate ((M1 - M2)*(q^-1 mod n_p) mod n_p) <*> n_q */
	c[idx] = 0;
	c[idx + S] = 0;
	t[idx] = 0;
	t[idx + S] = 0;
	n[idx] = N[1 * MAX_S + idx];

	for (int i = 0; i < S; i++) {
		WORD hi = mp_umul_hi(p[i], n[idx]);
		WORD lo = mp_umul_lo(p[i], n[idx]);

		ADD_CARRY(c[i + idx + 2], t[i + idx + 1], t[i + idx + 1], hi);
		ADD_CARRY(c[i + idx + 1], t[i + idx], t[i + idx], lo);
	}

	while (vote_any<S>(c[idx] != 0 || c[idx + S] != 0)) {
		ADD_CARRY_CLEAR(c[idx + S + 1], t[idx + S], t[idx + S], c[idx + S]);
		ADD_CARRY_CLEAR(c[idx + 1], t[idx], t[idx], c[idx]);
	}

	/* ADD M2 */

	ADD_CARRY(c[idx], t[idx], t[idx], q[idx]);

	while (vote_any<S>(c[idx] != 0)) {
		ADD_CARRY_CLEAR(c[idx + 1], t[idx + 1],
				t[idx + 1], c[idx]);
	}

	/* integer to octets */
#if MP_USE_64BIT
	t[idx] =
		((t[idx] >> 56) & 0x00000000000000ffUL) |
		((t[idx] >> 40) & 0x000000000000ff00UL) |
		((t[idx] >> 24) & 0x0000000000ff0000UL) |
		((t[idx] >>  8) & 0x00000000ff000000UL) |
		((t[idx] <<  8) & 0x000000ff00000000UL) |
		((t[idx] << 24) & 0x0000ff0000000000UL) |
		((t[idx] << 40) & 0x00ff000000000000UL) |
		((t[idx] << 56) & 0xff00000000000000UL);

	t[idx + S] =
		((t[idx + S] >> 56) & 0x00000000000000ffUL) |
		((t[idx + S] >> 40) & 0x000000000000ff00UL) |
		((t[idx + S] >> 24) & 0x0000000000ff0000UL) |
		((t[idx + S] >>  8) & 0x00000000ff000000UL) |
		((t[idx + S] <<  8) & 0x000000ff00000000UL) |
		((t[idx + S] << 24) & 0x0000ff0000000000UL) |
		((t[idx + S] << 40) & 0x00ff000000000000UL) |
		((t[idx + S] << 56) & 0xff00000000000000UL);
#else
	t[idx] =
		(t[idx] >> 24) |
		(t[idx] << 24) |
		((t[idx] << 8) & 0xff0000) |
		((t[idx] >> 8) & 0xff00);

	t[idx + S] =
		(t[idx + S] >> 24) |
		(t[idx + S] << 24) |
		((t[idx + S] << 8) & 0xff0000) |
		((t[idx + S] >> 8) & 0xff00);
#endif

	sync_if_needed();

	RET[msg_idx * (2 * MAX_S) + 0 + idx] = t[2 * S - 1 - idx];
	RET[msg_idx * (2 * MAX_S) + S + idx] = t[S - 1 - idx];

	sync_if_needed();
	if (threadIdx.x == 0 && checkbits != 0)
		*(checkbits + blockIdx.x) = 1;

	return;
}
#endif /* MP_MODEXP_OFFLOAD_POST */

void mp_modexp_crt(WORD *a,
		   int cnt, int S,
		   WORD *ret_d, WORD *a_d,
		   struct mp_sw *sw_d,
		   WORD *n_d, WORD *np_d, WORD *r_sqr_d,
		   cudaStream_t stream,
		   unsigned int stream_id,
		   uint8_t *checkbits
)
{
	assert((cnt + MP_MSGS_PER_BLOCK - 1) / MP_MSGS_PER_BLOCK <= MP_MAX_NUM_PAIRS);

	checkCudaErrors(cudaMemcpyAsync(a_d, a,
				sizeof(WORD[2][MAX_S]) * cnt,
				cudaMemcpyHostToDevice,
				stream));

	int num_blocks = ((cnt + MP_MSGS_PER_BLOCK - 1) / MP_MSGS_PER_BLOCK) * 2;
	int num_threads = S * MP_MSGS_PER_BLOCK;	/* # of threads per block */

	/*
	 * We use CPU wall time for the multi-stream case and
	 * GPU event for the non-stream case.
	 * At first glance it looks silly, but this way works around
	 * two weird problems I found.
	 * 1) For multiple streams, cudaEvent_t should not be used
	 *    (otherwise copy and execution do not overlap at all)
	 * - Sangjin
	*/

	switch (S) {
	case S_256:
		mp_modexp_kernel<S_256><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)a_d,
				sw_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				stream_id,
				checkbits);
		break;
	case S_512:
		mp_modexp_kernel<S_512><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)a_d,
				sw_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				stream_id,
				checkbits);
		break;
	case S_1024:
		mp_modexp_kernel<S_1024><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)a_d,
				sw_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				stream_id,
				checkbits);
		break;
	case S_2048:
		mp_modexp_kernel<S_2048><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)a_d,
				sw_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				stream_id,
				checkbits);
		break;
	default:
		fprintf(stderr, "unsupported S(%d)\n", S);
		assert(false);
	}

	assert(cudaGetLastError() == cudaSuccess);
}

int mp_modexp_crt_sync(WORD *ret, WORD *ret_d, WORD *n_d, WORD *np_d, WORD *r_sqr_d, WORD *iqmp_d,
		       int cnt, int S,
		       bool block, cudaStream_t stream,
		       uint8_t *checkbits)
{
	if (block) {
		checkCudaErrors(cudaStreamSynchronize(stream));
	} else {
		cudaError_t ret = cudaStreamQuery(stream);
		if (ret == cudaErrorNotReady)
			return -1;

		assert(ret == cudaSuccess);
	}

#if MP_MODEXP_OFFLOAD_POST
	int num_blocks = ((cnt + MP_MSGS_PER_BLOCK - 1) / MP_MSGS_PER_BLOCK);
	int num_threads = S * MP_MSGS_PER_BLOCK;	/* # of threads per block */

	switch (S) {
	case S_256:
		mp_modexp_post_kernel<S_256><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				(WORD *)iqmp_d,
				checkbits);
		break;
	case S_512:
		mp_modexp_post_kernel<S_512><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				(WORD *)iqmp_d,
				checkbits);
		break;
	case S_1024:
		mp_modexp_post_kernel<S_1024><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				(WORD *)iqmp_d,
				checkbits);
		break;
	case S_2048:
		mp_modexp_post_kernel<S_2048><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				(WORD *)iqmp_d,
				checkbits);
		break;
default:
		fprintf(stderr, "unsupported S(%d)\n", S);
		assert(false);
	}
#endif

	checkCudaErrors(cudaMemcpyAsync(ret, ret_d,
				sizeof(WORD[2][MAX_S]) * cnt,
				cudaMemcpyDeviceToHost,
				stream));

	checkCudaErrors(cudaStreamSynchronize(stream));
	return 0;
}

int mp_modexp_crt_post_kernel(WORD *ret, WORD *ret_d, WORD *n_d, WORD *np_d, WORD *r_sqr_d, WORD *iqmp_d,
			      int cnt, int S,
			      bool block, cudaStream_t stream,
			      uint8_t *checkbits)
{
#if MP_MODEXP_OFFLOAD_POST
	int num_blocks = ((cnt + MP_MSGS_PER_BLOCK - 1) / MP_MSGS_PER_BLOCK);
	int num_threads = S * MP_MSGS_PER_BLOCK;	/* # of threads per block */

	switch (S) {
	case S_256:
		mp_modexp_post_kernel<S_256><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				(WORD *)iqmp_d,
				checkbits);
		break;
	case S_512:
		mp_modexp_post_kernel<S_512><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				(WORD *)iqmp_d,
				checkbits);
		break;
	case S_1024:
		mp_modexp_post_kernel<S_1024><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				(WORD *)iqmp_d,
				checkbits);
		break;
	case S_2048:
		mp_modexp_post_kernel<S_2048><<<num_blocks, num_threads, 0, stream>>>(cnt,
				(WORD *)ret_d,
				(WORD *)n_d,
				(WORD *)np_d,
				(WORD *)r_sqr_d,
				(WORD *)iqmp_d,
				checkbits);
		break;
default:
		fprintf(stderr, "unsupported S(%d)\n", S);
		assert(false);
	}
#endif
	return 0;
}
