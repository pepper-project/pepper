/* BN kernel functions */

#include <cstddef>
#include "bn_kn.h"

#define YINGLIN 0

/* we assume messages in the same block share the rns_ctx */
__shared__ MODULI a[MAX_BS];
__shared__ MODULI b[MAX_BS];
__shared__ MODULI _XI[MSGS_PER_BLOCK][MAX_BS];
__shared__ float _k[MSGS_PER_BLOCK][MAX_BS];

/* use of ret_A and ret_B is non-trivial for performance */
__shared__ MODULI ret_A[MSGS_PER_BLOCK][MAX_BS];	
__shared__ MODULI ret_B[MSGS_PER_BLOCK][MAX_BS];

__shared__ RNS_CTX *rns_ctx;
__shared__ int rns_bs;

__device__ uint32_t umul_lo(uint32_t x, uint32_t y)
{
#if __CUDA_ARCH__ >= 200
	return x * y;
#else
	return __umul24(x, y);
#endif
}

__device__ uint32_t modmul(uint32_t x, uint32_t y, uint32_t n)
{
#if MODULI_BITS <= 16
	uint32_t t = umul_lo(x, y);

	// It seems that this remainder operation takes 20% of total.
	return t % n;
#else
	return ((uint64_t)x * y) % n;
#endif
}

#if YINGLIN

__device__ void
BE_Kawa_yinglin(MODULI *t, MODULI *s, MODULI *s_bs, MODULI *t_bs, MODULI *SiI_mod_si, 
		MODULI Si_t[MAX_BS][MAX_BS], MODULI *S_t, int bsize, float alpha)
{
	int idx = threadIdx.x;

	int K_;
	float sigma;
	uint64_t y; // use 64bit integer to prevent overflow

	MODULI *XI = _XI[threadIdx.y];
	float *k = _k[threadIdx.y];

	/* step 1: XI[] = x[mi]*Mi^-1[mi] mod mi */
	XI[idx] = modmul(s[idx], SiI_mod_si[idx], s_bs[idx]);

	/* step 2: */
	k[idx] = (float)XI[idx]/(float)s_bs[idx];

	__syncthreads();

	sigma = alpha; y = 0; K_ = 0;

	/* step 3: */
	for(int j = 0; j < bsize; j++)
	{
		sigma = sigma + k[j];
		K_ += (int)sigma; 
		sigma = sigma - (int)sigma;
		y += XI[j] * (uint64_t)Si_t[j][idx];
	}

	/* step 4: */
	t[idx] = (y - K_*S_t[idx]) % t_bs[idx];
	//TODO: since 0 < K_ < n, K_*S_t[idx] can be read from a precomputed table
}

#else

#if MODULI_BITS > 15
	#error not implemented yet
#endif

/* 
 * extend <s>s_bs => <t>t_bs
 * output: t
 */
__device__ void
BE_Kawa(MODULI *t, MODULI *s, MODULI *s_bs, MODULI *t_bs, MODULI *SiI_mod_si, MODULI *Si_t, MODULI *S_t, float alpha)
{
	int32_t y = 0;
	int idx = threadIdx.x;
	MODULI *XI = _XI[threadIdx.y];
	float *k = _k[threadIdx.y];

	/* step 1: XI[] = x[mi]*Mi^-1[mi] mod mi */
	XI[idx] = modmul(s[idx], SiI_mod_si[idx], s_bs[idx]);
	
	/* step 2: */
	k[idx] = fdividef((float)XI[idx], (float)s_bs[idx]);

	__syncthreads();

	/* 
	 * step 3: calculate the sum of k,
	 * It seems that parallel reduction of k is beneficial, 
	 * for both 512 and 1024 bits, even with excessive sync().
	 */
#define span(stride) \
	if (idx + stride < MAX_BS && (idx & ((stride * 2) - 1)) == 0) \
		k[idx] += k[idx + stride]; \
	__syncthreads()

	span(1);
	span(2);
	span(4);
	span(8);
	span(16);
	span(32);
#if MAX_BS > 64
	span(64);
#endif
#if MAX_BS > 128
	span(128);
#endif
#undef span

	/* step 4: calculate the sum of y */
	for (int j = 0; j < rns_bs; j += 4) {
		y += umul_lo(XI[j], Si_t[idx + j * MAX_BS]) +
			umul_lo(XI[j + 1], Si_t[idx + (j + 1) * MAX_BS]) +
			umul_lo(XI[j + 2], Si_t[idx + (j + 2) * MAX_BS]) +
			umul_lo(XI[j + 3], Si_t[idx + (j + 3) * MAX_BS]);
		/* keep y from overflow */
		if (y > (t_bs[idx] << (MODULI_BITS + 2)))
			y -= (t_bs[idx] << (MODULI_BITS + 2));
	}

	/* step 5 */
	//t[idx] = (y - (umul_lo((uint32_t)(k[0] + alpha), S_t[idx])) % t_bs[idx];
	int32_t u = umul_lo((uint32_t)(k[0] + alpha), S_t[idx]);
	u = (y - u) % (int32_t)t_bs[idx];
	if (u < 0)
		t[idx] = u + t_bs[idx];
	else
		t[idx] = u;

	__syncthreads();
}

#endif	/* else YINGLIN */

/* RNS based montgomery multiplication. w = x*y*B^(-1) mod N 
 * Input: <x>a,b <y>a,b
 * Output: <w>a,b
 */
__device__ void
bn_mod_mul_MONT_RNS(MODULI *w_A, MODULI *w_B,
		MODULI *x_A, MODULI *x_B,
		MODULI *y_A, MODULI *y_B)
{
#define tmp0 w_A
#define tmp1 w_B

	/* Get thread id, #threads = RNS base size */
	int idx = threadIdx.x;

	/* step 1: s = x * y */
	tmp0[idx] = modmul(x_A[idx], y_A[idx], a[idx]);
	tmp1[idx] = modmul(x_B[idx], y_B[idx], b[idx]);

	/* step 2: t = s * N' mod R */
	tmp1[idx] = modmul(tmp1[idx], rns_ctx->Np_B[idx], b[idx]); //t_B

	/* step 3: base extension 1: <t>b => <t>a, compute t_A */
#if YINGLIN
	BE_Kawa_yinglin(tmp1, tmp1, b, a, rns_ctx->BiI_mod_bi, rns_ctx->Bi_A, rns_ctx->B_A, rns_bs, 0.0f); //tmp1 is t_A after BE
#else
	BE_Kawa(tmp1, tmp1, b, a, rns_ctx->BiI_mod_bi, (MODULI *)rns_ctx->Bi_A, rns_ctx->B_A, 0.0f); //tmp1 is t_A after BE
#endif

	/* step 4: u = t * N */
	tmp1[idx] = modmul(tmp1[idx], rns_ctx->N_A[idx], a[idx]); //u_A

	/* step 5: v = s + u */
	tmp1[idx] = tmp0[idx] + tmp1[idx]; //v_A
#if MODULI_BITS >= 15
	tmp1[idx] = min(tmp1[idx], tmp1[idx] - a[idx]);
#endif

	/* step 6: w = v/R, that is, <w>a = <v>a * <B^-1 mod A>a */
	w_A[idx] = modmul(tmp1[idx], rns_ctx->BI_modA_A[idx], a[idx]);

	/* step 7 base extension 2: <w>a => <w>a,b, compute w_B */
#if YINGLIN
	BE_Kawa_yinglin(w_B, w_A, a, b, rns_ctx->AiI_mod_ai, rns_ctx->Ai_B, rns_ctx->A_B, rns_bs, 0.5f); 
#else
	BE_Kawa(w_B, w_A, a, b, rns_ctx->AiI_mod_ai, (MODULI *)rns_ctx->Ai_B, rns_ctx->A_B, 0.5f); 
#endif

#undef tmp0
#undef tmp1
}

/* Constant Length Nonzero Windows method for exponentiation */
__device__ void
CLNW_exp(MODULI *y_A, MODULI *y_B, 
		MODULI *x_rsd_A, MODULI *x_rsd_B, 
		MODULI M_A[][MAX_WIN][MAX_BS],
		MODULI M_B[][MAX_WIN][MAX_BS])
{
	int tid = threadIdx.x;
	int mid = blockIdx.x * blockDim.y + threadIdx.y;

	/* Pre-compute the odd power of M, M[0]->M, M[1]->M^3 ... */

	// Copy M
	M_A[mid][0][tid] = x_rsd_A[tid];
	M_B[mid][0][tid] = x_rsd_B[tid];

	// Compute M^2
	bn_mod_mul_MONT_RNS(y_A, y_B,
			x_rsd_A, x_rsd_B,
			x_rsd_A, x_rsd_B);

	// Compute the odd power of M, M^3, M^5, M^7
	for(int i = 1; i <= ((rns_ctx->CLNW_maxwin - 1) >> 1); i++)
		bn_mod_mul_MONT_RNS(M_A[mid][i], M_B[mid][i], 
				M_A[mid][i - 1], M_B[mid][i - 1],
				y_A, y_B);

	/* Start exponentiation */
	int CLNW_num = rns_ctx->CLNW_num;
	int win;

	// Copy the most significant window to y
	win = ((rns_ctx->CLNW[CLNW_num - 1]) - 1) >> 1;
	y_A[tid] = M_A[mid][win][tid];
	y_B[tid] = M_B[mid][win][tid];

	// Iterate windows
	for(int i = CLNW_num - 2; i >= 0; i--) {
		for(int j = rns_ctx->CLNW_len[i]; j >= 1; j--)
			bn_mod_mul_MONT_RNS(y_A, y_B, y_A, y_B, y_A, y_B);

		if (rns_ctx->CLNW[i]) {
			win = rns_ctx->CLNW[i] >> 1;

			bn_mod_mul_MONT_RNS(y_A, y_B, 
					y_A, y_B, 
					M_A[mid][win], 
					M_B[mid][win]);
		}
	}
}

/* Modular exponentiation based on RNS montgomery multiplication
 * y = x^e (mod n)
 * y_A, y_B, x_A, x_B are 2D arrays. 1st dim is message num, and 2nd dim is rns representation.
 * 
 * Implementation: Each message is processed by one block.
 */
__global__ void
BN_mod_exp_RNS_MONT_batch_kn(MODULI *y_A, MODULI *y_B, 
		MODULI *x_A, MODULI *x_B,
		MODULI M_A[][MAX_WIN][MAX_BS],
		MODULI M_B[][MAX_WIN][MAX_BS],
		RNS_CTX *rns_ctx_arr, int *rns_ctx_idx)
{
	int tid = threadIdx.x;
	int mid = threadIdx.y;
	rns_ctx = &rns_ctx_arr[rns_ctx_idx[blockIdx.x]];
	
	rns_bs = rns_ctx->bs;
	a[tid] = rns_ctx->a[tid];
	b[tid] = rns_ctx->b[tid];

	for (int t = 0; t < MAX_BS; t += rns_bs) {
		if (t + tid < MAX_BS) {
			_XI[mid][t + tid] = 0;
			_k[mid][t + tid] = 0.0f;
		}
	}

	int offset = (blockIdx.x * blockDim.y + threadIdx.y) * MAX_BS;

	/* Montgomery reduction of x */
	bn_mod_mul_MONT_RNS(ret_A[mid], ret_B[mid],
			x_A + offset, x_B + offset,
			rns_ctx->Bsqr_modN_A, rns_ctx->Bsqr_modN_B);

	/* modular multiplication with Constant Length Nonzero Windows */
	CLNW_exp(ret_A[mid], ret_B[mid], ret_A[mid], ret_B[mid], M_A, M_B);

	/* De-Montgomeryizing */
	bn_mod_mul_MONT_RNS(y_A + offset, y_B + offset, 
			ret_A[mid], ret_B[mid], 
			rns_ctx->ONE_A, rns_ctx->ONE_B);
}
