/* 
 * This file includes wrappers for BN cuda kernel functions. 
 * This should be where gpu workload sheduler is. We can integrate this into 
 * OpenSSL engine to accumulate functions calls, block for a while, send 
 * together to device, and return results together. In addition, we should
 * filter calls according the size of exponent. Those with small exponent
 * should be forward to CPU, whereas those with big exponent should be 
 * offloaded to GPU.
 */

#include <iostream>
#include <sstream>
using namespace std;

#include <assert.h>

#include <openssl/bn.h>
#include <openssl/bio.h>

#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "rsa_cuda.h"

#define bnSafeCall(ret) __bnSafeCall(ret, __FILE__, __LINE__)

inline void __bnSafeCall( int ret, const char *file, const int line )
{
	if (1 != ret) {
		fprintf(stderr, "bnSafeCall() error in file <%s>, line %i.\n",
				file, line);
		exit(-1);
	}
}

void convertRNS2Radix(BIGNUM *r, MODULI *rns, 
		int bs, BIGNUM* M, BIGNUM *Mi[], MODULI *MiI_mod_mi);

void partCLNWin(BIGNUM *d, RNS_CTX *rns_ctx);
void partVLNWin(BIGNUM *d, RNS_CTX *rns_ctx);

int initRNSBase(BIGNUM* N, BIGNUM* bsA[], BIGNUM* bsB[])
{
	BIGNUM *gcd;
	BIGNUM *tmp, *N2, *N4, *A, *B;
	BIGNUM *word, *r_bound, *r;
	BN_CTX *ctx;

	/* initialize BN objects */
	ctx = BN_CTX_new();
	gcd = BN_new();
	tmp = BN_new();
	N2 = BN_new();
	N4 = BN_new();
	A = BN_new();
	B = BN_new();

	word = BN_new();
	r_bound = BN_new();
	r = BN_new();

	BN_lshift(N2, N, 1);
	BN_lshift(N4, N, 2);

	/* 2^32 % m < 2 ** R_BOUND_BITS */
	BN_set_bit(word, sizeof(MODULI_BITS));
	BN_set_bit(r_bound, R_BOUND_BITS);

	/* Search moduli for base set B, B>=4N */
	BN_set_bit(tmp, MODULI_BITS);
	BN_add_word(tmp, 1); 

	BN_set_word(A, 1);
	BN_set_word(B, 1);

	int bs = 0;

	while (BN_cmp(A, N2) == -1 && BN_cmp(B, N4)) {
again_B:
		BN_sub_word(tmp, 2);
		if (BN_cmp(tmp, BN_value_one()) == 0) {
			fprintf(stderr, "couldn't set residue base A and B\n");
			assert(false);
		}
		BN_nnmod(r, word, tmp, ctx);
		if (BN_cmp(r, r_bound) != -1)
			goto again_B;

		for (int i = 0; i < bs; i++) {
			BN_gcd(gcd, bsB[i], tmp, ctx);
			if (BN_cmp(gcd, BN_value_one()) != 0)
				goto again_B;

			BN_gcd(gcd, bsA[i], tmp, ctx);
			if (BN_cmp(gcd, BN_value_one()) != 0)
				goto again_B;
		}

		BN_copy(bsB[bs], tmp);
		BN_mul(B, B, tmp, ctx);

again_A:
		BN_sub_word(tmp, 2);
		if (BN_cmp(tmp, BN_value_one()) == 0) {
			fprintf(stderr, "couldn't set residue base A and B\n");
			assert(false);
		}
		BN_nnmod(r, word, tmp, ctx);
		if (BN_cmp(r, r_bound) != -1)
			goto again_B;

		for (int i = 0; i < bs + 1; i++) {
			BN_gcd(gcd, bsB[i], tmp, ctx);
			if (BN_cmp(gcd, BN_value_one()) != 0)
				goto again_A;

			if (i < bs) {
				BN_gcd(gcd, bsA[i], tmp, ctx);
				if (BN_cmp(gcd, BN_value_one()) != 0)
					goto again_A;
			}
		}

		BN_copy(bsA[bs], tmp);
		BN_mul(A, A, tmp, ctx);

		bs++;
		assert(bs <= MAX_BS);
	}

	/* release objects */
	BN_free(tmp);
	BN_free(gcd);
	BN_free(N2);
	BN_free(N4);
	BN_free(A);
	BN_free(B);
	BN_free(word);
	BN_free(r_bound);
	BN_free(r);
	BN_CTX_free(ctx);

	return bs;
}

static RNS_CTX g_rns_ctx[MAX_RNS_CTXS];
static RNS_CTX *g_rns_ctx_d = NULL;

/* initialize rns context object and return */
RNS_CTX *RNS_CTX_new(BIGNUM *N, BIGNUM *d)
{
	BN_CTX *bn_ctx = BN_CTX_new();
	BIGNUM *tmp = BN_new();

	RNS_CTX *rns_ctx = NULL;

	BIGNUM* BS_A[MAX_BS];
	BIGNUM* BS_B[MAX_BS];

	int bs;

	for (int i = 0; i < MAX_RNS_CTXS; i++) {
		if (g_rns_ctx[i].bs == 0) {
			rns_ctx = &g_rns_ctx[i];

			/* To be sure, clear the buffer */
			memset(rns_ctx, 0, sizeof(RNS_CTX));
			rns_ctx->index = i;
			break;
		}
	}

	if (rns_ctx == NULL)
		assert(false); /* no empty slot! */


	/* initialize moduli BN obj */
	for(int i=0; i<MAX_BS; i++)
	{
		BS_A[i] = BN_new();
		BS_B[i] = BN_new();
	}

	/* Generate moduli */
	bs = initRNSBase(N, BS_A, BS_B);

	//  cout << "base size: " << bs<< endl;

	/* fill up RNS_CTX */
	// base size
	rns_ctx->bs = bs;

	// exponent d, for any given RSA key, d is constant
	int d_numbits = BN_num_bits(d);
	rns_ctx->d_num_bits = d_numbits;
	rns_ctx->d_len = d->top;
	memcpy(rns_ctx->d, d->d, d->top*sizeof(BN_ULONG));

	// partition CLNW windows
	partCLNWin(d, rns_ctx);

	// partition VLNW windows
	//partVLNWin(d, rns_ctx);

	// copy base set A
	for(int i=0; i<bs; i++)
		rns_ctx->a[i] = BS_A[i]->d[0];

	// copy base set B
	for(int i=0; i<bs; i++)
		rns_ctx->b[i] = BS_B[i]->d[0];

	// Compute A 
	BIGNUM *A = BN_new();
	BN_set_word(A, 1);
	for(int i=0; i<bs; i++)
		BN_mul(A, A, BS_A[i], bn_ctx);

	rns_ctx->A = A;

	// A >= 2N
	BN_lshift(tmp, N, 1);
	assert( BN_cmp(A, tmp) >=0 );

	// Compute B
	BIGNUM *B = BN_new();
	BN_set_word(B, 1);
	for(int i=0; i<bs; i++)
		BN_mul(B, B, BS_B[i], bn_ctx);

	rns_ctx->B = B;

	// B >= 4N
	BN_lshift(tmp, N, 2);
	assert( BN_cmp(B, tmp) >=0 );

	// A_B
	for(int i=0; i<bs; i++)
	{
		bnSafeCall( BN_mod(tmp, A, BS_B[i], bn_ctx) );
		rns_ctx->A_B[i] = tmp->d[0];
	}

	// B_A
	for(int i=0; i<bs; i++)
	{
		bnSafeCall( BN_mod(tmp, B, BS_A[i], bn_ctx) );
		rns_ctx->B_A[i] = tmp->d[0];
	}

	// Bsqr_ModN
	BIGNUM *Bsqr_modN = BN_new();
	BN_mod_sqr(Bsqr_modN, B, N, bn_ctx);
	for(int i=0; i<bs; i++)
	{
		BN_mod(tmp, Bsqr_modN, BS_A[i], bn_ctx);
		rns_ctx->Bsqr_modN_A[i] = tmp->d[0];

		BN_mod(tmp, Bsqr_modN, BS_B[i], bn_ctx);
		rns_ctx->Bsqr_modN_B[i] = tmp->d[0];
	}

	rns_ctx->Bsqr_modN = Bsqr_modN;

	// Calculate Np, B*Bi(mod N) - N*Np = 1, here R = B
	BIGNUM *Np = BN_new();
	BN_mod_inverse(Np, B, N, bn_ctx);
	bnSafeCall( BN_mul(Np, Np, B, bn_ctx) );
	bnSafeCall( BN_sub_word(Np, 1) );
	bnSafeCall( BN_div(Np, NULL, Np, N, bn_ctx) );

	// Np_B[]
	for(int i=0; i<bs; i++)
	{
		bnSafeCall( BN_mod(tmp, Np, BS_B[i], bn_ctx) );
		rns_ctx->Np_B[i] = tmp->d[0];
	}

	// N_A[]
	for(int i=0; i<bs; i++)
	{
		bnSafeCall( BN_mod(tmp, N, BS_A[i], bn_ctx) );
		rns_ctx->N_A[i] = tmp->d[0];
	}

	// BI_modA_A[] = <B^-1 mod A>a
	BIGNUM *BI_modA = BN_new();
	BN_mod_inverse(BI_modA, B, A, bn_ctx);
	for(int i=0; i<bs; i++)
	{
		bnSafeCall( BN_mod(tmp, BI_modA, BS_A[i], bn_ctx) );
		rns_ctx->BI_modA_A[i] = tmp->d[0];
	}

	// Ai = A/a[i], AiI_mod_ai = Ai^-1 mod ai
	// Ai_B[][], Ai in set B
	for(int i=0; i<bs; i++)
	{
		BIGNUM *Ai = BN_new();
		// Ai
		bnSafeCall( BN_div(Ai, NULL, A, BS_A[i], bn_ctx) );

		for(int j=0; j<bs; j++)
		{
			bnSafeCall( BN_mod(tmp, Ai, BS_B[j], bn_ctx) );
			rns_ctx->Ai_B[i][j] = tmp->d[0];
		}

		BN_mod_inverse(tmp, Ai, BS_A[i], bn_ctx);
		rns_ctx->AiI_mod_ai[i] = tmp->d[0];

		rns_ctx->Ai[i] = Ai;
	}

	// Bi = B/b[i], BiI_mod_bi = Bi^-1 mod bi
	// Bi_A[][], Bi in set A
	for(int i=0; i<bs; i++)
	{
		BIGNUM *Bi = BN_new();
		// Bi
		bnSafeCall( BN_div(Bi, NULL, B, BS_B[i], bn_ctx) );
		for(int j=0; j<bs; j++)
		{
			bnSafeCall( BN_mod(tmp, Bi, BS_A[j], bn_ctx) );
			rns_ctx->Bi_A[i][j] = tmp->d[0];
		}

		BN_mod_inverse(tmp, Bi, BS_B[i], bn_ctx);
		rns_ctx->BiI_mod_bi[i] = tmp->d[0];

		rns_ctx->Bi[i] = Bi;
	}

	// constant one
	for(int i=0; i<bs; i++)
	{
		rns_ctx->ONE_A[i] = 1;
		rns_ctx->ONE_B[i] = 1;
	}

	/* release */ 
	BN_CTX_free(bn_ctx);
	BN_free(tmp);

	for(int i=0; i<MAX_BS; i++)
	{
		BN_free(BS_A[i]);
		BN_free(BS_B[i]);
	}

	return rns_ctx;
}

void cpyRNSCTX2Dev()
{
	if (g_rns_ctx_d == NULL) {
		checkCudaErrors(cudaMalloc(&g_rns_ctx_d, 
					MAX_RNS_CTXS * sizeof(RNS_CTX)));
	}

	checkCudaErrors(cudaMemcpy(g_rns_ctx_d, g_rns_ctx, 
			MAX_RNS_CTXS * sizeof(RNS_CTX), 
			cudaMemcpyHostToDevice));
}

void RNS_CTX_free(RNS_CTX *rns_ctx) 
{
	if (rns_ctx == NULL)
		return;

	int bsize = rns_ctx->bs;

	// Release all host objects
	BN_free(rns_ctx->A); BN_free(rns_ctx->B);
	BN_free(rns_ctx->Bsqr_modN);

	for(int i = 0; i < bsize; i++) { 
		BN_free(rns_ctx->Ai[i]); 
		BN_free(rns_ctx->Bi[i]); 
	}

	rns_ctx->bs = 0;
}

__global__ void
BN_mod_exp_RNS_MONT_batch_kn(MODULI *y_A, MODULI *y_B, 
		MODULI *x_A, MODULI *x_B, 
		MODULI M_A[][MAX_WIN][MAX_BS],
		MODULI M_B[][MAX_WIN][MAX_BS],
		RNS_CTX *rns_ctx, int *rns_ctx_idx);


static struct dev_context {
	bool initialized;

	cudaEvent_t evt_begin;
	cudaEvent_t evt_end;

	MODULI *M_A_d;
	MODULI *M_B_d;

	MODULI *b_A, *b_B, *b_A_d, *b_B_d;
	MODULI *r_A, *r_B, *r_A_d, *r_B_d;

	int *rns_ctx_idx;
	int *rns_ctx_idx_d;
} g_dev[8];

static void *alloc_pinned_mem(int size)
{
	cudaError_t err;
	void *ret;

	err = cudaHostAlloc(&ret, size, cudaHostAllocPortable);
	assert(err == cudaSuccess);

	return ret;
}

static struct dev_context *get_device()
{
	int dev_id;
	struct dev_context *dev;
	
	checkCudaErrors(cudaGetDevice(&dev_id));
	dev = &g_dev[dev_id];

	if (!dev->initialized) {
		int size;

		dev->initialized = true;

		checkCudaErrors(cudaEventCreate(&dev->evt_begin));
		checkCudaErrors(cudaEventCreate(&dev->evt_end));

		// XXX
		size = sizeof(MODULI) * MAX_NUM_MSG * MAX_WIN * MAX_BS;
		checkCudaErrors(cudaMalloc(&dev->M_A_d, size));
		checkCudaErrors(cudaMalloc(&dev->M_B_d, size));

		size = sizeof(MODULI) * MAX_NUM_MSG * MAX_BS;
		dev->b_A = (MODULI *)alloc_pinned_mem(size);
		dev->b_B = (MODULI *)alloc_pinned_mem(size);
		dev->r_A = (MODULI *)alloc_pinned_mem(size);
		dev->r_B = (MODULI *)alloc_pinned_mem(size);
		
		checkCudaErrors(cudaMalloc(&dev->b_A_d, size));
		checkCudaErrors(cudaMalloc(&dev->b_B_d, size));
		checkCudaErrors(cudaMalloc(&dev->r_A_d, size));
		checkCudaErrors(cudaMalloc(&dev->r_B_d, size));

		size = MAX_NUM_MSG * sizeof(int);
		dev->rns_ctx_idx = (int *)alloc_pinned_mem(size);
		checkCudaErrors(cudaMalloc(&dev->rns_ctx_idx_d, size));
	}

	return dev;
}

struct msg {
	RNS_CTX *ctx;
	int org_id;
};

int msgcmp(const void *v1, const void *v2)
{
	struct msg *a1 = (struct msg *)v1;
	struct msg *a2 = (struct msg *)v2;

	if ((unsigned long)a1->ctx > (unsigned long)a2->ctx)
		return 1;
	if ((unsigned long)a1->ctx < (unsigned long)a2->ctx)
		return 1;

	return 0;
}

/* shuffle so that all messages in a block have the same context */
int map(int n, RNS_CTX *rns_ctx[], int *fmap)
{
	struct msg msgs[MAX_NUM_MSG];

	for (int i = 0; i < n; i++) {
		msgs[i].ctx = rns_ctx[i];
		msgs[i].org_id = i;
	}

	qsort((void *)msgs, n, sizeof(struct msg), msgcmp);

	int k = 0;
	for (int i = 0; i < n; i++) {
		if (i > 0 && msgs[i].ctx != msgs[i - 1].ctx)
			while (k % MSGS_PER_BLOCK != 0)
				k++;
		fmap[msgs[i].org_id] = k;
		k++;
	}

	while (k % MSGS_PER_BLOCK != 0)
		k++;

	return k;
}

/* Wrapper for BN exponentiation based on RNS Montgomery multiplication with batching */
float BN_mod_exp_mont_batch_cu(BIGNUM *r[], BIGNUM *b[], int n, RNS_CTX *rns_ctx[])
{
	assert(n > 0);
	assert(n <= MAX_NUM_MSG);

	int fmap[MAX_NUM_MSG];
	int m = map(n, rns_ctx, fmap);
	assert(m % MSGS_PER_BLOCK == 0);
	assert(m <= MAX_NUM_MSG);

	int bsize = rns_ctx[0]->bs;
	int memsize = m * MAX_BS * sizeof(MODULI);

	struct dev_context *dev = get_device();

	for (int i = 0; i < n; i++) {
		assert(bsize == rns_ctx[i]->bs);
		dev->rns_ctx_idx[fmap[i] / MSGS_PER_BLOCK] = rns_ctx[i]->index;
	}

	/* Convert base to RNS representation */
	// TODO: move this part to device
	for(int i = 0; i < n; i++) {
		for(int j = 0; j < bsize; j++) {
			*(dev->b_A + fmap[i] * MAX_BS + j) = 
					BN_mod_word(b[i], rns_ctx[i]->a[j]);
			*(dev->b_B + fmap[i] * MAX_BS + j) = 
					BN_mod_word(b[i], rns_ctx[i]->b[j]);
		}
	}

	checkCudaErrors(cudaMemcpyAsync(dev->rns_ctx_idx_d, dev->rns_ctx_idx, 
			(m / MSGS_PER_BLOCK) * sizeof(int), 
			cudaMemcpyHostToDevice, 0));

	// copy base numbers
	checkCudaErrors(cudaMemcpyAsync(dev->b_A_d, dev->b_A, memsize, 
				cudaMemcpyHostToDevice, 0));
	checkCudaErrors(cudaMemcpyAsync(dev->b_B_d, dev->b_B, memsize, 
				cudaMemcpyHostToDevice, 0));

	float elapsed_ms_kernel;
	dim3 threads_per_block(bsize, MSGS_PER_BLOCK);
	int num_blocks = m / MSGS_PER_BLOCK;

	checkCudaErrors(cudaEventRecord(dev->evt_begin, 0));
	/* call kernel function, use one block per message */
	BN_mod_exp_RNS_MONT_batch_kn<<<num_blocks, threads_per_block>>>(dev->r_A_d, dev->r_B_d, 
			dev->b_A_d, dev->b_B_d, 
			(MODULI (*)[MAX_WIN][MAX_BS])dev->M_A_d, 
			(MODULI (*)[MAX_WIN][MAX_BS])dev->M_B_d,
			g_rns_ctx_d, dev->rns_ctx_idx_d);
	checkCudaErrors(cudaEventRecord(dev->evt_end, 0));

	/* copy back result */
	checkCudaErrors(cudaMemcpyAsync(dev->r_A, dev->r_A_d, memsize, 
				cudaMemcpyDeviceToHost, 0));
	checkCudaErrors(cudaMemcpyAsync(dev->r_B, dev->r_B_d, memsize, 
				cudaMemcpyDeviceToHost, 0));

	checkCudaErrors(cudaThreadSynchronize());
	checkCudaErrors(cudaEventElapsedTime(&elapsed_ms_kernel, 
			dev->evt_begin, dev->evt_end));

	/* Convert results from rns to radix representation */
	// TODO: move this part to device
	for(int i = 0; i < n; i++) { 
		// convert with set A
		convertRNS2Radix(r[i], dev->r_A + fmap[i] * MAX_BS, bsize, 
				rns_ctx[i]->A, rns_ctx[i]->Ai, 
				rns_ctx[i]->AiI_mod_ai);
	}

	return elapsed_ms_kernel;
}

void convertRNS2Radix(BIGNUM *r, MODULI *rns, 
		int bs, BIGNUM* M, BIGNUM *Mi[], MODULI *MiI_mod_mi)
{
	BN_CTX *bn_ctx = BN_CTX_new();
	BIGNUM *p = BN_new();

	BN_zero(r);

	for(int i=0; i<bs; i++)
	{
		BN_copy(p, Mi[i]);
		BN_mul_word(p, rns[i]);
		BN_mul_word(p, MiI_mod_mi[i]);

		BN_add(r, r, p);
	}

	BN_nnmod(r, r, M, bn_ctx);

	/* release object */
	BN_free(p);
	BN_CTX_free(bn_ctx);
}

/* Partition CLNW windows */
void partCLNWin(BIGNUM *d, RNS_CTX *rns_ctx)
{
	int i=0;
	int w=0;
	int maxwin = 0;
	int nzoWlen = 0;

	int d_numbits = BN_num_bits(d);

	/* convert bits to integer array, this can make partition easier. */
	int dbits[d_numbits];
	for(int i=0; i<d_numbits; i++)
		dbits[i] = BN_is_bit_set(d, i);

	/* determine non-zero window size according to "High-Speed RSA Implementation" */
	if (d_numbits < 256)
		nzoWlen = 4;
	else if (d_numbits < 768)
		nzoWlen = 5;
	else if (d_numbits < 1792)
		nzoWlen = 6;
	else
		nzoWlen = 7;

	/* Start to do partition */
	while(i<d_numbits)
	{
		// skip next 0s
		if (dbits[i] == 0)
		{
			rns_ctx->CLNW[w] = 0;

			rns_ctx->CLNW_len[w] = 1;
			while(1)
				if (dbits[++i]==1)
				{
					w++;
					break;
				}
				else
					rns_ctx->CLNW_len[w]++;
		}

		// collect next nzoWLen bits
		int j;
		rns_ctx->CLNW[w] = 0;
		rns_ctx->CLNW_len[w] = 0;

		for(j=i; j<i+nzoWlen && j<d_numbits; j++)
		{
			rns_ctx->CLNW[w] += (dbits[j]<< (j-i));
			rns_ctx->CLNW_len[w]++;
		}

		if (rns_ctx->CLNW[w] > maxwin)
			maxwin = rns_ctx->CLNW[w];

		w++;
		i = j;
	}

	rns_ctx->CLNW_num = w;
	rns_ctx->CLNW_maxwin = maxwin;

#if 0
	printf("%d: ", w);
	for (int i = 1; i < w; i++)
		printf("%d(%d) ", rns_ctx->CLNW_len[i], rns_ctx->CLNW[i]);
	printf("\n");

	int cnt0 = (rns_ctx->CLNW_maxwin - 1) >> 1;
	int cnt1 = 0;
	int cnt2 = 0;
	for (int i = w - 2; i >= 0; i--) {
		cnt1 += rns_ctx->CLNW_len[i];
		if (rns_ctx->CLNW[i])
			cnt2 += 1;
	}
	printf("total # of modular multiplication = %d (%d + %d + %d)\n", 
			cnt0 + cnt1 + cnt2, cnt0, cnt1, cnt2);
#endif

	/* verify */
	ostringstream out0;
	ostringstream out1;

	for(int i=d_numbits-1; i>=0; i--)
		out0 << (BN_is_bit_set(d, i) ? 1 : 0);

	for(int i=rns_ctx->CLNW_num-1; i>=0; i--)
	{
		for(int j=rns_ctx->CLNW_len[i]-1; j>=0; j--)
			out1 << ( (rns_ctx->CLNW[i] >> j) & 1 );
	}

	assert( out0.str().compare(out1.str()) == 0 );
}


/* Partition VLNW windows */
#if 0
void partVLNWin(BIGNUM *e, RNS_CTX *rns_ctx)
{
	int i, w, maxwin;
	int d, q;

	int numbits = BN_num_bits(e);

	/* convert bits to integer array, this can make partition easier. */
	int bits[numbits];
	for(int i=0; i<numbits; i++)
		bits[i] = BN_is_bit_set(e, i);

	/* determine d, q, and l according to "High-Speed RSA Implementation" */
	q = 2;
	if (numbits < 512)
	{ d = 4; }
	else if (numbits < 1024)
	{ d = 5; }
	else
	{ d = 6; }

	/* Start to do partition */
	i = 0, w = 0, maxwin = 0;
	while(i<numbits)
	{
		/* skip next 0s */
		if (bits[i] == 0)
		{
			rns_ctx->VLNW[w] = 0;

			rns_ctx->VLNW_len[w] = 1;
			while(1)
				if (bits[++i]==1)
				{
					w++;
					break;
				}
				else
					rns_ctx->VLNW_len[w]++;
		}

		/* collect next d bits */
		int j = 0, qq = 0;
		rns_ctx->VLNW[w] = 0;
		rns_ctx->VLNW_len[w] = 0;

		for(j=i; j<i+d && j<numbits; j++)
		{
			if (bits[j] == 0) qq++; else qq = 0;

			rns_ctx->VLNW[w] += (bits[j]<< (j-i));
			rns_ctx->VLNW_len[w]++;

			if (qq == q) break;
		}

		if (rns_ctx->VLNW[w] > maxwin)
			maxwin = rns_ctx->VLNW[w];

		if (qq == q)
		{
			rns_ctx->VLNW_len[w] -= q;
			w++;
			i = j-q+1;
		} 
		else if (qq > 0)
		{
			rns_ctx->VLNW_len[w] -= qq;
			w++;
			i = j - qq;
		}
		else
		{
			w++;
			i = j;
		}
	}

	rns_ctx->VLNW_num = w;
	rns_ctx->VLNW_maxwin = maxwin;

	/* verify */
	ostringstream out0;
	ostringstream out1;

	for(int i=numbits-1; i>=0; i--)
		out0 << (BN_is_bit_set(e, i) ? 1 : 0);

	for(int i=rns_ctx->VLNW_num-1; i>=0; i--)
	{
		for(int j=rns_ctx->VLNW_len[i]-1; j>=0; j--) {
			out1 << ( (rns_ctx->VLNW[i] >> j) & 1 );
		}
	}

	assert( out0.str().compare(out1.str()) == 0 );
}
#endif
