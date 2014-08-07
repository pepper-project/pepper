#ifndef MP_MODEXP_GPU
#define MP_MODEXP_GPU

void mp_modexp_crt(WORD *a,
		int cnt, int S,
		WORD *ret_d, WORD *ar_d,
		struct mp_sw *sw_d,
		WORD *n_d, WORD *np_d, WORD *r_sqr_d,
		cudaStream_t stream,
		unsigned int stream_id,
		uint8_t *checkbits = 0);

int mp_modexp_crt_sync(WORD *ret, WORD *ret_d,
		WORD *n_d, WORD *np_d, WORD *r_sqr_d, WORD *iqmp_d,
		int cnt, int S,
		bool block, cudaStream_t stream,
		uint8_t *checkbits = 0);

int mp_modexp_crt_post_kernel(WORD *ret, WORD *ret_d, WORD *n_d, WORD *np_d, WORD *r_sqr_d, WORD *iqmp_d,
			      int cnt, int S,
			      bool block, cudaStream_t stream,
			      uint8_t *checkbits = 0);

void mp_modexp_mont_gpu(WORD *ret, const WORD *a, const WORD *e, 
		const WORD *r_sq, const WORD *n, const WORD *np, int S);

#endif
