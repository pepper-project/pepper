#ifndef MP_MODEXP
#define MP_MODEXP

#include <stdint.h>

#include <openssl/bn.h>

#include <cuda_runtime.h>

#define MAX_STREAMS		16
#define MP_MAX_NUM_PAIRS	1024

// Define some useful constants.
#define WINDOW_SIZE     8         // For caching powers of a base.
                                  // Should divide BITS_PER_WORD
#define WARP_SIZE       32        // Unused at the moment.
#define MAX_BLOCK_SIZE  (256)
#define MAX_GRID_SIZE   (128)
#define MAX_BLOCKY_DIM  (MAX_BLOCK_SIZE / WARP_SIZE)
#define MAX_NUM_THREADS (MAX_BLOCK_SIZE * MAX_GRID_SIZE)

// Some useful macros
#define MAX_ELEMS_PER_BLOCK(S)  (MAX_BLOCK_SIZE / (S))
#define MAX_ELEMS_PER_KERNEL(S) (MAX_NUM_THREADS / (S))

#if MP_USE_64BIT == 1

#define BITS_PER_WORD 64
typedef uint64_t WORD;

// the maximum number of WORDS in a mp number
#define S_256	4
#define S_512	8
#define S_1024	16
#define S_2048	32
#define MAX_S   S_1024

#define MP_MSGS_PER_BLOCK (16 / (S / S_256))

#elif MP_USE_64BIT == 0

#define BITS_PER_WORD 32
typedef uint32_t WORD;

// the maximum number of WORDS in a mp number
#define MAX_S 32	
#define S_256	8
#define S_512	16
#define S_1024	32
#define S_2048	64

#define MP_MSGS_PER_BLOCK (8 / (S / S_256))

#else

#error MP_USE_64BIT is not defined

#endif

#if (BITS_PER_WORD / WINDOW_SIZE) * WINDOW_SIZE != BITS_PER_WORD
#error WINDOW_SIZE must divide BITS_PER_WORD
#endif

/* CRT postprocessing offloading */
#define MP_MODEXP_OFFLOAD_POST 1

/* these two are only valid for test code */
#define MONTMUL_FAST_CPU 1
#define MONTMUL_FAST_GPU 1

#ifdef __GPU__
#define sync_if_needed()	do { if (S > 32) __syncthreads(); } while(0)
#else
#define sync_if_needed()
#endif

/* c: carry (may increment by 1)
   s: partial sum
   x, y: operands */
#define ADD_CARRY(c, s, x, y) \
		do { \
			WORD _t = (x) + (y); \
			(c) += (_t < (x)); \
			sync_if_needed(); \
			(s) = _t; \
			sync_if_needed(); \
		} while (0)

/* Same with ADD_CARRY, but sets y to 0 */
#define ADD_CARRY_CLEAR(c, s, x, y) \
		do { \
			WORD _t = (x) + (y); \
			(y) = 0; \
			sync_if_needed(); \
			(c) += (_t < (x)); \
			(s) = _t; \
			sync_if_needed(); \
		} while (0)

/* b: borrow (may increment by 1)
   d: partial difference
   x, y: operands (a - b) */
#define SUB_BORROW(b, d, x, y) \
		do { \
			WORD _t = (x) - (y); \
			(b) += (_t > (x)); \
			sync_if_needed(); \
			(d) = _t; \
			sync_if_needed(); \
		} while (0)

/* Same with SUB_BORROW, but sets y to 0 */
#define SUB_BORROW_CLEAR(b, d, x, y) \
		do { \
			WORD _t = (x) - (y); \
			(y) = 0; \
			sync_if_needed(); \
			(b) += (_t > (x)); \
			(d) = _t; \
			sync_if_needed(); \
		} while (0)

#define MP_USE_CLNW 1
#define MP_USE_VLNW 0

#if MP_USE_CLNW + MP_USE_VLNW != 1
#error Use one and only one sliding window technique
#endif

#define MP_SW_MAX_NUM_FRAGS 512
//#define MP_SW_MAX_FRAGMENT 128
// A value of 32 only allows for exponents up to 768 bits.
// Increase by powers of 7 to improve this bound. Up to a
// maximum of 128.
#define MP_SW_MAX_FRAGMENT 32
#define MP_MAX_MULTI_EXP 32

/* for sliding algorithms (both CLNW and VLNW) */
struct mp_sw {
	uint16_t fragment[MP_SW_MAX_NUM_FRAGS];
	uint16_t length[MP_SW_MAX_NUM_FRAGS];
	int num_fragments;
	int max_fragment;
};

/* for simultaneous exp sliding algorithm */
struct mp_multi_sw {
	uint16_t length[MP_SW_MAX_NUM_FRAGS * MP_MAX_MULTI_EXP];
	uint16_t fragment[MP_SW_MAX_NUM_FRAGS * MP_MAX_MULTI_EXP];
        uint16_t base[MP_SW_MAX_NUM_FRAGS * MP_MAX_MULTI_EXP];
	uint16_t max_fragment[MP_MAX_MULTI_EXP];
	int num_fragments;
        int num_bases;
};

void mp_print(const char *name, const WORD *a, int word_len = MAX_S);
void mp_bn2mp(WORD *a, const BIGNUM *bn, int word_len = MAX_S);
void mp_mp2bn(BIGNUM *bn, const WORD *a, int word_len = MAX_S);
void mp_copy(WORD *dst, const WORD *org, int word_len = MAX_S);
void mp_get_sw(struct mp_sw *ret, const WORD *a, int word_len = MAX_S);



void mp_test_cpu();
void mp_test_gpu();

/* all mp_*_cpu() and mp_*_gpu() functions are single-threaded */
void mp_mul_cpu(WORD *ret, const WORD *a, const WORD *b);
int mp_add_cpu(WORD *ret, const WORD *x, const WORD *y);
int mp_add1_cpu(WORD *ret, const WORD *x);
int mp_sub_cpu(WORD *ret, const WORD *x, const WORD *y);
void mp_montmul_cpu(WORD *ret, const WORD *a, const WORD *b,
		const WORD *n, const WORD *np);
void mp_modexp_cpu(WORD *ret, const WORD *ar, const WORD *e,
		const WORD *n, const WORD *np);

void mp_mul_gpu(WORD *ret, const WORD *x, const WORD *y);
void mp_add_gpu(WORD *ret, const WORD *x, const WORD *y);
void mp_add1_gpu(WORD *ret, const WORD *x);
void mp_sub_gpu(WORD *ret, const WORD *x, const WORD *y);
void mp_montmul_gpu(WORD *ret, const WORD *a, const WORD *b,
		const WORD *n, const WORD *np, int S);
void mp_modexp_gpu(WORD *ret, const WORD *ar, const WORD *e,
		const WORD *n, const WORD *np, int S);

void mp_many_modexp_mont_gpu_nocopy(
    int len_vector, WORD *ret, WORD *a, struct mp_sw *sw,
    const WORD *r_sq, const WORD *n, const WORD *np,
    int S);

void mp_montgomerize_gpu_nocopy(
    int len_vector, WORD *ret, WORD *a,
    const WORD *r_sq, const WORD *n, const WORD *np, int S);

void mp_modexp_cached_gpu_nocopy(
    int len_vector, WORD *ret, const WORD *a, const WORD *powm_cache,
    const WORD *r_sq, const WORD *n, const WORD *np, int S);

void mp_vec_multi_modexp(
    int vector_len, WORD *ret, WORD *base, struct mp_multi_sw *multi_sw,
    const WORD *r_sq, const WORD *n, const WORD *np, int S);

void mp_many_modexp_mont_cpu_nocopy(
    int len_vector, WORD *ret, WORD *a, struct mp_sw *sw,
    const WORD *r_sq, const WORD *n, const WORD *np,
    int S);

void mp_montgomerize_cpu_nocopy(
    int len_vector, WORD *ret, WORD *a,
    const WORD *r_sq, const WORD *n, const WORD *np, int S);

void mp_modexp_cached_cpu_nocopy(
    int len_vector, WORD *ret, const WORD *a, const WORD *powm_cache,
    const WORD *r_sq, const WORD *n, const WORD *np, int S);
#endif
