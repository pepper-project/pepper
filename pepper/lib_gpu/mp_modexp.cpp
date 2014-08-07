#include <cassert>
#include <cstring>

#include <sys/time.h>

#include "mp_modexp.h"

void mp_print(const char *name, const WORD *a, int word_len)
{
	printf("%-8s: ", name);
	for (int i = word_len - 1; i >= 0; i--)
#if MP_USE_64BIT
		printf("%016lX ", a[i]);
#else
		printf("%08X ", a[i]);
#endif

	printf("\n");
}

void mp_bin2mp(WORD *a, int word_len)
{
	for (int i = 0; i < word_len; i++)
#if MP_USE_64BIT
		a[i] =
			((a[i] >> 56) & 0x00000000000000ffUL) |
			((a[i] >> 40) & 0x000000000000ff00UL) |
			((a[i] >> 24) & 0x0000000000ff0000UL) |
			((a[i] >>  8) & 0x00000000ff000000UL) |
			((a[i] <<  8) & 0x000000ff00000000UL) |
			((a[i] << 24) & 0x0000ff0000000000UL) |
			((a[i] << 40) & 0x00ff000000000000UL) |
			((a[i] << 56) & 0xff00000000000000UL);
#else
		a[i] = (a[i] >> 24) | (a[i] << 24) | ((a[i] << 8) & 0xff0000) | ((a[i] >> 8) & 0xff00);
#endif
	for (int i = 0; i < word_len / 2; i++) {
		WORD t = a[i];
		a[i] = a[word_len - i - 1];
		a[word_len - i - 1] = t;
	}
}

void mp_bn2mp(WORD *a, const BIGNUM *bn, int word_len)
{
	assert(word_len * (int)sizeof(WORD) >= BN_num_bytes(bn));

	if (sizeof(WORD) == sizeof(BN_ULONG)) {
		memcpy(a, bn->d, word_len * sizeof(WORD));
	} else {
		memset(a, 0, sizeof(WORD) * word_len);
		BN_bn2bin(bn, (unsigned char *)a + (word_len * sizeof(WORD) - BN_num_bytes(bn)));

		mp_bin2mp(a, word_len);
	}
}

void mp_mp2bn(BIGNUM *bn, const WORD *a, int word_len)
{
	BN_zero(bn);

	if (sizeof(WORD) == sizeof(BN_ULONG)) {
		BN_set_bit(bn, word_len * sizeof(WORD) * 8 - 1);
		memcpy(bn->d, a, word_len * sizeof(WORD));
	} else {
		for (int i = word_len - 1; i >= 0; i--) {
			BN_lshift(bn, bn, BITS_PER_WORD);
			BN_add_word(bn, (BN_ULONG)a[i]);
		}
	}
}

void mp_copy(WORD *dst, const WORD *org, int word_len)
{
	for (int i = word_len - 1; i >= 0; i--)
		dst[i] = org[i];
}

static int get_bit(const WORD *a, int i)
{
	return (a[i / BITS_PER_WORD] >> (i % BITS_PER_WORD)) & 1;
}

#if MP_USE_CLNW
static void mp_get_clnw(struct mp_sw *ret, const WORD *a, int word_len)
{
	const int num_bits = word_len * BITS_PER_WORD;
	int d;

	int n = 0;
	int i = 0;

	if (num_bits < 256)
		d = 4;
	else if (num_bits < 768)
		d = 5;
	else if (num_bits < 1792)
		d = 6;
	else
		d = 7;

	memset(ret, 0, sizeof(*ret));

	while (i < num_bits) {
		if (get_bit(a, i) == 0) {
			ret->fragment[n] = 0;
			while (i < num_bits && get_bit(a, i) == 0) {
				i++;
				ret->length[n]++;
				assert(ret->length[n] > 0);
			}

			n++;
			assert(n <= MP_SW_MAX_NUM_FRAGS);
		}

		int j;

		for (j = i; j < i + d && j < num_bits; j++) {
			ret->fragment[n] |= (get_bit(a, j) << (j - i));
			ret->length[n]++;
			assert(ret->length[n] > 0);
		}

		if (ret->fragment[n] > ret->max_fragment)
			ret->max_fragment = ret->fragment[n];
		i = j;
		n++;
		assert(n <= MP_SW_MAX_NUM_FRAGS);
	}

	while (n > 0 && ret->fragment[n - 1] == 0)
		n--;

	assert(n >= 0);
	assert(ret->max_fragment < MP_SW_MAX_FRAGMENT);
	assert((ret->max_fragment % 2 == 1) || (n == 0));
	ret->num_fragments = n;

        ret->length[n-1] = num_bits;
        for (int i = 0; i < n - 1; i++)
          ret->length[n-1] -= ret->length[i];
#if 0
	printf("CLNW: d = %d\n", d);
#endif
}
#endif

#if MP_USE_VLNW
static void mp_get_vlnw(struct mp_sw *ret, const WORD *a, int word_len)
{
	const int num_bits = word_len * BITS_PER_WORD;
	int d;
	int q;
	int r;
	int l = 0;

	int n = 0;
	int i = 0;

	if (num_bits < 512)
		d = 4;
	else if (num_bits < 1024)
		d = 5;
	else
		d = 6;

	if (512 <= num_bits && num_bits < 1024)
		q = 3;
	else
		q = 2;

	r = d - 1;
	while (r > q) {
		l++;
		r -= q;
	}

	memset(ret, 0, sizeof(*ret));

	while (i < num_bits) {
		if (get_bit(a, i) == 0) {
			ret->fragment[n] = 0;
			while (i < num_bits && get_bit(a, i) == 0) {
				i++;
				ret->length[n]++;
				assert(ret->length[n] > 0);
			}

			n++;
			assert(n <= MP_SW_MAX_NUM_FRAGS);
		}

		if (i >= num_bits)
			break;

		if (get_bit(a, i) == 0)
			continue;

		ret->fragment[n] = 1;
		ret->length[n] = 1;

		int j = i + 1;

		while (j < i + d && j < num_bits) {
			bool allzero = true;
			int k;

			if (j - i + 1 > r) {
				// check incoming q bits
				for (k = j; k < j + q && k < i + d; k++) {
					if (get_bit(a, k)) {
						ret->fragment[n] |= (1 << (k - i));
						allzero = false;
					}
				}
			} else {
				// last step: check incoming r bits
				for (k = j; k < j + r && k < i + d; k++) {
					if (get_bit(a, k)) {
						ret->fragment[n] |= (1 << (k - i));
						allzero = false;
					}
				}
			}

			if (allzero)
				break;

			ret->length[n] += (k - j);
			j = k;
		}

		if (ret->fragment[n] > ret->max_fragment)
			ret->max_fragment = ret->fragment[n];
		i = j;
		n++;
		assert(n <= MP_SW_MAX_NUM_FRAGS);
	}

	while (n > 0 && ret->fragment[n - 1] == 0)
		n--;

	assert(n > 0);
	assert(ret->max_fragment < MP_SW_MAX_FRAGMENT);
	assert(ret->max_fragment % 2 == 1);
	ret->num_fragments = n;

        ret->length[n-1] = num_bits;
        for (int i = 0; i < n - 1; i++)
          ret->length[n-1] -= ret->length[i];
#if 0
	printf("VLNW: d = %d, l = %d, q = %d, r = %d\n", d, l, q, r);
#endif
}
#endif

void mp_get_sw(struct mp_sw *ret, const WORD *a, int word_len)
{
#if MP_USE_CLNW
	mp_get_clnw(ret, a, word_len);
#elif MP_USE_VLNW
	mp_get_vlnw(ret, a, word_len);
#endif

#if 0
	int total_len= 0;
	int cnt_nz = 0;

	printf("i\tL\tF\n");
	for (int i = ret->num_fragments - 1; i >= 0; i--) {
		printf("%d\t%d\t%d\n", i, ret->length[i], ret->fragment[i]);

		total_len += ret->length[i];
		if (ret->fragment[i])
			cnt_nz++;
	}

	printf("n = %d\n", ret->num_fragments);
	printf("max_fragment = %d\n", ret->max_fragment);
	printf("# of nonzero windows = %d\n", cnt_nz);
	printf("# of multiplication operations = %d\n",
			(ret->max_fragment + 1) / 2 + total_len + cnt_nz);
#endif
}

#if 0
static void test_mp_mul_cpu(const BIGNUM *A, const BIGNUM *B)
{
	WORD a[MAX_S];
	WORD b[MAX_S];
	WORD x[MAX_S * 2];

	mp_bn2mp(a, A);
	mp_bn2mp(b, B);
	mp_mul_cpu(x, a, b);

	BIGNUM *X = BN_new();
	mp_mp2bn(X, x, MAX_S * 2);

	BIGNUM *Y = BN_new();
	BN_CTX *ctx = BN_CTX_new();
	BN_mul(Y, A, B, ctx);

	if (BN_cmp(X, Y) != 0) {
		printf("correct: "); BN_print_fp(stdout, Y); printf("\n");
		printf("wrong:   "); BN_print_fp(stdout, X); printf("\n");
		assert(false);
	}

	BN_free(X);
	BN_free(Y);
	BN_CTX_free(ctx);
}

static void test_mp_add_cpu(const BIGNUM *A, const BIGNUM *B)
{
	WORD a[MAX_S];
	WORD b[MAX_S];
	WORD x[MAX_S];

	mp_bn2mp(a, A);
	mp_bn2mp(b, B);
	int c = mp_add_cpu(x, a, b);

	BIGNUM *X = BN_new();
	mp_mp2bn(X, x);
	if (c)
		BN_set_bit(X, MAX_S * sizeof(WORD) * 8);

	BIGNUM *Y = BN_new();
	BN_add(Y, A, B);

	if (BN_cmp(X, Y) != 0) {
		printf("correct: "); BN_print_fp(stdout, Y); printf("\n");
		printf("wrong:   "); BN_print_fp(stdout, X); printf("\n");
		assert(false);
	}

	BN_free(X);
	BN_free(Y);
}

static void test_mp_add1_cpu(const BIGNUM *A)
{
	WORD a[MAX_S];
	WORD x[MAX_S];

	mp_bn2mp(a, A);
	int c = mp_add1_cpu(x, a);

	BIGNUM *X = BN_new();
	mp_mp2bn(X, x);
	if (c)
		BN_set_bit(X, MAX_S * sizeof(WORD) * 8);

	BIGNUM *Y = BN_new();
	BN_copy(Y, A);
	BN_add_word(Y, 1);

	if (BN_cmp(X, Y) != 0) {
		printf("correct: "); BN_print_fp(stdout, Y); printf("\n");
		printf("wrong:   "); BN_print_fp(stdout, X); printf("\n");
		assert(false);
	}

	BN_free(X);
	BN_free(Y);
}

static void test_mp_sub_cpu(const BIGNUM *A, const BIGNUM *B)
{
	WORD a[MAX_S];
	WORD b[MAX_S];
	WORD x[MAX_S];

	mp_bn2mp(a, A);
	mp_bn2mp(b, B);
	int borrow = mp_sub_cpu(x, a, b);

	BIGNUM *X = BN_new();
	mp_mp2bn(X, x);

	BIGNUM *Y = BN_new();
	BN_copy(Y, A);
	if (borrow)
		BN_set_bit(Y, MAX_S * sizeof(WORD) * 8);
	BN_sub(Y, Y, B);

	if (BN_cmp(X, Y) != 0) {
		printf("correct: "); BN_print_fp(stdout, Y); printf("\n");
		printf("wrong:   "); BN_print_fp(stdout, X); printf("\n");
		assert(false);
	}

	BN_free(X);
	BN_free(Y);
}

static void test_mp_montmul_cpu(const BIGNUM *A, const BIGNUM *B, const BIGNUM *N)
{
	WORD a[MAX_S];
	WORD b[MAX_S];
	WORD n[MAX_S];
	WORD np[MAX_S];
	WORD x[MAX_S];

	BN_CTX *ctx = BN_CTX_new();
	BIGNUM *NP = BN_new();
	BIGNUM *R = BN_new();
	BIGNUM *R_INV = BN_new();

	BN_set_bit(R, MAX_S * sizeof(WORD) * 8);
	BN_mod_inverse(R_INV, R, N, ctx);
	BN_mul(NP, R, R_INV, ctx);
	BN_sub_word(NP, 1);
	BN_div(NP, NULL, NP, N, ctx);

	mp_bn2mp(a, A);
	mp_bn2mp(b, B);
	mp_bn2mp(n, N);
	mp_bn2mp(np, NP);
	mp_montmul_cpu(x, a, b, n, np);

	BIGNUM *X = BN_new();
	mp_mp2bn(X, x);

	BIGNUM *Y = BN_new();
	BIGNUM *U = BN_new();
	BIGNUM *M = BN_new();
	BIGNUM *T = BN_new();
	BN_mul(T, A, B, ctx);
	BN_mod_mul(M, T, NP, R, ctx);
	BN_mul(U, M, N, ctx);
	BN_add(U, U, T);
	BN_div(U, NULL, U, R, ctx);
	if (BN_cmp(U, N) >= 0)
		BN_sub(Y, U, N);
	else
		BN_copy(Y, U);

	if (BN_cmp(X, Y) != 0) {
		printf("correct: "); BN_print_fp(stdout, Y); printf("\n");
		printf("wrong:   "); BN_print_fp(stdout, X); printf("\n");
		assert(false);
	}

	BN_CTX_free(ctx);
	BN_free(NP);
	BN_free(R);
	BN_free(R_INV);
	BN_free(X);
	BN_free(Y);
	BN_free(U);
	BN_free(M);
	BN_free(T);
}

static void test_mp_modmul_cpu(const BIGNUM *A, const BIGNUM *B, const BIGNUM *N)
{
	WORD ar[MAX_S];
	WORD b[MAX_S];
	WORD n[MAX_S];
	WORD np[MAX_S];
	WORD x[MAX_S];

	BN_CTX *ctx = BN_CTX_new();
	BIGNUM *NP = BN_new();
	BIGNUM *R = BN_new();
	BIGNUM *R_INV = BN_new();

	BN_set_bit(R, MAX_S * sizeof(WORD) * 8);
	BN_mod_inverse(R_INV, R, N, ctx);
	BN_mul(NP, R, R_INV, ctx);
	BN_sub_word(NP, 1);
	BN_div(NP, NULL, NP, N, ctx);

	BIGNUM *AR = BN_new();
	BN_mod_mul(AR, A, R, N, ctx);

	mp_bn2mp(ar, AR);
	mp_bn2mp(b, B);
	mp_bn2mp(n, N);
	mp_bn2mp(np, NP);
	mp_montmul_cpu(x, ar, b, n, np);

	BIGNUM *X = BN_new();
	mp_mp2bn(X, x);

	BIGNUM *Y = BN_new();
	BN_mod_mul(Y, A, B, N, ctx);

	if (BN_cmp(X, Y) != 0) {
		printf("correct: "); BN_print_fp(stdout, Y); printf("\n");
		printf("wrong:   "); BN_print_fp(stdout, X); printf("\n");
		assert(false);
	}

	BN_CTX_free(ctx);
	BN_free(NP);
	BN_free(R);
	BN_free(R_INV);
	BN_free(AR);
	BN_free(X);
	BN_free(Y);
}

static void test_mp_modexp_cpu(const BIGNUM *A, const BIGNUM *E, const BIGNUM *N)
{
	WORD ar[MAX_S];
	WORD e[MAX_S];
	WORD n[MAX_S];
	WORD np[MAX_S];
	WORD x[MAX_S];

	BN_CTX *ctx = BN_CTX_new();
	BIGNUM *NP = BN_new();
	BIGNUM *R = BN_new();
	BIGNUM *R_SQR = BN_new();
	BIGNUM *R_INV = BN_new();

	BN_set_bit(R, MAX_S * sizeof(WORD) * 8);
	BN_sqr(R_SQR, R, ctx);
	BN_mod_inverse(R_INV, R, N, ctx);
	BN_mul(NP, R, R_INV, ctx);
	BN_sub_word(NP, 1);
	BN_div(NP, NULL, NP, N, ctx);

	BIGNUM *AR = BN_new();
	BN_mod_mul(AR, A, R, N, ctx);

	mp_bn2mp(ar, AR);
	mp_bn2mp(e, E);
	mp_bn2mp(n, N);
	mp_bn2mp(np, NP);

	mp_modexp_cpu(x, ar, e, n, np);

	BIGNUM *X = BN_new();
	mp_mp2bn(X, x);
	BN_mod_mul(X, X, R_INV, N, ctx);

	BIGNUM *Y = BN_new();
	BN_mod_exp(Y, A, E, N, ctx);

	if (BN_cmp(X, Y) != 0) {
		printf("correct: "); BN_print_fp(stdout, Y); printf("\n");
		printf("wrong:   "); BN_print_fp(stdout, X); printf("\n");
		assert(false);
	}

	BN_CTX_free(ctx);
	BN_free(NP);
	BN_free(R);
	BN_free(R_SQR);
	BN_free(R_INV);
	BN_free(AR);
	BN_free(X);
	BN_free(Y);
}

void mp_test_cpu()
{
	BIGNUM *a = BN_new();
	BIGNUM *b = BN_new();
	BIGNUM *n = BN_new();

	BN_rand(a, MAX_S * BITS_PER_WORD, -1, 0);
	BN_rand(b, MAX_S * BITS_PER_WORD, -1, 0);
	BN_rand(n, MAX_S * BITS_PER_WORD, 1, 1);
	test_mp_mul_cpu(a, b);
	test_mp_add_cpu(a, b);
	test_mp_add1_cpu(a);
	test_mp_sub_cpu(a, b);
	test_mp_montmul_cpu(a, b, n);
	test_mp_modmul_cpu(a, b, n);
	test_mp_modexp_cpu(a, b, n);

	BN_free(a);
	BN_free(b);
	BN_free(n);
}

static void test_mp_mul_gpu(const BIGNUM *X, const BIGNUM *Y)
{
	WORD x[MAX_S];
	WORD y[MAX_S];
	WORD ret_gpu[MAX_S * 2];
	WORD ret_cpu[MAX_S * 2];

	mp_bn2mp(x, X);
	mp_bn2mp(y, Y);

	mp_mul_gpu(ret_gpu, x, y);
	mp_mul_cpu(ret_cpu, x, y);

	for (int i = 0; i < MAX_S * 2; i++) {
		if (ret_cpu[i] != ret_gpu[i]) {
			mp_print("x", x);
			mp_print("y", y);
			mp_print("GPU", ret_gpu, MAX_S * 2);
			mp_print("CPU", ret_cpu, MAX_S * 2);
			assert(false);
		}
	}
}

static void test_mp_add_gpu(const BIGNUM *X, const BIGNUM *Y)
{
	WORD x[MAX_S];
	WORD y[MAX_S];
	WORD ret_gpu[MAX_S];
	WORD ret_cpu[MAX_S];

	mp_bn2mp(x, X);
	mp_bn2mp(y, Y);

	mp_add_gpu(ret_gpu, x, y);
	mp_add_cpu(ret_cpu, x, y);

	for (int i = 0; i < MAX_S; i++) {
		if (ret_cpu[i] != ret_gpu[i]) {
			mp_print("x", x);
			mp_print("y", y);
			mp_print("GPU", ret_gpu);
			mp_print("CPU", ret_cpu);
			assert(false);
		}
	}
}

static void test_mp_add1_gpu(const BIGNUM *X)
{
	WORD x[MAX_S];
	WORD ret_gpu[MAX_S];
	WORD ret_cpu[MAX_S];

	mp_bn2mp(x, X);

	mp_add1_gpu(ret_gpu, x);
	mp_add1_cpu(ret_cpu, x);

	for (int i = 0; i < MAX_S; i++) {
		if (ret_cpu[i] != ret_gpu[i]) {
			mp_print("x", x);
			mp_print("GPU", ret_gpu);
			mp_print("CPU", ret_cpu);
			assert(false);
		}
	}
}

static void test_mp_sub_gpu(const BIGNUM *X, const BIGNUM *Y)
{
	WORD x[MAX_S];
	WORD y[MAX_S];
	WORD ret_gpu[MAX_S];
	WORD ret_cpu[MAX_S];

	mp_bn2mp(x, X);
	mp_bn2mp(y, Y);

	mp_sub_gpu(ret_gpu, x, y);
	mp_sub_cpu(ret_cpu, x, y);

	for (int i = 0; i < MAX_S; i++) {
		if (ret_cpu[i] != ret_gpu[i]) {
			mp_print("x", x);
			mp_print("y", y);
			mp_print("GPU", ret_gpu);
			mp_print("CPU", ret_cpu);
			assert(false);
		}
	}
}

static void test_mp_montmul_gpu(const BIGNUM *A, const BIGNUM *B, const BIGNUM *N)
{
	WORD a[MAX_S];
	WORD b[MAX_S];
	WORD n[MAX_S];
	WORD np[MAX_S];
	WORD ret_gpu[MAX_S];
	WORD ret_cpu[MAX_S];

	BN_CTX *ctx = BN_CTX_new();
	BIGNUM *NP = BN_new();
	BIGNUM *R = BN_new();
	BIGNUM *R_INV = BN_new();

	BN_set_bit(R, MAX_S * sizeof(WORD) * 8);
	BN_mod_inverse(R_INV, R, N, ctx);
	BN_mul(NP, R, R_INV, ctx);
	BN_sub_word(NP, 1);
	BN_div(NP, NULL, NP, N, ctx);

	mp_bn2mp(a, A);
	mp_bn2mp(b, B);
	mp_bn2mp(n, N);
	mp_bn2mp(np, NP);

	BN_CTX_free(ctx);
	BN_free(NP);
	BN_free(R);
	BN_free(R_INV);

	mp_montmul_gpu(ret_gpu, a, b, n, np);
	mp_montmul_cpu(ret_cpu, a, b, n, np);

	for (int i = 0; i < MAX_S; i++) {
		if (ret_cpu[i] != ret_gpu[i]) {
			mp_print("a", a);
			mp_print("b", b);
			mp_print("n", n);
			mp_print("np", np);
			mp_print("GPU", ret_gpu);
			mp_print("CPU", ret_cpu);
			assert(false);
		}
	}
}

static void test_mp_modmul_gpu(const BIGNUM *A, const BIGNUM *B, const BIGNUM *N)
{
	WORD ar[MAX_S];
	WORD b[MAX_S];
	WORD n[MAX_S];
	WORD np[MAX_S];
	WORD ret_gpu[MAX_S];
	WORD ret_cpu[MAX_S];

	BN_CTX *ctx = BN_CTX_new();
	BIGNUM *NP = BN_new();
	BIGNUM *R = BN_new();
	BIGNUM *R_INV = BN_new();

	BN_set_bit(R, MAX_S * sizeof(WORD) * 8);
	BN_mod_inverse(R_INV, R, N, ctx);
	BN_mul(NP, R, R_INV, ctx);
	BN_sub_word(NP, 1);
	BN_div(NP, NULL, NP, N, ctx);

	BIGNUM *AR = BN_new();
	BN_mod_mul(AR, A, R, N, ctx);

	mp_bn2mp(ar, AR);
	mp_bn2mp(b, B);
	mp_bn2mp(n, N);
	mp_bn2mp(np, NP);

	BN_CTX_free(ctx);
	BN_free(NP);
	BN_free(R);
	BN_free(R_INV);

	mp_montmul_gpu(ret_gpu, ar, b, n, np);
	mp_montmul_cpu(ret_cpu, ar, b, n, np);

	for (int i = 0; i < MAX_S; i++) {
		if (ret_cpu[i] != ret_gpu[i]) {
			mp_print("ar", ar);
			mp_print("b", b);
			mp_print("n", n);
			mp_print("np", np);
			mp_print("GPU", ret_gpu);
			mp_print("CPU", ret_cpu);
			assert(false);
		}
	}
}

static void test_mp_modexp_gpu(const BIGNUM *A, const BIGNUM *E, const BIGNUM *N)
{
	WORD ar[MAX_S];
	WORD e[MAX_S];
	WORD n[MAX_S];
	WORD np[MAX_S];
	WORD ret_gpu[MAX_S];
	WORD ret_cpu[MAX_S];

	BN_CTX *ctx = BN_CTX_new();
	BIGNUM *NP = BN_new();
	BIGNUM *R = BN_new();
	BIGNUM *R_INV = BN_new();

	BN_set_bit(R, MAX_S * sizeof(WORD) * 8);
	BN_mod_inverse(R_INV, R, N, ctx);
	BN_mul(NP, R, R_INV, ctx);
	BN_sub_word(NP, 1);
	BN_div(NP, NULL, NP, N, ctx);

	BIGNUM *AR = BN_new();
	BN_mod_mul(AR, A, R, N, ctx);

	mp_bn2mp(ar, AR);
	mp_bn2mp(e, E);
	mp_bn2mp(n, N);
	mp_bn2mp(np, NP);

	BN_CTX_free(ctx);
	BN_free(NP);
	BN_free(R);
	BN_free(R_INV);

	mp_modexp_gpu(ret_gpu, ar, e, n, np);
	mp_modexp_cpu(ret_cpu, ar, e, n, np);

	for (int i = 0; i < MAX_S; i++) {
		if (ret_cpu[i] != ret_gpu[i]) {
			mp_print("ar", ar);
			mp_print("e", e);
			mp_print("n", n);
			mp_print("np", np);
			mp_print("GPU", ret_gpu);
			mp_print("CPU", ret_cpu);
			assert(false);
		}
	}
}

void mp_test_gpu()
{
	BIGNUM *a = BN_new();
	BIGNUM *b = BN_new();
	BIGNUM *n = BN_new();

	BN_rand(a, MAX_S * BITS_PER_WORD, -1, 0);
	BN_rand(b, MAX_S * BITS_PER_WORD, -1, 0);
	BN_rand(n, MAX_S * BITS_PER_WORD, 1, 1);

	test_mp_mul_gpu(a, b);
	test_mp_add_gpu(a, b);
	test_mp_add1_gpu(a);
	test_mp_sub_gpu(a, b);
	test_mp_montmul_gpu(a, b, n);
	test_mp_modmul_gpu(a, b, n);
	test_mp_modexp_gpu(a, b, n);

	BN_free(a);
	BN_free(b);
	BN_free(n);
}

#endif
