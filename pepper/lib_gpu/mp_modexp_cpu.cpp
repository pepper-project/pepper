#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "mp_modexp.h"

static WORD umulhi(WORD a, WORD b)
{
#if MP_USE_64BIT
#define UMUL_HI_ASM(a,b)   ({      \
	register BN_ULONG ret,discard;  \
	asm ("mulq      %3"             \
		      : "=a"(discard),"=d"(ret)  \
		      : "a"(a), "g"(b)           \
		      : "cc");                   \
	ret;})

	return UMUL_HI_ASM(a, b);
#else
	uint64_t t;

	t = (uint64_t)a * (uint64_t)b;
	return (WORD)(t >> BITS_PER_WORD);
#endif
}

static WORD umullo(WORD a, WORD b)
{
	return a * b;
}

template <int S>
void mp_mul_cpu(WORD *ret, const WORD *a, const WORD *b)
{
	WORD t[S * 2 + 1];
	WORD c[S * 2 + 1];	// carry

	for (int i = 0; i < S; i++) {
		c[i] = 0;
		c[i + S] = 0;
		t[i] = 0;
		t[i + S] = 0;
	}

	for (int j = 0; j < S; j++) {
		for (int i = 0; i < S; i++) {
			WORD hi = umulhi(a[i], b[j]);
			WORD lo = umullo(a[i], b[j]);
		
			ADD_CARRY(c[i + j + 2], t[i + j + 1], t[i + j + 1], hi);
			ADD_CARRY(c[i + j + 1], t[i + j], t[i + j], lo);
		}

		//printf("j = %d\n", j);
		//mp_print(c + MAX_S); mp_print(c); printf("\n");
		//mp_print(t + MAX_S); mp_print(t); printf("\n");
	}

	while (1) {
		bool all_zero = true;
		for (int j = 0; j < S; j++) {
			if (c[j])
				all_zero = false;
			if (c[j + S])
				all_zero = false;
		}
		if (all_zero)
			break;

		for (int j = 2 * S - 1; j >= 1; j--)
			ADD_CARRY_CLEAR(c[j + 1], t[j], t[j], c[j]);
	}

	for (int i = 0; i < S; i++) {
		ret[i] = t[i];
		ret[i + S] = t[i + S];
	}
}

/* returns 1 for the most significant carry. 0 otherwise */
int mp_add_cpu(WORD *ret, const WORD *x, const WORD *y)
{
	WORD c[MAX_S];	// carry. c[i] is set by a[i] and b[i]

	for (int i = 0; i < MAX_S; i++) {
		c[i] = 0;
		ADD_CARRY(c[i], ret[i], x[i], y[i]);
	}

	while (1) {
		bool all_zero = true;
		/* NOTE MAX_S - 1, not just MAX_S */ 
		for (int j = 0; j < MAX_S - 1; j++) { 
			if (c[j])
				all_zero = false;
		}

		if (all_zero)
			break;

		for (int j = MAX_S - 2; j >= 0; j--)
			ADD_CARRY_CLEAR(c[j + 1], ret[j + 1], ret[j + 1], c[j]);
	}

	return c[MAX_S - 1];
}

/* returns 1 for the most significant carry. 0 otherwise */
int mp_add1_cpu(WORD *ret, const WORD *x)
{
	WORD c[MAX_S];	// carry. c[i] is set by a[i]

	for (int i = 0; i < MAX_S; i++) {
		c[i] = 0;
		ADD_CARRY(c[i], ret[i], x[i], (i == 0) ? 1 : 0);
	}

	while (1) {
		bool all_zero = true;
		/* NOTE MAX_S - 1, not just MAX_S */ 
		for (int j = 0; j < MAX_S - 1; j++) { 
			if (c[j])
				all_zero = false;
		}

		if (all_zero)
			break;

		for (int j = MAX_S - 2; j >= 0; j--)
			ADD_CARRY_CLEAR(c[j + 1], ret[j + 1], ret[j + 1], c[j]);
	}

	return c[MAX_S - 1];
}

/* returns 1 for the most significant borrow. 0 otherwise */
template<int S>
int mp_sub_cpu(WORD *ret, const WORD *x, const WORD *y)
{
	WORD b[S]; // borrow

	for (int i = 0; i < S; i++) {
		b[i] = 0;
		SUB_BORROW(b[i], ret[i], x[i], y[i]);
	}

	while (1) {
		bool all_zero = true;
		/* NOTE S - 1, not just S */ 
		for (int j = 0; j < S - 1; j++) { 
			if (b[j])
				all_zero = false;
		}

		if (all_zero)
			break;

		for (int j = S - 2; j >= 0; j--)
			SUB_BORROW_CLEAR(b[j + 1], ret[j + 1], ret[j + 1], b[j]);
	}

	return b[S - 1];
}

#if !MONTMUL_FAST_CPU

/* assumes a and b are 'montgomeritized' */
void mp_montmul_cpu(WORD *ret, const WORD *a, const WORD *b, 
		const WORD *n, const WORD *np)
{
	WORD t[MAX_S * 2];
	WORD m[MAX_S * 2];
	WORD mn[MAX_S * 2];
	WORD u[MAX_S];
	int c = 0;

	mp_mul_cpu(t, a, b);
	mp_mul_cpu(m, t, np);
	mp_mul_cpu(mn, m, n);
	c = mp_add_cpu(u, t + MAX_S, mn + MAX_S);

	bool half_zero = true;
	for (int i = 0; i < MAX_S; i++) {
		if (t[i])
			half_zero = false;
	}
	if (!half_zero)
		c |= mp_add1_cpu(u, u);

	// c may be 0 or 1, but not 2
	if (c)	
		goto u_is_bigger;

	/* Ugly, but practical. 
	 * Can we do this much better with Fermi's ballot()? */
	for (int i = MAX_S - 1; i >= 0; i--) {
		if (u[i] > n[i])
			goto u_is_bigger;
		if (u[i] < n[i])
			goto n_is_bigger;
	}

u_is_bigger:
	mp_sub_cpu(ret, u, n);
	return;

n_is_bigger:
	for (int i = 0; i < MAX_S; i++)
		ret[i] = u[i];
	return;
}

#else

/* fast version */
template <int S>
void mp_montmul_cpu(WORD *ret, const WORD *a, const WORD *b, 
		const WORD *n, const WORD *np)
{
	WORD t[S * 2];
	WORD c[S * 2];
	WORD u[S];
	int carry = 0;
	
	for (int i = 0; i < S; i++) {
		c[i] = 0;
		c[i + S] = 0;
	}

	mp_mul_cpu<S>(t, a, b);
		
	for (int j = 0; j < S; j++) {
		WORD m = t[j] * np[0];
		for (int i = 0; i < S; i++) {
			WORD hi = umulhi(m, n[i]);
			WORD lo = umullo(m, n[i]);

			ADD_CARRY(c[i + j + 1], t[i + j + 1], t[i + j + 1], hi);
			ADD_CARRY(c[i + j], t[i + j], t[i + j], lo);
		}
//		printf("j = %d, m = %X\n", j, m);
//		mp_print(c, MAX_S * 2); printf("\n");
//		mp_print(t, MAX_S * 2); printf("\n");
		
		while (1) {
			bool all_zero = true;
			for (int j = 0; j < S; j++) {
				if (c[j])
					all_zero = false;
				if (j < S - 1 && c[j + S])
					all_zero = false;
			}
			if (all_zero)
				break;

			for (int j = 2 * S - 1; j >= 1; j--)
				ADD_CARRY_CLEAR(c[j], t[j], t[j], c[j - 1]);
		}
	}

	for (int i = 0; i < S; i++)
		u[i] = t[i + S];

	//carry = mp_add_cpu(u, t + MAX_S, mn + MAX_S);
	carry = c[2 * S - 1];

	// c may be 0 or 1, but not 2
	if (carry)	
		goto u_is_bigger;

	/* Ugly, but practical. 
	 * Can we do this much better with Fermi's ballot()? */
	for (int i = S - 1; i >= 0; i--) {
		if (u[i] > n[i])
			goto u_is_bigger;
		if (u[i] < n[i])
			goto n_is_bigger;
	}

u_is_bigger:
	mp_sub_cpu<S>(ret, u, n);
	return;

n_is_bigger:
	for (int i = 0; i < S; i++)
		ret[i] = u[i];
	return;
}

#endif

void mp_modexp_cpu_org(WORD *ret, const WORD *ar, const WORD *e, 
		const WORD *n, const WORD *np)
{
	int t = MAX_S * BITS_PER_WORD - 1;

	while (((e[t / BITS_PER_WORD] >> (t % BITS_PER_WORD)) & 1) == 0 && t > 0)
		t--;

	for (int i = 0; i < MAX_S; i++)
		ret[i] = ar[i];

	t--;

	while (t >= 0) {
		mp_montmul_cpu<MAX_S>(ret, ret, ret, n, np);
		if (((e[t / BITS_PER_WORD] >> (t % BITS_PER_WORD)) & 1) == 1)
			mp_montmul_cpu<MAX_S>(ret, ret, ar, n, np);

		t--;
	}
}

/* assumes ar is 'montgomeritized' */
template<int S> void
mp_modexp_cpu(WORD *ret, const WORD *ar, const struct mp_sw* sw, 
		const WORD *n, const WORD *np)
{
	WORD ar_sqr[S];

	/* odd powers of ar (ar, (ar)^3, (ar)^5, ... ) */
	WORD ar_pow[MP_SW_MAX_FRAGMENT / 2][S];

	for (int i = 0; i < S; i++)
		ar_pow[0][i] = ar[i];

	mp_montmul_cpu<S>(ar_sqr, ar_pow[0], ar_pow[0], n, np);

	for (int i = 3; i <= sw->max_fragment; i += 2)
		mp_montmul_cpu<S>(ar_pow[i >> 1], ar_pow[(i >> 1) - 1], ar_sqr, n, np);

	for (int i = 0; i < S; i++)
		ret[i] = ar_pow[sw->fragment[sw->num_fragments - 1] >> 1][i];

	for (int k = sw->num_fragments - 2; k >= 0; k--) {
		for (int i = 0; i < sw->length[k]; i++)
			mp_montmul_cpu<S>(ret, ret, ret, n, np);

		if (sw->fragment[k])
			mp_montmul_cpu<S>(ret, ret, ar_pow[sw->fragment[k] >> 1], n, np);
	}
}

template<int S>
void static
copy(WORD *ret, const WORD *a)
{
  for (int i = 0; i < S; i++)
    ret[i] = a[i];
}

template<int S>
void static
make_one(WORD *w)
{
  for (int i = 0; i < S; i++)
    w[i] = 0;
  w[0] = 1;
}

template<int S>
void mp_modexp_mont_cpu(
    WORD *ret, const WORD *a, const struct mp_sw *sw,
    const WORD *r_sq, const WORD *n, const WORD *np)
{
  WORD one[S];
  WORD s_a[S];

  // Account for the case when sw represents the exponent 0.
  if (sw->num_fragments == 0) {
    make_one<S>(ret);
  } else {
    mp_montmul_cpu<S>(s_a, a, r_sq, n, np);

    mp_modexp_cpu<S>(ret, s_a, sw, n, np);

    make_one<S>(one);
    mp_montmul_cpu<S>(ret, ret, one, n, np);
  }
}

template<int S>
static void
mp_montgomerize_cpu(
    WORD *ret, const WORD *a, const WORD *r_sq,
    const WORD *n, const WORD*np)
{
  WORD tmp[S];
  copy<S>(tmp, a);
  mp_montmul_cpu<S>(ret, tmp, r_sq, n, np);
}

// A macro to access the nth chunk of a word.
#define WINDOW(word, n) ((word) >> (WINDOW_SIZE * (n)) & ((1 << WINDOW_SIZE) - 1))

/**
 * assumes ar is 'montgomeritized'
 * assumes exp != 0.
 */
template<int S>
static void
mp_modexp_cached_cpu(
    WORD *ret, const WORD *exp, const WORD *powm_cache,
    const WORD* r_sq, const WORD *n, const WORD *np)
{
  WORD tmp[S];

  // Ret = mont(1)
  make_one<S>(tmp);
  mp_montmul_cpu<S>(ret, tmp, r_sq, n, np);

  for (int i = 0; i < S; i++) {
    for (int j = 0; j < BITS_PER_WORD / WINDOW_SIZE; j++) {
      if (WINDOW(exp[i], j) > 0) {
        copy<S>(tmp, &powm_cache[(WINDOW(exp[i], j) - 1) * S]);

        mp_montmul_cpu<S>(ret, ret, tmp, n, np);
      }

      powm_cache += ((1 << WINDOW_SIZE) - 1) * S;
    }
  }

  make_one<S>(tmp);
  mp_montmul_cpu<S>(ret, ret, tmp, n, np);
}

void mp_montgomerize_cpu_nocopy(
    int len_vector, WORD *ret, WORD *a,
    const WORD *r_sq, const WORD *n, const WORD *np, int S)
{
  //mp_modexp_mont_kernel<<<len_vector/8, dim3(S, 8)>>>(S, ret, a, sw, r_sq, n, np);
  //return;

  for (int i = 0; i < len_vector; i++) {
    switch (S) {
      case S_256: mp_montgomerize_cpu<S_256>(
                          &ret[i * S], &a[i * S],
                          r_sq, n, np); break;
      case S_512: mp_montgomerize_cpu<S_512>(
                          &ret[i * S], &a[i * S],
                          r_sq, n, np); break;
      case S_1024: mp_montgomerize_cpu<S_1024>(
                          &ret[i * S], &a[i * S],
                          r_sq, n, np); break;
      case S_2048: mp_montgomerize_cpu<S_2048>(
                          &ret[i * S], &a[i * S],
                          r_sq, n, np); break;
      default: assert(false);
    }
  }
}

void mp_modexp_cached_cpu_nocopy(
    int len_vector, WORD *ret, const WORD *exp, const WORD *powm_cache,
    const WORD *r_sq, const WORD *n, const WORD *np, int S)
{
  //mp_modexp_mont_kernel<<<len_vector/8, dim3(S, 8)>>>(S, ret, a, sw, r_sq, n, np);
  //return;

  for (int i = 0; i < len_vector; i++) {

    switch (S) {
      case S_256: mp_modexp_cached_cpu<S_256>(
                          &ret[i * S],
                          &exp[i * S], powm_cache,
                          r_sq, n, np); break;
      case S_512: mp_modexp_cached_cpu<S_512>(
                          &ret[i * S],
                          &exp[i * S], powm_cache,
                          r_sq, n, np); break;
      case S_1024: mp_modexp_cached_cpu<S_1024>(
                          &ret[i * S],
                          &exp[i * S], powm_cache,
                          r_sq, n, np); break;
      case S_2048: mp_modexp_cached_cpu<S_2048>(
                          &ret[i * S],
                          &exp[i * S], powm_cache,
                          r_sq, n, np); break;
      default: assert(false);
    }
  }
}


void mp_many_modexp_mont_cpu_nocopy(
    int len_vector, WORD *ret, WORD *a, struct mp_sw *sw,
    const WORD *r_sq, const WORD *n, const WORD *np, int S)
{
  //mp_modexp_mont_kernel<<<len_vector/8, dim3(S, 8)>>>(S, ret, a, sw, r_sq, n, np);
  //return;

  for (int i = 0; i < len_vector; i++) {
    switch (S) {
      case S_256: mp_modexp_mont_cpu<S_256>(
                          &ret[i * S],
                          &a[i * S], &sw[i],
                          r_sq, n, np); break;
      case S_512: mp_modexp_mont_cpu<S_512>(
                          &ret[i * S],
                          &a[i * S], &sw[i],
                          r_sq, n, np); break;
      case S_1024: mp_modexp_mont_cpu<S_1024>(
                          &ret[i * S],
                          &a[i * S], &sw[i],
                          r_sq, n, np); break;
      case S_2048: mp_modexp_mont_cpu<S_2048>(
                          &ret[i * S],
                          &a[i * S], &sw[i],
                          r_sq, n, np); break;
      default: assert(false);
    }
  }
}


