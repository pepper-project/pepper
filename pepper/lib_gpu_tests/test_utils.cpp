#include <stdlib.h>
#include "test_utils.h"

static gmp_randstate_t* get_rand_state() {
  static gmp_randstate_t rand_state;
  static bool allocated = false;

  if (!allocated)
  {
    gmp_randinit_default(rand_state);
    allocated = true;
  }

  return &rand_state;
}

void init(int *array, size_t n)
{
  for (size_t i = 0; i < n; i++)
    array[i] = i;
}

void init(mpz_t a, int nbits)
{
  mpz_init2(a, nbits);
}

void init(mpz_t *array, size_t n, int nbits)
{
  for (size_t i = 0; i < n; i++)
    init(array[i], nbits);
}

void rand_init(int *array, size_t n)
{
  for (size_t i = 0; i < n; i++)
    array[i] = rand();
}

void rand_init(mpz_t *array, size_t n, int nbits)
{
  gmp_randstate_t *rand_state = get_rand_state();
  init(array, n, nbits);

  for (size_t i = 0; i < n; i++) {
    mpz_urandomb(array[i], *rand_state, nbits);
  }
}

void rand(mpz_t rop, mpz_t op) {
  mpz_urandomm(rop, *get_rand_state(), op);
}

void clear(mpz_t *array, size_t n)
{
  for (size_t i = 0; i < n; i++) {
    mpz_clear(array[i]);
  }
}

static inline void
cmp_fail(const mpz_t correct, const mpz_t actual, int i, cmp_callback_func f) {
  if (i < 0) {
    gmp_printf("FAILED!\n");
  }
  else {
    gmp_printf("FAILED! %d\n", i);
  }
    
  f(i);
  gmp_printf("correct: %Zx\n", correct);
  gmp_printf("actual : %Zx\n", actual);
}

bool
cmp(const mpz_t correct, const mpz_t actual, cmp_callback_func f) {
  if (mpz_cmp(correct, actual)) {
    cmp_fail(correct, actual, -1, f);
    return false;
  }
  else {
    return true;
  }
}

bool
cmp(const mpz_t *correct, const mpz_t *actual, int n, cmp_callback_func f)
{
  for (int i = 0; i < n; i++) {
    if (mpz_cmp(correct[i], actual[i])) {
      cmp_fail(correct[i], actual[i], i, f);
      return false;
    }
  }
  return true;
}

bool
cmp(const char *correct, const char *actual, int n, cmp_callback_func f)
{
  for (int i = 0; i < n; i++) {
    if (correct[i] != actual[i]) {
      printf("FAILED! %d\n", i);
      f(i);
      return false;
    }
  }
  return true;
}

