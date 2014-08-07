#include "crypto/powm_cache.h"
#include "test_utils.h"

#define NBITS 1024
#define WINDOW_SIZE 4

static size_t
get_num_chunks(size_t max_exp_bits, size_t window_size) {
  if (window_size == 0)
    return 0;

  return (max_exp_bits + window_size - 1) / window_size;
}

static size_t
get_num_pow_per_chunk(size_t window_size) { return (1 << window_size) - 1; }

static mpz_t*
compute_powers(const mpz_t base, const mpz_t p, size_t max_exp_bits, size_t window_size) {
  mpz_t exp, exp_base;
  mpz_init(exp);
  mpz_init(exp_base);

  int num_pow_per_chunk = get_num_pow_per_chunk(window_size);
  int num_chunks = get_num_chunks(max_exp_bits, window_size);
  int size = num_chunks * num_pow_per_chunk + 1;
  mpz_t *powers = (mpz_t *)malloc(size * sizeof(mpz_t));
  init(powers, size, 0);

  mpz_set_ui(exp, 0);
  mpz_setbit(exp, 0);

  for (int i = 0; i < num_chunks; i++) {
    mpz_set(exp_base, exp);

    for (int j = 0; j < num_pow_per_chunk; j++) {
      mpz_powm(powers[i * num_pow_per_chunk + j], base, exp, p);
      mpz_add(exp, exp, exp_base);
    }
  }

  mpz_powm(powers[size - 1], base, exp, p);

  mpz_clear(exp);
  mpz_clear(exp_base);

  return powers;
}

int main() {
  int nbits = NBITS;
  int win_size = WINDOW_SIZE;
  mpz_t base, prime;
  
  init(base, nbits);
  init(prime, nbits);

  mpz_setbit(base, nbits - 1);
  mpz_nextprime(prime, base);
  rand(base, prime);

  START_WORK("Computing powers");
  PowmCache c(base, prime, nbits, win_size);
  PowmCache c_ext(base, prime, nbits / 3, win_size);
  mpz_t *powers = compute_powers(base, prime, nbits, win_size);
  DONE();

  START_TEST("PowmCache init");
  RUN_TEST(cmp(powers, c.getCache(), c.size()) &&
           cmp(powers, c_ext.getCache(), c_ext.size()));

  START_TEST("PowmCache extend");
  c_ext.extendCache(nbits);
  RUN_TEST(c.size() == c_ext.size() && cmp(c.getCache(), c_ext.getCache(), c.size()));
}
