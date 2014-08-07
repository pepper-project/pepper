#include <gmp.h>

#include "common/utility.h"
#include "powm_cache.h"

//#ifdef ON_TACC
#include <cassert>
//#endif

static size_t
get_num_chunks(size_t max_exp_bits, size_t window_size) {
  if (window_size == 0)
    return 0;

  return (max_exp_bits + window_size - 1) / window_size;
}

static size_t
get_num_pow_per_chunk(size_t window_size) {
  return (1 << window_size) - 1;
}

static size_t
get_cache_size(size_t num_chunks, size_t window_size) {
  return num_chunks * get_num_pow_per_chunk(window_size) + 1;
}

static void
free_cache(mpz_t *cache, int size) {
  for (int i = 0; i < size; i++)
    mpz_clear(cache[i]);
  free(cache);
}

PowmCache::
PowmCache(const mpz_t b, const mpz_t p, size_t max_exp_bits, size_t win_size)
  : windowSize(win_size), numChunks(0), cacheSize(0) {

  mpz_init(prime);
  mpz_set(prime, p);

  numChunks = get_num_chunks(max_exp_bits, windowSize);
  cacheSize = get_cache_size(numChunks, windowSize);
  alloc_init_vec(&cache, cacheSize);
  mpz_set(cache[0], b);

  computeCache(0);
}

PowmCache::
~PowmCache() {
  mpz_clear(prime);
  free_cache(cache, cacheSize);
}

void PowmCache::
computeCache(int start_chunk) {
  const int num_pow_per_chunk = get_num_pow_per_chunk(windowSize);
  for (size_t i = start_chunk; i < numChunks; i++) {
    mpz_t *cache_base = &cache[i * num_pow_per_chunk];

    for (int j = 1; j < num_pow_per_chunk + 1; j++) {
      mpz_mul(cache_base[j], cache_base[0], cache_base[j - 1]);
      mpz_mod(cache_base[j], cache_base[j], prime);
    }
  }
}

void PowmCache::
extendCache(size_t max_exp_bits) {
  const int num_pow_per_chunk = get_num_pow_per_chunk(windowSize);

  if (num_pow_per_chunk == 0)
    return;

  size_t new_num_chunks = get_num_chunks(max_exp_bits, windowSize);
  if (new_num_chunks <= numChunks)
    return;

  // Initialize new state.
  int new_cache_size = get_cache_size(new_num_chunks, windowSize);
  mpz_t *new_cache;
  alloc_init_vec(&new_cache, new_cache_size);

  // Copy over old cache.
  for (size_t i = 0; i < cacheSize; i++)
    mpz_set(new_cache[i], cache[i]);

  // Save new state.
  int start_chunk = numChunks;
  free_cache(cache, cacheSize);
  cache = new_cache;
  cacheSize = new_cache_size;
  numChunks = new_num_chunks;

  // Compute extra cache.
  computeCache(start_chunk);
}

mpz_t* PowmCache::
getItem(int chunk, int chunk_value) const {
  size_t idx = chunk * get_num_pow_per_chunk(windowSize) + chunk_value;
  assert(idx < cacheSize);
  return &cache[idx];
}

