#ifndef _CRYPTO_POWM_CACHE_H_
#define _CRYPTO_POWM_CACHE_H_

#include <gmp.h>

class PowmCache {
    mpz_t prime;
    mpz_t *cache;
    size_t windowSize;
    size_t numChunks;
    size_t cacheSize;   // number of allocated mpz_t

    virtual void computeCache(int start_chunk);

  public:
    PowmCache(const mpz_t b, const mpz_t p, size_t max_exp_bits, size_t win_size);
    ~PowmCache();

    void extendCache(size_t max_exp_bits);

    mpz_t* getItem(int chunk, int chunk_value) const;
    mpz_t* getCache() const {
        return cache;
    }
    size_t size() const {
        return cacheSize;
    }
};

static void
montmul(mpz_t rop, const mpz_t op1, const mpz_t op2,
        const mpz_t n, const mpz_t np, int num_bits);

#endif
