#ifndef CODE_PEPPER_CRYPTO_CPU_CRYPTO_H_
#define CODE_PEPPER_CRYPTO_CPU_CRYPTO_H_

#include "crypto.h"
#include "lib_gpu/mp_modexp.h"
#include <lib_gpu/mpz_utils.h>
#include <math.h>
#include <algorithm>
#include <iostream>
#include <common/measurement.h>

#define VERIFIER_NOT_ON_GPU 0
#define PROVER_NOT_ON_GPU 0

class CPUCrypto : public Crypto {
  private:
    int max_kernel_vec_size;
    bool gpu_enabled, initialized;

    gmpz_array exp, rop, base;
    WORD *p_host, *pp_host, *nsq_host;
    mpz_t *result;

    gmpz_array g_cache;
    gmpz_array gr_cache;

    struct mp_sw *sw_host;

    void modexp(gmpz_array *rop, gmpz_array *base, struct mp_sw *sw_host);
    void modexp(gmpz_array *rop, gmpz_array *exp, gmpz_array *pow_cache);
    void create_cache(gmpz_array *cache, const mpz_t base);

  public:
    // GENERAL:
    CPUCrypto(int c_type, int c_in_use, int png_in_use,
              bool gen_keys = false,
              int field_mod_size = DEFAULT_FIELD_SIZE,
              int public_key_mod_size = DEFAULT_PUBLIC_KEY_MOD_BITS,
              int elgamal_priv_size = DEFAULT_ELGAMAL_PRIV_RAND_BITS);

    void init_crypto_state();

    virtual void elgamal_enc(mpz_t c1, mpz_t c2, mpz_t plain);
    virtual void elgamal_enc_vec(mpz_t *cipher, mpz_t *plain, int size);
    void dot_product_enc(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2);
};
#endif
