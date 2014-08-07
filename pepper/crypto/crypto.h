#ifndef CODE_PEPPER_CRYPTO_CRYPTO_H_
#define CODE_PEPPER_CRYPTO_CRYPTO_H_

#include <crypto/prng.h>

#define CRYPTO_ELGAMAL 0
#define DEFAULT_PUBLIC_KEY_MOD_BITS 1024
#define DEFAULT_ELGAMAL_PRIV_RAND_BITS 160
#define DEFAULT_FIELD_SIZE 128
#define CRYPTO_TYPE_PUBLIC 0
#define CRYPTO_TYPE_PRIVATE 1
#define BUFLEN 10240

// enable fixed base exponentiation and multiexp tricks by default
#define FIXED_BASE_EXP_TRICK
#define SIMUL_EXP_TRICK

#ifdef FIXED_BASE_EXP_TRICK
// gmpmee is a C library
extern "C" {
  #include <gmpmee.h>
}
#endif

class Crypto {
  protected:
    // global state
    gmp_randstate_t state;

    int type;
    int crypto_in_use;

    int public_key_mod_bits;
    int elgamal_priv_rand_bits;
    int expansion_factor;
    int field_mod_size;

    mpz_t n, p, q, lambda, g, mu, n2, temp1, temp2, gn, gr, r;

    Prng *prng_private;
    Prng *prng_decommit_queries;
#ifdef FIXED_BASE_EXP_TRICK
    gmpmee_fpowm_tab table_g;
    gmpmee_fpowm_tab table_gr;
#endif

  public:
    // GENERAL:
    Crypto(int c_type, int c_in_use, int png_in_use,
           bool gen_keys = false,
           int field_mod_size = DEFAULT_FIELD_SIZE,
           int public_key_mod_size = DEFAULT_PUBLIC_KEY_MOD_BITS,
           int elgamal_priv_size = DEFAULT_ELGAMAL_PRIV_RAND_BITS);

    virtual ~Crypto();

    // initialize/generate crypto state
    void init_crypto_state(bool, int);
    void load_crypto_state(int);
    void generate_crypto_keys();
    int get_crypto_in_use();
    int get_public_modulus_size();
    int get_elgamal_priv_key_size();

    void init_prng_decommit_queries();
    void dump_seed_decommit_queries();

    // ElGamal
    // init
    void generate_elgamal_crypto_keys();
    void set_elgamal_pub_key(mpz_t p_arg, mpz_t g_arg);
    void set_elgamal_priv_key(mpz_t p_arg, mpz_t g_arg, mpz_t r_arg);
    void elgamal_precompute();
    void elgamal_get_generator(mpz_t *g_arg);
    void elgamal_get_public_modulus(mpz_t *p_arg);
    void elgamal_get_order(mpz_t *q_arg);

    // operations
    virtual void elgamal_enc(mpz_t c1, mpz_t c2, mpz_t plain);
    void elgamal_enc_with_rand(mpz_t c1, mpz_t c2, mpz_t plain, mpz_t r);
    virtual void elgamal_enc_vec(mpz_t *cipher, mpz_t *plain, int size);
    void elgamal_dec(mpz_t plain, mpz_t c1, mpz_t c2);
    void
    elgamal_hadd(mpz_t res1, mpz_t res2, mpz_t c1_1, mpz_t c1_2, mpz_t
                 c2_1, mpz_t c2_2);
    void elgamal_smul(mpz_t res1, mpz_t res2, mpz_t c1, mpz_t c2, mpz_t
                      coefficient);
    virtual void dot_product_enc(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2);
    void dot_product_simel_enc(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2);
    void dot_product_regular_enc(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2); 
    // public randomness (prover gets to see the seed of this PRNG)
    void get_random_pub(mpz_t m, mpz_t n);
    void get_randomb_pub(mpz_t m, int nbits);
    void get_random_vec_pub(uint32_t size, mpz_t *vec, mpz_t n);
    void get_random_vec_pub(uint32_t size, mpz_t *vec, int nbits);
    void get_random_vec_pub(uint32_t size, mpq_t *vec, int nbits);

    // private randomness functions
    void get_randomb_priv(char* buf, int nbits);
    void get_random_priv(mpz_t m, mpz_t n);
    void get_randomb_priv(mpz_t m, int nbits);
    void get_random_vec_priv(uint32_t size, mpz_t *vec, mpz_t n);
    void get_random_vec_priv(uint32_t size, mpz_t *vec, int nbits);
    void get_random_vec_priv(uint32_t size, mpq_t *vec, int nbits);

    void get_vec(uint32_t size, mpz_t * vec, int nbits);
    void find_prime(mpz_t prime, unsigned long int n);

    // function for fixed base exponentiation
    void g_fpowm(mpz_t ans, mpz_t exp);
};
#endif  // CODE_PEPPER_CRYPTO_CRYPTO_H_
