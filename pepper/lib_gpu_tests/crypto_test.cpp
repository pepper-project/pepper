#include "crypto/crypto.h"
#include "crypto/gpu_crypto.h"
#include "crypto/cpu_crypto.h"
#include "test_utils.h"

bool MICROBENCHMARKS = true;

#define FAST_INIT 1

#define QUICK_TEST  0
#define NORMAL_TEST 1
#define LARGE_TEST  2
#define STRESS_TEST 3

#define TEST_LEVEL NORMAL_TEST

#if TEST_LEVEL >= LARGE_TEST
#define ARRY_SIZE (1024 * 20 * 32)
#elif TEST_LEVEL >= NORMAL_TEST
#define ARRY_SIZE (1024 * 20)
#elif TEST_LEVEL >= QUICK_TEST
#define ARRY_SIZE (2049)
#endif

#define EGGROUP_NBITS 192
#define FIELD_NBITS 192

static mpz_t plain[ARRY_SIZE];
static mpz_t encoded[ARRY_SIZE];
static mpz_t cipher[2 * ARRY_SIZE];
static mpz_t cipher_gpu[2 * ARRY_SIZE];
static mpz_t cipher_cpu2[2 * ARRY_SIZE];
static mpz_t plain_cpu[ARRY_SIZE];
static mpz_t plain_cpu2[ARRY_SIZE];
static mpz_t plain_gpu[ARRY_SIZE];

static mpz_t pi[ARRY_SIZE];     // Vectors of FIELD_NBITS

static Crypto *c;
static GPUCrypto *gpuc;
static CPUCrypto *cpu2_c;

// Move over to Crypto eventually.
static void elgamal_dec_vec(Crypto *c, mpz_t *plain, mpz_t *cipher, int size) {
  for (int i = 0; i < size; i++) {
    c->elgamal_dec(plain[i], cipher[2 * i], cipher[2 * i + 1]);
  }
}

static void encode_plain(Crypto *c, mpz_t encode, mpz_t plain) {
  mpz_t g, p;
  mpz_init_set_ui(g, 0);
  mpz_init_set_ui(p, 0);

  c->elgamal_get_generator(&g);
  c->elgamal_get_public_modulus(&p);
  mpz_powm(encode, g, plain, p);
  mpz_clear(g);
  mpz_clear(p);
}

static void encode_plain(Crypto *c, mpz_t *encode, mpz_t *plain, int size) {
  mpz_t g, p;
  mpz_init_set_ui(g, 0);
  mpz_init_set_ui(p, 0);

  c->elgamal_get_generator(&g);
  c->elgamal_get_public_modulus(&p);
  for (int i = 0; i < size; i++) {
    mpz_powm(encode[i], g, plain[i], p);
  }

  mpz_clear(g);
  mpz_clear(p);
}

static void dot_product_enc(Crypto *c, mpz_t dp, const mpz_t *a, const mpz_t *b, int size) {
  mpz_t temp, p;
  mpz_init_set_ui(temp, 0);
  mpz_init_set_ui(p, 0);

  mpz_set_ui(dp, 0);
  for (int i = 0; i < size; i++) {
    mpz_addmul(dp, a[i], b[i]);
    //mpz_add(dp, dp, temp);
  }

  c->elgamal_get_public_modulus(&p);

  mpz_sub_ui(p, p, 1);
  mpz_mod(dp, dp, p);

  encode_plain(c, dp, dp);

  mpz_clear(temp);
  mpz_clear(p);
}

static void cmp_callback(int i)
{
  mpz_t g, p;
  mpz_init(g);
  mpz_init(p);

  gpuc->elgamal_get_generator(&g);
  gpuc->elgamal_get_public_modulus(&p);

  gmp_printf("g    : %Zx\n", g);
  gmp_printf("plain: %Zx\n", plain[i]);
  //gmp_printf("encoded cpu2: %Zx\n", cipher_cpu2[2 * i + 1]);
  //gmp_printf("encoded gpu: %Zx\n", cipher_gpu[2 * i + 1]);
  //
  mpz_clear(g);
  mpz_clear(p);
}

static void
test_elgamal_enc(int arry_size) {
  // We must decrypt since cipher texts contain randomness that we can't test.
  START_TEST("CPU elgamal_enc");
  c->elgamal_enc_vec(cipher, plain, arry_size);
  elgamal_dec_vec(c, plain_cpu, cipher, arry_size);
  RUN_TEST(cmp(encoded, plain_cpu, arry_size));

  // We must decrypt since cipher texts contain randomness that we can't test.
  START_TEST("CPU2 elgamal_enc");
  cpu2_c->elgamal_enc_vec(cipher_cpu2, plain, arry_size);
  elgamal_dec_vec(cpu2_c, plain_cpu2, cipher_cpu2, arry_size);
  RUN_TEST(cmp(encoded, plain_cpu2, arry_size, cmp_callback));
  //RUN_TEST(cmp(encoded, plain_cpu2, 1, cmp_callback));

  START_TEST("GPU elgamal_enc");
  gpuc->elgamal_enc_vec(cipher_gpu, plain, arry_size);
  elgamal_dec_vec(gpuc, plain_gpu, cipher_gpu, arry_size);
  //RUN_TEST(cmp(encoded, cipher_gpu, 1, cmp_callback));
  RUN_TEST(cmp(encoded, plain_gpu, arry_size, cmp_callback));
}

static void
test_dot_product_enc(int arry_size, bool reencrypt = true) {
  mpz_t dp_cipher[2];
  mpz_t dp, dp_cpu, dp_gpu, dp_cpu2;

  init(dp_cipher, 2, EGGROUP_NBITS);
  mpz_init_set_ui(dp, 0);
  mpz_init_set_ui(dp_cpu, 0);
  mpz_init_set_ui(dp_gpu, 0);
  mpz_init_set_ui(dp_cpu2, 0);

  dot_product_enc(c, dp, pi, pi, arry_size);

  if (reencrypt)
  {
#if FAST_INIT == 1
    gpuc->elgamal_enc_vec(cipher, pi, arry_size);
#else
    c->elgamal_enc_vec(cipher, pi, arry_size);
#endif
  }

  /*
  START_TEST("CPU dot_product_enc");
  c->dot_product_enc(arry_size, cipher, pi, dp_cipher[0], dp_cipher[1]);
  c->elgamal_dec(dp_cpu, dp_cipher[0], dp_cipher[1]);
  RUN_TEST(cmp(dp, dp_cpu));
  */

  START_TEST("CPU2 dot_product_enc");
  cpu2_c->dot_product_enc(arry_size, cipher, pi, dp_cipher[0], dp_cipher[1]);
  cpu2_c->elgamal_dec(dp_cpu2, dp_cipher[0], dp_cipher[1]);
  RUN_TEST(cmp(dp, dp_cpu2));

  START_TEST("GPU dot_product_enc");
  gpuc->dot_product_enc(arry_size, cipher, pi, dp_cipher[0], dp_cipher[1]);
  gpuc->elgamal_dec(dp_gpu, dp_cipher[0], dp_cipher[1]);
  RUN_TEST(cmp(dp, dp_gpu));

  mpz_clear(dp);
  clear(dp_cipher, 2);
}

int main() {
  c = new Crypto(CRYPTO_TYPE_PRIVATE, CRYPTO_ELGAMAL, PNG_CHACHA, false, FIELD_NBITS);
  gpuc = new GPUCrypto(CRYPTO_TYPE_PRIVATE, CRYPTO_ELGAMAL, PNG_CHACHA, false, FIELD_NBITS);
  cpu2_c = new CPUCrypto(CRYPTO_TYPE_PRIVATE, CRYPTO_ELGAMAL, PNG_CHACHA, false, FIELD_NBITS);

  // Always enable gpu
  if(!gpuc->is_gpu_enabled()) gpuc->enable_gpu(true);

  rand_init(plain, ARRY_SIZE, EGGROUP_NBITS);
  rand_init(pi, ARRY_SIZE, FIELD_NBITS);

  //for (int i = 0; i < ARRY_SIZE; i++)
    //mpz_set_ui(pi[i], i);//(i+1));

  mpz_set_ui(plain[0], 1);

  init(encoded, ARRY_SIZE, EGGROUP_NBITS);
  init(cipher, 2 * ARRY_SIZE, EGGROUP_NBITS);
  init(cipher_gpu, 2 * ARRY_SIZE, EGGROUP_NBITS);
  init(plain_cpu, ARRY_SIZE, EGGROUP_NBITS);
  init(plain_gpu, ARRY_SIZE, EGGROUP_NBITS);

  // Encode plaintext as g^{plain}
  encode_plain(c, encoded, plain, ARRY_SIZE);

  test_elgamal_enc(ARRY_SIZE);
  test_dot_product_enc(ARRY_SIZE);

#if TEST_LEVEL >= STRESS_TEST
  {
    int numTests = 0;
    int min = 1;
    int max = 2;
    for (int i = 0; i < ARRY_SIZE; i += min + (rand() % (max - min)))
    {
      cout << "[n = " << i << "]" << endl;
      test_dot_product_enc(i, false);
      numTests++;

      if (numTests % 5 == 0)
        max <<= 1;

      if (max > 128 && numTests % 5 == 0)
        min <<= 1;
    }
  }
#endif
}
