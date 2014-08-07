#include <crypto/crypto.h>
#include <common/measurement.h>
#include <common/utility.h>
#include <common/pairing_util.h>

#define BUFLEN 10240
#define NUM_SAMPLES 10
#define MICROBENCHMARKS 1

// global state
Crypto *crypto;
mpz_t prime_128, prime_192, prime_220, prime_a_256, prime_e_256, prime_f_256;

// ========= common functions =========
void load_prime(int size, mpz_t prime) {
  char scratch_str[BUFLEN];
  snprintf(scratch_str, BUFLEN - 1, "prime_%d.txt", size);
  load_txt_scalar(prime, scratch_str, const_cast<char *>("static_state"));
}

void load_pairing_prime(string type, int size, mpz_t prime) {
  char scratch_str[BUFLEN];
  snprintf(scratch_str, BUFLEN - 1, "prime_%s_%d.txt", type.c_str(), size);
  load_txt_scalar(prime, scratch_str, const_cast<char *>("static_state"));
}

// ======== cost of ciphertext operations ======
double do_encrypt(int size_input, mpz_t *plain, mpz_t *cipher,
    int prime_size, mpz_t prime) {
  
  Measurement m;
  crypto->get_random_vec_priv(size_input, plain, prime);
  
  m.begin_with_init();
  for (int i=0; i<size_input; i++)
    crypto->elgamal_enc(cipher[2*i], cipher[2*i+1], plain[i]);
  m.end();
  return m.get_ru_elapsed_time();
}

double do_decrypt(int size_input, mpz_t *plain, mpz_t *cipher,
    int prime_size, mpz_t prime) {
  
  Measurement m;
  m.begin_with_init();
  for (int i=0; i<size_input; i++)
    crypto->elgamal_dec(plain[i], cipher[2*i], cipher[2*i+1]);
  m.end();
  return m.get_ru_elapsed_time();
}

double do_simel_ciphermul(int size_input, mpz_t *plain, mpz_t *cipher,
    int prime_size, mpz_t prime) {
  mpz_t out1, out2;
  alloc_init_scalar(out1);
  alloc_init_scalar(out2);

  Measurement m;
  m.begin_with_init();
  crypto->dot_product_simel_enc(size_input, cipher, plain, out1, out2);
  m.end();
  
  clear_scalar(out1);
  clear_scalar(out2);
  return m.get_ru_elapsed_time();
}

double do_regular_ciphermul(int size_input, mpz_t *plain, mpz_t *cipher,
    int prime_size, mpz_t prime) {
  mpz_t out1, out2;
  alloc_init_scalar(out1);
  alloc_init_scalar(out2);

  Measurement m;
  m.begin_with_init();
  crypto->dot_product_regular_enc(size_input, cipher, plain, out1, out2);
  m.end();
  
  clear_scalar(out1);
  clear_scalar(out2);
  return m.get_ru_elapsed_time();
}

void measure_encrypt(int size_input, mpz_t *plain,
    mpz_t *cipher, int prime_size, mpz_t prime) {
  
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++)
    measurements[i] = 
      do_encrypt(size_input, plain, cipher, prime_size, prime)/size_input;

  snprintf(scratch_str, BUFLEN-1, "e_%d", prime_size);
  print_stats(scratch_str, measurements);
}

void measure_decrypt(int size_input, mpz_t *plain,
    mpz_t *cipher, int prime_size, mpz_t prime) {
  
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++) {
    do_encrypt(size_input, plain, cipher, prime_size, prime);
    measurements[i] = do_decrypt(size_input, plain, cipher, prime_size, prime)/size_input;
  }
  snprintf(scratch_str, BUFLEN-1, "d_%d", prime_size);
  print_stats(scratch_str, measurements);
}

void measure_ciphermul(int size_input, mpz_t *plain, mpz_t *cipher,
    int prime_size, mpz_t prime) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++) {
    do_encrypt(size_input, plain, cipher, prime_size, prime);
    measurements[i] = do_simel_ciphermul(size_input, plain, cipher, prime_size, prime)/size_input;
  }
  snprintf(scratch_str, BUFLEN-1, "h_simel_%d", prime_size);
  print_stats(scratch_str, measurements);

  for (int i=0; i<NUM_SAMPLES; i++) {
    do_encrypt(size_input, plain, cipher, prime_size, prime);
    measurements[i] = do_regular_ciphermul(size_input, plain, cipher, prime_size, prime)/size_input;
  }
  
  snprintf(scratch_str, BUFLEN-1, "h_regular_%d", prime_size);
  print_stats(scratch_str, measurements);
}

void measure_ciphertext_ops() {
  int size_input = 1000;
  int expansion_factor = 2;

  // generate inputs
  mpz_t *plain, *cipher;
  alloc_init_vec(&plain, size_input);
  alloc_init_vec(&cipher, expansion_factor*size_input);

  measure_encrypt(size_input, plain, cipher, 128, prime_128);
  measure_encrypt(size_input, plain, cipher, 192, prime_192);
  measure_encrypt(size_input, plain, cipher, 220, prime_220);
  
  measure_decrypt(size_input, plain, cipher, 128, prime_128);
  measure_decrypt(size_input, plain, cipher, 192, prime_192);
  measure_decrypt(size_input, plain, cipher, 220, prime_220);
  
  measure_ciphermul(size_input, plain, cipher, 128, prime_128);
  measure_ciphermul(size_input, plain, cipher, 192, prime_192);
  measure_ciphermul(size_input, plain, cipher, 220, prime_220);
}

// ======== cost of plaintext operations =========
double do_field_mult(int size_input, mpz_t *vec1,
    mpz_t *vec2, mpz_t *vec3, mpz_t prime) {
  Measurement m;

  crypto->get_random_vec_priv(size_input, vec1, prime);
  crypto->get_random_vec_priv(size_input, vec2, prime);
  
  m.begin_with_init();
  for (int i=0; i<size_input; i++) {
    mpz_mul(vec3[i], vec1[i], vec2[i]);
    mpz_mod(vec3[i], vec3[i], prime);
  }
 
  for (int i=0; i<size_input; i++) {
    mpz_add(vec3[i], vec1[i], vec2[i]);
    mpz_mod(vec3[i], vec3[i], prime);
  }
  m.end();
  return m.get_ru_elapsed_time();
}

void measure_field_mult(int size_input, mpz_t *vec1, mpz_t *vec2, 
    mpz_t *vec3, int prime_size, mpz_t prime) {

  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++)
    measurements[i] = 
      do_field_mult(size_input, vec1, vec2, vec3, prime)/size_input;

  snprintf(scratch_str, BUFLEN-1, "f_mod_%d", prime_size);
  print_stats(scratch_str, measurements);
}

// cost of regular multiplication
double do_mult(int size_input, mpz_t *vec1,
    mpz_t *vec2, mpz_t *vec3, int operand_size1, int operand_size2) {
  Measurement m;

  crypto->get_random_vec_priv(size_input, vec1, operand_size1);
  crypto->get_random_vec_priv(size_input, vec2, operand_size2);
  
  m.begin_with_init();
  for (int i=0; i<size_input; i++) {
    mpz_mul(vec3[i], vec1[i], vec2[i]);
  }
  for (int i=0; i<size_input; i++) {
    mpz_add(vec3[i], vec1[i], vec2[i]);
  }
  m.end();
  return m.get_ru_elapsed_time();
}

void measure_mult(int size_input, mpz_t *vec1, mpz_t *vec2, 
    mpz_t *vec3, int operand_size1, int operand_size2) {

  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++)
    measurements[i] = 
      do_mult(size_input, vec1, vec2, vec3, operand_size1, operand_size2)/size_input;

  snprintf(scratch_str, BUFLEN-1, "f_%d_%d", operand_size1, operand_size2);
  print_stats(scratch_str, measurements);
}

double do_div(int size_input, mpz_t *vec1,
    mpz_t *vec2, mpz_t *vec3, int operand_size1, int operand_size2,
    mpz_t prime) {
  Measurement m;

  crypto->get_random_vec_priv(size_input, vec1, operand_size1);
  crypto->get_random_vec_priv(size_input, vec2, operand_size2);
  
  m.begin_with_init();
  for (int i=0; i<size_input; i++) {
    mpz_invert(vec3[i], vec2[i], prime);
    mpz_mul(vec3[i], vec3[i], vec1[i]);
    mpz_mod(vec3[i], vec3[i], prime);
  }
  m.end();
  return m.get_ru_elapsed_time();
}

void measure_div(int size_input, mpz_t *vec1, mpz_t *vec2,
    mpz_t *vec3, int operand_size1, int operand_size2,
    mpz_t prime) {

  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++)
    measurements[i] =
      do_div(size_input, vec1, vec2, vec3, operand_size1, operand_size2, prime)/size_input;

  snprintf(scratch_str, BUFLEN-1, "f_div_%d_%d", operand_size1, operand_size2);
  print_stats(scratch_str, measurements);
}

double do_rand_gen(int size_input, mpz_t *vec, mpz_t prime) {
  Measurement m;
  m.begin_with_init();
  crypto->get_random_vec_priv(size_input, vec, prime);
  m.end();
  return m.get_ru_elapsed_time();
}

void measure_rand(int size_input, mpz_t *vec, int prime_size, mpz_t prime) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++)
    measurements[i] = do_rand_gen(size_input, vec, prime)/size_input;

  snprintf(scratch_str, BUFLEN-1, "c_%d", prime_size);
  print_stats(scratch_str, measurements);
}

// cost of field addition
// cost of generating a random number
void measure_plaintext_ops() {
  int size_input = 1000000;

  mpz_t *vec1, *vec2, *vec3;
  
  alloc_init_vec(&vec1, size_input);
  alloc_init_vec(&vec2, size_input);
  alloc_init_vec(&vec3, size_input);
   // cost of a field multiplication followed by a field addition
  measure_field_mult(size_input, vec1, vec2, vec3, 128, prime_128);  
  measure_field_mult(size_input, vec1, vec2, vec3, 192, prime_192);  
  measure_field_mult(size_input, vec1, vec2, vec3, 220, prime_220);  

  // cost of a regular multiplication followed by an addition
  measure_mult(size_input, vec1, vec2, vec3, 32, 32);
  measure_mult(size_input, vec1, vec2, vec3, 128, 32);
  measure_mult(size_input, vec1, vec2, vec3, 192, 32);
  measure_mult(size_input, vec1, vec2, vec3, 220, 32);
  measure_mult(size_input, vec1, vec2, vec3, 128, 128);
  measure_mult(size_input, vec1, vec2, vec3, 192, 192);
  measure_mult(size_input, vec1, vec2, vec3, 220, 220); 

  measure_div(size_input, vec1, vec2, vec3, 128, 128, prime_128);
  measure_div(size_input, vec1, vec2, vec3, 220, 220, prime_220);

  // cost of generating pseudorandom numbers
  measure_rand(size_input, vec1, 128, prime_128);
  measure_rand(size_input, vec1, 192, prime_192);
  measure_rand(size_input, vec1, 220, prime_220);

  // cleanup
  clear_vec(size_input, vec1);
  clear_vec(size_input, vec2);
  clear_vec(size_input, vec3);
}

#if NONINTERACTIVE == 1
double do_fixed_exp_G1(int size_input, G1_t g, mpz_t *exp) {
  Measurement m;
  G1_t *result;
  alloc_init_vec_G1(&result, size_input);

  m.begin_with_init();
  G1_fixed_exp(result, g, exp, size_input);
  m.end();
  clear_vec_G1(size_input, result);

  return m.get_ru_elapsed_time();
}

double do_fixed_exp_G2(int size_input, G2_t g, mpz_t *exp) {
  Measurement m;
  G2_t *result;
  alloc_init_vec_G2(&result, size_input);

  m.begin_with_init();
  G2_fixed_exp(result, g, exp, size_input);
  m.end();

  clear_vec_G2(size_input, result);
  return m.get_ru_elapsed_time();
}


void measure_fixed_exp_base_curve(int size_input, int prime_size, mpz_t prime) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  G1_t g1;
  mpz_t *exp;

  alloc_init_vec(&exp, size_input);
  alloc_init_scalar_G1(g1);
  G1_random(g1);

  crypto->get_random_vec_priv(size_input, exp, prime);

  for (int i=0; i<NUM_SAMPLES; i++) {
    measurements[i] = do_fixed_exp_G1(size_input, g1, exp) / size_input;
  }

  snprintf(scratch_str, BUFLEN-1, "fixed_exp_base_%d", prime_size);
  print_stats(scratch_str, measurements);

  clear_scalar_G1(g1);
  clear_vec(size_input, exp);
}

void measure_fixed_exp_twist_curve(int size_input, int prime_size, mpz_t prime) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  G2_t g2;
  mpz_t *exp;

  alloc_init_vec(&exp, size_input);
  alloc_init_scalar_G2(g2);
  G2_random(g2);

  crypto->get_random_vec_priv(size_input, exp, prime);

  for (int i=0; i<NUM_SAMPLES; i++) {
    measurements[i] = do_fixed_exp_G2(size_input, g2, exp) / size_input;
  }

  snprintf(scratch_str, BUFLEN-1, "fixed_exp_twist_%d", prime_size);
  print_stats(scratch_str, measurements);

  clear_scalar_G2(g2);
  clear_vec(size_input, exp);
}

void measure_fixed_exp_base_curve_signedint(int size_input, int prime_size, mpz_t prime, int length) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  G1_t g1;
  mpz_t *exp;

  alloc_init_vec(&exp, size_input);
  alloc_init_scalar_G1(g1);
  G1_random(g1);

  mpz_t tmp;
  alloc_init_scalar(tmp);
  mpz_set_ui(tmp, 1);
  mpz_mul_2exp(tmp, tmp, length);
  for (int i = 0; i < size_input; i++) {
    crypto->get_randomb_priv(exp[i], length);
    mpz_sub(exp[i], exp[i], tmp);
  }
  clear_scalar(tmp);

  for (int i=0; i<NUM_SAMPLES; i++) {
    measurements[i] = do_fixed_exp_G1(size_input, g1, exp) / size_input;
  }

  snprintf(scratch_str, BUFLEN-1, "fixed_exp_base_%d_signedint_%d", prime_size, length);
  print_stats(scratch_str, measurements);

  clear_scalar_G1(g1);
  clear_vec(size_input, exp);
}

void measure_fixed_exp_twist_curve_signedint(int size_input, int prime_size, mpz_t prime, int length) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  G2_t g2;
  mpz_t *exp;

  alloc_init_vec(&exp, size_input);
  alloc_init_scalar_G2(g2);
  G2_random(g2);

  mpz_t tmp;
  alloc_init_scalar(tmp);
  mpz_set_ui(tmp, 1);
  mpz_mul_2exp(tmp, tmp, length);
  for (int i = 0; i < size_input; i++) {
    crypto->get_randomb_priv(exp[i], length);
    mpz_sub(exp[i], exp[i], tmp);
  }
  clear_scalar(tmp);

  for (int i=0; i<NUM_SAMPLES; i++) {
    measurements[i] = do_fixed_exp_G2(size_input, g2, exp) / size_input;
  }

  snprintf(scratch_str, BUFLEN-1, "fixed_exp_twist_%d_signedint_%d", prime_size, length);
  print_stats(scratch_str, measurements);

  clear_scalar_G2(g2);
  clear_vec(size_input, exp);
}

double do_multi_exp_G1(int size_input, G1_t *base, mpz_t *exp) {
  Measurement m;
  G1_t tmp;
  alloc_init_scalar_G1(tmp);

  m.begin_with_init();
  multi_exponentiation_G1(size_input, base, exp, tmp);
  m.end();

  clear_scalar_G1(tmp);

  return m.get_ru_elapsed_time();
}

double do_multi_exp_G2(int size_input, G2_t *base, mpz_t *exp) {
  Measurement m;
  G2_t tmp;
  alloc_init_scalar_G2(tmp);

  m.begin_with_init();
  multi_exponentiation_G2(size_input, base, exp, tmp);
  m.end();

  clear_scalar_G2(tmp);

  return m.get_ru_elapsed_time();
}

void measure_multi_exp_base_curve(int size_input, int prime_size, mpz_t prime) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  G1_t *base;
  mpz_t *exp;

  alloc_init_vec(&exp, size_input);
  alloc_init_vec_G1(&base, size_input);

  crypto->get_random_vec_priv(size_input, exp, prime);

  for (int i=0; i < size_input; i++) {
    G1_random(base[i]);
  }

  for (int i=0; i < NUM_SAMPLES; i++) {
    measurements[i] = do_multi_exp_G1(size_input, base, exp) / size_input;
  }

  snprintf(scratch_str, BUFLEN-1, "multi_exp_base_%d", prime_size);
  print_stats(scratch_str, measurements);

  clear_vec_G1(size_input, base);
  clear_vec(size_input, exp);
}

void measure_multi_exp_twist_curve(int size_input, int prime_size, mpz_t prime) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  G2_t *base;
  mpz_t *exp;

  alloc_init_vec(&exp, size_input);
  alloc_init_vec_G2(&base, size_input);

  crypto->get_random_vec_priv(size_input, exp, prime);

  for (int i=0; i < size_input; i++) {
    G2_random(base[i]);
  }

  for (int i=0; i < NUM_SAMPLES; i++) {
    measurements[i] = do_multi_exp_G2(size_input, base, exp) / size_input;
  }

  snprintf(scratch_str, BUFLEN-1, "multi_exp_twist_%d", prime_size);
  print_stats(scratch_str, measurements);

  clear_vec_G2(size_input, base);
  clear_vec(size_input, exp);
}

void measure_multi_exp_base_curve_signedint(int size_input, int prime_size, mpz_t prime, int length) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  G1_t *base;
  mpz_t *exp;

  alloc_init_vec(&exp, size_input);
  alloc_init_vec_G1(&base, size_input);

  mpz_t tmp;
  alloc_init_scalar(tmp);
  mpz_set_ui(tmp, 1);
  mpz_mul_2exp(tmp, tmp, length);
  for (int i = 0; i < size_input; i++) {
    crypto->get_randomb_priv(exp[i], length);
    mpz_sub(exp[i], exp[i], tmp);
  }
  clear_scalar(tmp);

  for (int i=0; i < size_input; i++) {
    G1_random(base[i]);
  }

  for (int i=0; i < NUM_SAMPLES; i++) {
    measurements[i] = do_multi_exp_G1(size_input, base, exp) / size_input;
  }

  snprintf(scratch_str, BUFLEN-1, "multi_exp_base_%d_signedint_%d", prime_size, length);
  print_stats(scratch_str, measurements);

  clear_vec_G1(size_input, base);
  clear_vec(size_input, exp);
}

void measure_multi_exp_twist_curve_signedint(int size_input, int prime_size, mpz_t prime, int length) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  G2_t *base;
  mpz_t *exp;

  alloc_init_vec(&exp, size_input);
  alloc_init_vec_G2(&base, size_input);

  mpz_t tmp;
  alloc_init_scalar(tmp);
  mpz_set_ui(tmp, 1);
  mpz_mul_2exp(tmp, tmp, length);
  for (int i = 0; i < size_input; i++) {
    crypto->get_randomb_priv(exp[i], length);
    mpz_sub(exp[i], exp[i], tmp);
  }
  clear_scalar(tmp);

  for (int i=0; i < size_input; i++) {
    G2_random(base[i]);
  }

  for (int i=0; i < NUM_SAMPLES; i++) {
    measurements[i] = do_multi_exp_G2(size_input, base, exp) / size_input;
  }

  snprintf(scratch_str, BUFLEN-1, "multi_exp_twist_%d_signedint_%d", prime_size, length);
  print_stats(scratch_str, measurements);

  clear_vec_G2(size_input, base);
  clear_vec(size_input, exp);
}

double do_pairing(int size_input, G1_t *g1, G2_t *g2) {
  Measurement m;
  GT_t tmp;
  alloc_init_scalar_GT(tmp);

  m.begin_with_init();
  for (int i = 0; i < size_input; i++) {
    do_pairing(tmp, g1[i], g2[i]);
  }
  m.end();

  clear_scalar_GT(tmp);

  return m.get_ru_elapsed_time();
}

void measure_pairing(int size_input, int prime_size, mpz_t prime) {
  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  G1_t* g1;
  G2_t* g2;

  alloc_init_vec_G1(&g1, size_input);
  alloc_init_vec_G2(&g2, size_input);

  for (int i=0; i < size_input; i++) {
    G1_random(g1[i]);
    G2_random(g2[i]);
  }

  for (int i=0; i<NUM_SAMPLES; i++) {
    measurements[i] = do_pairing(size_input, g1, g2) / size_input;
  }

  snprintf(scratch_str, BUFLEN-1, "pairing_%d", prime_size);
  print_stats(scratch_str, measurements);

  clear_vec_G1(size_input, g1);
  clear_vec_G2(size_input, g2);
}

void measure_pairing_ops() {
  int size_input = 10000;
  //pairing_t a_pairing, e_pairing, f_pairing;

  // TODO fix this for bn curve
  // load the pairing
  init_pairing_from_file("static_state/f_256.param", prime_f_256);
  //init_pairing_from_file(e_pairing, "static_state/e_256.param");
  //init_pairing_from_file(a_pairing, "static_state/a_256.param");

  // do the micro benchmark

  measure_fixed_exp_base_curve(size_input, 256, prime_f_256);
  measure_fixed_exp_twist_curve(size_input, 256, prime_f_256);
  measure_multi_exp_base_curve(size_input, 256, prime_f_256);
  measure_multi_exp_twist_curve(size_input, 256, prime_f_256);

  measure_fixed_exp_base_curve_signedint(size_input, 256, prime_f_256, 32);
  measure_fixed_exp_twist_curve_signedint(size_input, 256, prime_f_256, 32);
  measure_multi_exp_base_curve_signedint(size_input, 256, prime_f_256, 32);
  measure_multi_exp_twist_curve_signedint(size_input, 256, prime_f_256, 32);

  measure_pairing(size_input/100, 256, prime_f_256);

  //measure_pairing(size_input/10, a_pairing, 256, prime_a_256);

  //pairing_clear(f_pairing);
  //pairing_clear(e_pairing);
  //pairing_clear(a_pairing);
}
#endif

int main(int argc, char **argv) {
  // TODO: the parameter "128" below should be made "192" or "220"
  // depending on the field size. Modify the code to create one crypto
  // object for each field size. This matters for performance of
  // encryption
  crypto = new Crypto(CRYPTO_TYPE_PRIVATE, CRYPTO_ELGAMAL, PNG_CHACHA, false, 128, 1024, 160);
  alloc_init_scalar(prime_128);
  alloc_init_scalar(prime_192);
  alloc_init_scalar(prime_220);

  load_prime(128, prime_128);
  load_prime(192, prime_192);
  load_prime(220, prime_220);

#if NONINTERACTIVE == 1
  load_pairing_prime("a", 256, prime_a_256);
  load_pairing_prime("f", 256, prime_f_256);
  load_pairing_prime("e", 256, prime_e_256);
  measure_pairing_ops();
  clear_scalar(prime_a_256);
  clear_scalar(prime_f_256);
  clear_scalar(prime_e_256);
#endif

  measure_plaintext_ops();
  measure_ciphertext_ops();


  delete crypto;
  clear_scalar(prime_128);
  clear_scalar(prime_192);
  clear_scalar(prime_220);

  return 0;
}
