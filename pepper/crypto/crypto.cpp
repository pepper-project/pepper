#include <crypto/crypto.h>
#include <common/measurement.h>

Crypto::Crypto(int c_type, int c_in_use, int png_in_use, bool gen_keys,
               int c_field_mod_size,
               int public_key_mod_size, int elgamal_priv_size)
  : type(c_type), crypto_in_use(c_in_use), public_key_mod_bits(public_key_mod_size),
    elgamal_priv_rand_bits(elgamal_priv_size), field_mod_size(c_field_mod_size) {
  init_crypto_state(gen_keys, png_in_use);
}

Crypto::~Crypto() {
  if (prng_private != NULL)
    delete prng_private;

  if (prng_decommit_queries != NULL)
    delete prng_decommit_queries;

  mpz_clear(temp1);
  mpz_clear(temp2);
  mpz_clear(p);
  mpz_clear(q);
  mpz_clear(lambda);
  mpz_clear(g);
  mpz_clear(mu);
  mpz_clear(n2);
  mpz_clear(n);
  mpz_clear(gr);
  mpz_clear(r);

  gmp_randclear(state);

#ifdef FIXED_BASE_EXP_TRICK
  // free up memory.
  gmpmee_fpowm_clear(table_g);
  gmpmee_fpowm_clear(table_gr);
#endif
}

int Crypto::get_crypto_in_use() {
  return crypto_in_use;
}

void Crypto::init_prng_decommit_queries() {
  u8 key[CHACHA_KEY_SIZE];
  u8 iv[CHACHA_IV_SIZE];
  
  if (prng_decommit_queries != NULL)
    delete prng_decommit_queries;
  load_prng_seed(CHACHA_KEY_SIZE/8, key, CHACHA_IV_SIZE/8, iv);
  prng_decommit_queries = new Prng(PNG_CHACHA, key, iv);
}

void Crypto::dump_seed_decommit_queries() {
  u8 key[CHACHA_KEY_SIZE];
  u8 iv[CHACHA_IV_SIZE];
  prng_decommit_queries->get_seed(key, iv);
  dump_prng_seed(CHACHA_KEY_SIZE/8, key, CHACHA_IV_SIZE/8, iv);
}

void Crypto::init_crypto_state(bool gen_keys, int png_in_use) {
  gmp_randinit_default(state);

  // temp variables used in all crypto functions
  mpz_init(temp1);
  mpz_init(temp2);
  mpz_init(p);
  mpz_init(q);
  mpz_init(lambda);
  mpz_init(g);
  mpz_init(mu);
  mpz_init(n2);
  mpz_init(n);
  mpz_init(gr);
  mpz_init(r);

  // create a private prng
  prng_private = new Prng(png_in_use);

  // create a prng for decommitment queries
  u8 key[CHACHA_KEY_SIZE];
  u8 iv[CHACHA_IV_SIZE];

  if (type == CRYPTO_TYPE_PRIVATE) {
    ifstream rand;
    rand.open("/dev/urandom", ifstream::in);
    rand.read((char*)&key, (size_t)(CHACHA_KEY_SIZE/8));
    rand.read((char*)&iv, (size_t)(CHACHA_IV_SIZE/8));
    rand.close();
    prng_decommit_queries = new Prng(PNG_CHACHA, key, iv);
  } else {
    prng_decommit_queries = NULL;
  }

  // make sure to initialize prng_private before the next set of statements
  if (gen_keys == false)
    load_crypto_state(type);
  else
    generate_crypto_keys();

#ifdef FIXED_BASE_EXP_TRICK
  // determine the maximum number of bits in an exponent
  int max_bits_in_exponent = field_mod_size;
  if (max_bits_in_exponent < elgamal_priv_rand_bits)
    max_bits_in_exponent = elgamal_priv_rand_bits;

  // fixed base exponentiation trick: precompute powers of g and gr
  gmpmee_fpowm_init_precomp(table_g, g, p, 20, max_bits_in_exponent);
  gmpmee_fpowm_init_precomp(table_gr, gr, p, 20, max_bits_in_exponent);
#endif
}

void Crypto::load_crypto_state(int type) {
  char file_name[BUFLEN];

  mpz_t p, q, g, r, n;
  mpz_init(p);
  mpz_init(q);
  mpz_init(g);
  mpz_init(r);
  mpz_init(n);

  if (crypto_in_use == CRYPTO_ELGAMAL)
    expansion_factor = 2;

  // initialize the keys
  if (type == CRYPTO_TYPE_PRIVATE) {
    if (crypto_in_use == CRYPTO_ELGAMAL) {
      snprintf(file_name, BUFLEN-1, "pubkey_prime_%d.txt", public_key_mod_bits);
      load_txt_scalar(p, file_name, (char *)"static_state");

      snprintf(file_name, BUFLEN-1, "pubkey_gen_%d.txt", public_key_mod_bits);
      load_txt_scalar(g, file_name, (char *)"static_state");

      snprintf(file_name, BUFLEN-1, "privkey_r_%d.txt", public_key_mod_bits);
      load_txt_scalar(r, file_name, (char *)"static_state");

      set_elgamal_priv_key(p, g, r);
    }
  } else {
    if (crypto_in_use == CRYPTO_ELGAMAL) {
      snprintf(file_name, BUFLEN-1, "pubkey_prime_%d.txt", public_key_mod_bits);
      load_txt_scalar(p, file_name, (char *)"static_state");

      snprintf(file_name, BUFLEN-1, "pubkey_gen_%d.txt", public_key_mod_bits);
      load_txt_scalar(g, file_name, (char *)"static_state");

      set_elgamal_pub_key(p, g);
    }
  }
  mpz_clear(p);
  mpz_clear(q);
  mpz_clear(g);
  mpz_clear(r);
  mpz_clear(n);
}

void Crypto::generate_crypto_keys() {
  if (crypto_in_use == CRYPTO_ELGAMAL)
    generate_elgamal_crypto_keys();
}

void Crypto::generate_elgamal_crypto_keys() {
  //cout << "Key generation" << endl;
  //cout << "cryptosystem elgamal" << endl;
  //cout << "public_key_mod_size " << public_key_mod_bits << endl;
  //cout << "private_rand_size " << elgamal_priv_rand_bits << endl;
  //cout << "randomness "<< prng_private->get_type() << endl;

  // select a prime
  do {
    find_prime(q, public_key_mod_bits-1);

    mpz_mul_ui(p, q, 2);
    mpz_add_ui(p, p, 1);
  } while ((mpz_sizeinbase(p, 2) != (uint32_t)public_key_mod_bits) && (mpz_probab_prime_p(p, 15) != 0));

  // select a random g in Z_{p-1}
  mpz_sub_ui(temp1, p, 1);
  prng_private->get_random(g, temp1);
  mpz_mul(g, g, g);
  mpz_mod(g, g, p);


  prng_private->get_randomb(r, elgamal_priv_rand_bits);
  elgamal_precompute();
}


void Crypto::elgamal_precompute() {
  mpz_powm(gr, g, r, p);

  mpz_sub_ui(q, p, 1);
  mpz_div_ui(q, q, 2);
}

void Crypto::elgamal_get_generator(mpz_t *g_arg) {
  mpz_set(*g_arg, g);
}

void Crypto::elgamal_get_public_modulus(mpz_t *p_arg) {
  mpz_set(*p_arg, p);
}

void Crypto::elgamal_get_order(mpz_t *q_arg) {
  mpz_set(*q_arg, q);
}

int Crypto::get_public_modulus_size() {
  return public_key_mod_bits;
}

int Crypto::get_elgamal_priv_key_size() {
  return elgamal_priv_rand_bits;
}

void Crypto::g_fpowm(mpz_t ans, mpz_t exp) {
  gmpmee_fpowm(ans, table_g, exp);
}

void Crypto::elgamal_enc(mpz_t c1, mpz_t c2, mpz_t plain) {
  // First encode plain text as g^{plain}
  // NOTE: this isn't part of ElGamal encryption. BUT ginger uses
  // ElGamal encryption as an additively homomorphic encryption via an
  // exponentiation trick; so there is an additional exponentation

#ifdef FIXED_BASE_EXP_TRICK
  gmpmee_fpowm(temp2, table_g, plain); // fixed base trick
#else
  mpz_powm(temp2, g, plain, p);
#endif

  // select a random number of size elgamal_priv_rand_bits
  prng_private->get_randomb(temp1, elgamal_priv_rand_bits);

  // compute the first part of the ciphertext as g^{randomness}
#ifdef FIXED_BASE_EXP_TRICK
  gmpmee_fpowm(c1, table_g, temp1); // fixed base trick
#else
  mpz_powm(c1, g, temp1, p);
#endif

  // compute the second part of the ciphertext
#ifdef FIXED_BASE_EXP_TRICK
  gmpmee_fpowm(c2, table_gr, temp1); // fixed base trick
#else
  mpz_powm(c2, gr, temp1, p);
#endif

  mpz_mul(c2, c2, temp2);
  mpz_mod(c2, c2, p);
}

void Crypto::elgamal_enc_with_rand(mpz_t c1, mpz_t c2, mpz_t plain, mpz_t r) {
  // First encode plain text as g^{plain}
  // NOTE: this isn't part of ElGamal encryption. BUT ginger uses
  // ElGamal encryption as an additively homomorphic encryption via an
  // exponentiation trick; so there is an additional exponentation
  mpz_t temp2;
  mpz_init(temp2);

#ifdef FIXED_BASE_EXP_TRICK
  gmpmee_fpowm(temp2, table_g, plain); // fixed base trick
#else
  mpz_powm(temp2, g, plain, p);
#endif

  // select a random number of size elgamal_priv_rand_bits
  //prng_private->get_randomb(temp1, elgamal_priv_rand_bits);

  // compute the first part of the ciphertext as g^{randomness}
#ifdef FIXED_BASE_EXP_TRICK
  gmpmee_fpowm(c1, table_g, r); // fixed base trick
#else
  mpz_powm(c1, g, r, p);
#endif

  // compute the second part of the ciphertext
#ifdef FIXED_BASE_EXP_TRICK
  gmpmee_fpowm(c2, table_gr, r); // fixed base trick
#else
  mpz_powm(c2, gr, r, p);
#endif

  mpz_mul(c2, c2, temp2);
  mpz_mod(c2, c2, p);
  mpz_clear(temp2);
}


void Crypto::elgamal_enc_vec(mpz_t *cipher, mpz_t *plain, int size) {
  mpz_t *r;
  alloc_init_vec(&r, size);

  //Measurement m;
  //m.begin_with_init();
  for (int i = 0; i < size; i++) {
    prng_private->get_randomb(r[i], elgamal_priv_rand_bits);
  }

  omp_set_num_threads(NUM_THREADS);
  #pragma omp parallel
  {
    #pragma omp for
    for (int i = 0; i < size; i++)
      elgamal_enc_with_rand(cipher[2 * i], cipher[2 * i + 1], plain[i], r[i]);
  }
  //m.end();
  //cout<<"Enc size is "<<size<<" and time taken per encryption= "<<m.get_ru_elapsed_time()/size<<" usec"<<endl;
  clear_vec(size, r);
}

void Crypto::elgamal_dec(mpz_t plain, mpz_t c1, mpz_t c2) {
  mpz_powm(plain, c1, r, p);
  mpz_invert(plain, plain, p);
  mpz_mul(plain, plain, c2);
  mpz_mod(plain, plain, p);
}

void Crypto::elgamal_hadd(mpz_t res1, mpz_t res2, mpz_t c1_1, mpz_t c1_2, mpz_t c2_1, mpz_t c2_2) {
  mpz_mul(res1, c1_1, c2_1);
  mpz_mod(res1, res1, p);

  mpz_mul(res2, c1_2, c2_2);
  mpz_mod(res2, res2, p);
}

void Crypto::elgamal_smul(mpz_t res1, mpz_t res2, mpz_t c1, mpz_t c2, mpz_t coefficient) {
  if (mpz_cmp_ui(coefficient, 0) == 0) {
    mpz_set_ui(res1, 1);
    mpz_set_ui(res2, 1);
  } else {
    mpz_powm(res1, c1, coefficient, p);
    mpz_powm(res2, c2, coefficient, p);
  }
}

void Crypto::set_elgamal_pub_key(mpz_t p_arg, mpz_t g_arg) {
  mpz_set(p, p_arg);
  mpz_set(g, g_arg);
}

void Crypto::set_elgamal_priv_key(mpz_t p_arg, mpz_t g_arg, mpz_t r_arg) {
  mpz_set(p, p_arg);
  mpz_set(g, g_arg);
  mpz_set(r, r_arg);
  elgamal_precompute();
}

void Crypto::get_random_pub(mpz_t m, mpz_t n) {
  prng_decommit_queries->get_random(m, n);
}

void Crypto::get_randomb_pub(mpz_t m, int nbits) {
  prng_decommit_queries->get_randomb(m, nbits);
}

void Crypto::get_random_vec_pub(uint32_t size, mpz_t *vec, mpz_t n) {
  for (uint32_t i = 0; i < size; i++)
    prng_decommit_queries->get_random(vec[i], n);
}

void Crypto::get_random_vec_pub(uint32_t size, mpz_t *vec, int nbits) {
  for (uint32_t i = 0; i < size; i++)
    prng_decommit_queries->get_randomb(vec[i], nbits);
}

void Crypto::get_random_vec_pub(uint32_t size, mpq_t *vec, int nbits) {
  for (uint32_t i = 0; i < size; i++) {
    prng_decommit_queries->get_randomb(mpq_numref(vec[i]), nbits);
    mpz_set_ui(mpq_denref(vec[i]), 1);
    mpq_canonicalize(vec[i]);
  }
}

void Crypto::get_randomb_priv(char* buf, int nbits) {
  prng_private->get_randomb(buf, nbits);
}

void Crypto::get_random_priv(mpz_t m, mpz_t n) {
  prng_private->get_random(m, n);
}

void Crypto::get_randomb_priv(mpz_t m, int nbits) {
  prng_private->get_randomb(m, nbits);
}

void Crypto::get_random_vec_priv(uint32_t size, mpz_t *vec, mpz_t n) {
  for (uint32_t i = 0; i < size; i++)
    prng_private->get_random(vec[i], n);
}

void Crypto::get_random_vec_priv(uint32_t size, mpz_t *vec, int nbits) {
  for (uint32_t i = 0; i < size; i++)
    prng_private->get_randomb(vec[i], nbits);
}

void Crypto::get_random_vec_priv(uint32_t size, mpq_t *vec, int nbits) {
  for (uint32_t i = 0; i < size; i++) {
    prng_private->get_randomb(mpq_numref(vec[i]), nbits);
    mpz_set_ui(mpq_denref(vec[i]), 1);
    mpq_canonicalize(vec[i]);
  }
}

void Crypto::find_prime(mpz_t prime, unsigned long int n) {
  prng_private->get_randomb(prime, n);

  while (mpz_probab_prime_p(prime, 15) == 0)
    prng_private->get_randomb(prime, n);
}


void Crypto::dot_product_enc(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2) {
  // Redirect to simultaneous encrypted dotp on CPU if SIMUL_EXP_TRICK
  // is enabled
#ifdef SIMUL_EXP_TRICK
  return dot_product_simel_enc(size, q, d, output, output2);
#else
  return dot_product_regular_enc(size, q, d, output, output2);
#endif
}

void Crypto::dot_product_regular_enc(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2) {
  mpz_set_ui(output, 1);
  mpz_set_ui(output2, 1);

  //Measurement m;
  //m.begin_with_init();
#ifdef DEBUG_MALICIOUS_PROVER
  mpz_set(output, q[0]);
  mpz_set(output2, q[1]);
  return;
#endif

  omp_set_num_threads(NUM_THREADS);
  #pragma omp parallel shared(output,output2)
  {
    //cout <<"dot_product_enc_num_threads "<< omp_get_num_threads()<<endl;
    mpz_t temp1, temp2, priv_output, priv_output2;
    mpz_init_set_ui(priv_output, 1);
    mpz_init_set_ui(priv_output2, 1);
    mpz_init(temp1);
    mpz_init(temp2);

    if (crypto_in_use == CRYPTO_ELGAMAL) {
      #pragma omp for
      for (uint32_t i=0; i<size; i++) {
        elgamal_smul(temp1, temp2, q[i], q[size+i], d[i]);
        elgamal_hadd(priv_output, priv_output2, priv_output, priv_output2, temp1, temp2);
      }
      #pragma omp critical
      {
        elgamal_hadd(output, output2, output, output2, priv_output, priv_output2);
      }
    }
    mpz_clear(temp1);
    mpz_clear(temp2);
    mpz_clear(priv_output);
    mpz_clear(priv_output2);
  }
  //m.end();
  //cout<<"Time taken per exponentiation = "<<m.get_papi_elapsed_time()/(2*size)<<" usec"<<endl;
  //cout<<"#exps/sec = "<<(2*size*1000.0*1000)/m.get_papi_elapsed_time()<<endl;
}

void Crypto::dot_product_simel_enc(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2) {
#ifdef DEBUG_MALICIOUS_PROVER
  mpz_set(output, q[0]);
  mpz_set(output2, q[1]);
  return;
#endif

  //Measurement m;
  mpz_set_ui(output, 1);
  mpz_set_ui(output2, 1);

  //m.begin_with_init();
  int num_threads = NUM_THREADS;
  int sizes[num_threads];
  int sum = 0;
  for (int i=0; i<num_threads-1; i++) {
    sizes[i] = size/num_threads;
    sum += sizes[i];
  }
  sizes[num_threads-1] = size - sum;

  omp_set_num_threads(num_threads);
  #pragma omp parallel shared(output,output2)
  {
    mpz_t temp1, temp2, priv_output, priv_output2;
    mpz_init_set_ui(priv_output, 1);
    mpz_init_set_ui(priv_output2, 1);
    mpz_init(temp1);
    mpz_init(temp2);

    int id = omp_get_thread_num();
    int base = 0;
    for (int i=0; i<id; i++)
      base = base + sizes[i];

    gmpmee_spowm_block(priv_output, &q[base], &d[base], sizes[id], p, 5);
    gmpmee_spowm_block(priv_output2, &q[size+base], &d[base], sizes[id], p, 5);

    #pragma omp critical
    {
      elgamal_hadd(output, output2, output, output2, priv_output, priv_output2);
    }
    mpz_clear(temp1);
    mpz_clear(temp2);
    mpz_clear(priv_output);
    mpz_clear(priv_output2);
  }
  //m.end();
  //cout<<"Array size is "<<size<<" and time taken per exponentiation = "<<m.get_ru_elapsed_time()/(2*size)<<" usec"<<endl;
}
