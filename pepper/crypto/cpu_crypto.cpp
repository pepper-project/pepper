#include <cassert>

#include "cpu_crypto.h"
#include "powm_cache.h"
#include "lib_gpu/mp_modexp.h"
#include "lib_gpu/mpz_utils.h"
#include "common/utility.h"

//#ifdef ON_TACC
using std::min;
//#endif

CPUCrypto::CPUCrypto(int c_type, int c_in_use, int png_in_use, bool gen_keys,
                     int field_mod_size,
                     int public_key_mod_size, int elgamal_priv_size):
  Crypto(c_type, c_in_use, png_in_use, gen_keys,
         field_mod_size,
         public_key_mod_size, elgamal_priv_size),
  initialized(false) {
  init_crypto_state();
}

void CPUCrypto::
init_crypto_state() {
  if (initialized)
    return;

  int bits_per_limb = sizeof(gmp_limb_t) * 8;
  int public_key_num_limbs = (public_key_mod_bits + bits_per_limb - 1) / bits_per_limb;
  max_kernel_vec_size = MAX_NUM_THREADS / public_key_num_limbs;

  exp.     alloc(max_kernel_vec_size, public_key_num_limbs);
  rop.     alloc(max_kernel_vec_size, public_key_num_limbs);
  base.    alloc(max_kernel_vec_size, public_key_num_limbs);
  g_cache. alloc(max_kernel_vec_size, public_key_num_limbs);
  gr_cache.alloc(max_kernel_vec_size, public_key_num_limbs);

  alloc_init_vec(&result, max_kernel_vec_size);
  sw_host = (struct mp_sw*)malloc(sizeof(struct mp_sw) * max_kernel_vec_size);

  mpz_t n, pp, nsq;
  mpz_init_set_ui(n,   0);
  mpz_init_set_ui(pp,  0);
  mpz_init_set_ui(nsq, 0);

  // Calculate 2^public_key_mod_bits
  mpz_set_ui(n, 0);
  mpz_setbit(n, public_key_mod_bits);

  // Calculate primep such that n*n^{-1} - p*primep = 1
  mpz_invert(pp, n, p);
  mpz_mul(pp, pp, n);
  mpz_sub_ui(pp, pp, 1);
  mpz_tdiv_q(pp, pp, p);

  // Calculate n^2 mod prime
  mpz_mul(nsq, n, n);
  mpz_mod(nsq, nsq, p);

  p_host   = to_gpu_format(p,   public_key_num_limbs);
  pp_host  = to_gpu_format(pp,  public_key_num_limbs);
  nsq_host = to_gpu_format(nsq, public_key_num_limbs);

  create_cache(&g_cache,  g);
  create_cache(&gr_cache, gr);

  //mpz_clears(n, pp, nsq, 0);
  mpz_clear(n);
  mpz_clear(pp);
  mpz_clear(nsq);
}

void CPUCrypto::
create_cache(gmpz_array *cache, const mpz_t base) {
  PowmCache c(base, p, public_key_mod_bits, WINDOW_SIZE);
  cache->fromMPZArray(c.getCache(), c.size());
  assert(int(cache->elementSize * sizeof(cache->data[0]) * 8) == public_key_mod_bits);

  mp_montgomerize_cpu_nocopy(
    cache->numElements,
    cache->host_mem_pool, cache->host_mem_pool,
    nsq_host, p_host, pp_host,
    cache->elementSize);
}

void CPUCrypto::
modexp(gmpz_array *rop, gmpz_array *exp, gmpz_array *powm_cache) {
  assert(rop->numElements <= exp->numElements);
  assert(rop->elementSize == exp->elementSize);
  assert(rop->elementSize == powm_cache->elementSize);

  // Use rop to get array parameters (numElements, elementSize). base should
  // not be trusted since it isn't being written to (and can therefore include a
  // larger array that's been cached).
  mp_modexp_cached_cpu_nocopy(
    rop->numElements, rop->host_mem_pool, exp->host_mem_pool,
    powm_cache->host_mem_pool,
    nsq_host, p_host, pp_host,
    rop->elementSize);
}

void CPUCrypto::
modexp(gmpz_array *rop, gmpz_array *base, struct mp_sw *sw_host) {
  assert(rop->numElements <= base->numElements);
  assert(rop->elementSize == base->elementSize);

  // Use rop to get array parameters (numElements, elementSize). base should
  // not be trusted since it isn't being written to (and can therefore include a
  // larger array that's been cached).
  mp_many_modexp_mont_cpu_nocopy(
    rop->numElements,
    rop->host_mem_pool, base->host_mem_pool, sw_host,
    nsq_host, p_host, pp_host,
    rop->elementSize);
}

static void
build_sw(struct mp_sw *sw, gmpz_array *exp, int exp_bits) {
  for (int i=0; i < exp->numElements; i++) {
    mp_get_sw(&sw[i], exp->getElemHost(i), exp_bits / 8 / sizeof(WORD));
  }
}

void CPUCrypto::
elgamal_enc(mpz_t c1, mpz_t c2, mpz_t plain) {
  // calls the CPU version
  Crypto::elgamal_enc(c1, c2, plain);
}

//#define DBG(n, name) cout << n << ": " << sw[0].name << ", " << sw[1].name << endl;

void CPUCrypto::
elgamal_enc_vec(mpz_t *cipher, mpz_t *plain, int size) {
  Measurement m;
  m.begin_with_init();
  for (int i = 0; i < size; i += max_kernel_vec_size) {
    int vec_size = min(exp.maxSize, size - i);
    rop.setSize(vec_size);

    // Compute g^{plain}
    exp.fromMPZArray(&plain[i], vec_size, 1, false);

    modexp(&rop, &exp, &g_cache);

    // While the modexp happens, we generate random values.
    for (int j = 0; j < vec_size; j++) {
      // select a random number of size PRIV_KEY_BITS
      // priv_rand_bits should fit in one gmpz_array element.
      prng_private->get_randomb((char*)exp.getElemHost(j), elgamal_priv_rand_bits);
    }

    // Load result of g^{plain}.
    rop.toMPZArray(&cipher[2 * i + 1], 2, false);

    // Compute gr^{rand}
    modexp(&rop, &exp, &gr_cache);
    rop.toMPZArray(&cipher[2 * i], 2, false);

    // We compute gr^{rand} * g^{plain} mod p
    for (int j = 0; j < vec_size; j++) {
      mpz_mul(cipher[2 * (i + j) + 1],
              cipher[2 * (i + j) + 1],
              cipher[2 * (i + j)]);
      mpz_mod(cipher[2 * (i + j) + 1], cipher[2 * (i + j) + 1], p);
    }

    // Compute g^{rand}
    modexp(&rop, &exp, &g_cache);

    // Load result of g^{rand}
    rop.toMPZArray(&cipher[2 * i], 2, false);
  }
  m.end();
  cout<<"Time taken per exponentiation = "<<m.get_papi_elapsed_time()/(3*size)<<" usec"<<endl;
  cout<<"#exps/sec = "<<(3*size*1000.0*1000)/m.get_papi_elapsed_time()<<endl;
}

void CPUCrypto::
dot_product_enc(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2) {
  mpz_set_ui(output, 1);
  mpz_set_ui(output2, 1);
  Measurement m;
  m.begin_with_init();

  const int maxCiphertextVecSize = max_kernel_vec_size / expansion_factor;

  for (uint32_t i = 0; i < size; i += max_kernel_vec_size) {
    int vec_size = min(max_kernel_vec_size, (int)(size - i));

    assert(vec_size <= max_kernel_vec_size);
    rop.setSize(vec_size);

    // Compute q^{d}
    exp.fromMPZArray(&d[i], vec_size, 1, false);

    // TODO: OPTIMIZE: remove this hardcoded 192 to use computation-provided exponent values
    // VV: Actually, this might not be necessary, since the sliding window
    // algorithm should ignore the most significant zeroes anyways.
    build_sw(sw_host, &exp, 192);

    for (int comp = 0; comp < expansion_factor; comp++) {
      base.fromMPZArray(&q[expansion_factor * i + comp],
                        vec_size,
                        expansion_factor,
                        false);
      modexp(&rop, &base, sw_host);

      // Load result of q^{d}.
      // TODO: This blocks until the modexp is done. Consider moving the reduce
      // step above this (would require some extra indexing trickery).
      rop.toMPZArray(result, 1, false);

      for (int j = 0; j < vec_size; j++) {
        if (comp % 2 == 0) {
          mpz_mul(output, output, result[j]);
          mpz_mod(output, output, p);
        } else {
          mpz_mul(output2, output2, result[j]);
          mpz_mod(output2, output2, p);
        }
      }
    }
  }
  m.end();
  cout<<"Time taken per exponentiation = "<<m.get_papi_elapsed_time()/(2*size)<<" usec"<<endl;
  cout<<"#exps/sec = "<<(2*size*1000.0*1000)/m.get_papi_elapsed_time()<<endl;
}
