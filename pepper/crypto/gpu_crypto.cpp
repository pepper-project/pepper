#include <iostream>
#include <cassert>
#include <cuda_runtime_api.h>
#include <helper_cuda.h>

#include "gpu_crypto.h"
#include "powm_cache.h"
#include "lib_gpu/mp_modexp.h"
#include "lib_gpu/mpz_utils.h"
#include "common/utility.h"

#include <common/mpnvector.h>
//#ifdef ON_TACC
using std::min;
using std::cerr;
//#endif

// VV: Debug code. Will move soon.
#define AL_NONE   0
#define AL_TESTED 1
#define AL_DEBUG  2

#ifdef NDEBUG
#define ASSERT_LEVEL AL_NONE
#else
#define ASSERT_LEVEL AL_NONE
#endif

GPUCrypto::GPUCrypto(int c_type, int c_in_use, int png_in_use, bool gen_keys,
                     int field_mod_size,
                     int public_key_mod_size, int elgamal_priv_size):
  Crypto(c_type, c_in_use, png_in_use, gen_keys,
         field_mod_size,
         public_key_mod_size, elgamal_priv_size),
  initialized(false) {
  if (type == CRYPTO_TYPE_PRIVATE) {
#if VERIFIER_NOT_ON_GPU == 1
    enable_gpu(false);
#else
    enable_gpu(true);
#endif
  }

  if (type == CRYPTO_TYPE_PUBLIC) {
#if PROVER_NOT_ON_GPU == 1
    enable_gpu(false);
#else
    enable_gpu(true);
#endif
  }
}

void GPUCrypto::
init_crypto_state() {
  if (!is_gpu_enabled() || initialized)
    return;

  int bits_per_limb = sizeof(gmp_limb_t) * 8;
  int public_key_num_limbs = div_roundup(public_key_mod_bits, bits_per_limb);
  maxModexpInstances = MAX_NUM_THREADS / public_key_num_limbs;

  exp.     alloc(maxModexpInstances, public_key_num_limbs);
  rop.     alloc(maxModexpInstances, public_key_num_limbs);
  base.    alloc(MP_MAX_MULTI_EXP * maxModexpInstances, public_key_num_limbs);
  g_cache. alloc(maxModexpInstances, public_key_num_limbs);
  gr_cache.alloc(maxModexpInstances, public_key_num_limbs);

  alloc_init_vec(&result, maxModexpInstances);

#ifdef SIMUL_EXP_TRICK
  checkCudaErrors(cudaMalloc((void**)&multi_sw_dev,
                           sizeof(struct mp_multi_sw) * maxModexpInstances));
#else
  checkCudaErrors(cudaMalloc((void**)&sw_dev, sizeof(struct mp_sw) * maxModexpInstances));
#endif

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

  p_dev   = to_dev_memory(p,   public_key_num_limbs);
  pp_dev  = to_dev_memory(pp,  public_key_num_limbs);
  nsq_dev = to_dev_memory(nsq, public_key_num_limbs);

  // Setup GPU and all that.
  checkCudaErrors(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

  create_cache(&g_cache,  g);
  create_cache(&gr_cache, gr);

  //mpz_clears(n, pp, nsq, 0);
  mpz_clear(n);
  mpz_clear(pp);
  mpz_clear(nsq);

  initialized = true;
}

void GPUCrypto::
create_cache(gmpz_array *cache, const mpz_t base) {
  PowmCache c(base, p, public_key_mod_bits, WINDOW_SIZE);
  cache->fromMPZArray(c.getCache(), c.size());
  assert(int(cache->elementSize * sizeof(cache->data[0]) * 8) == public_key_mod_bits);

  mp_montgomerize_gpu_nocopy(
    cache->numElements,
    cache->data, cache->data,
    nsq_dev, p_dev, pp_dev,
    cache->elementSize);
}

void GPUCrypto::
enable_gpu(bool enable) {
  gpu_enabled = enable;

  if (type == CRYPTO_TYPE_PRIVATE) {
    cout << "Running the verifier on ";
  } else if (type == CRYPTO_TYPE_PUBLIC) {
    cout << "Running the prover on ";
  } else {
    assert(false);
    return;
  }

  cout << (enable ? "GPU" : "CPU") << "." << endl;

  // Initialize state if need be.
  init_crypto_state();
}

bool GPUCrypto::
is_gpu_enabled() const {
  return gpu_enabled;
}

void GPUCrypto::
modexp(gmpz_array *rop, gmpz_array *exp, gmpz_array *powm_cache) {
  assert(rop->numElements <= exp->numElements);
  assert(rop->elementSize == exp->elementSize);
  assert(rop->elementSize == powm_cache->elementSize);

  // Use rop to get array parameters (numElements, elementSize). base should
  // not be trusted since it isn't being written to (and can therefore include a
  // larger array that's been cached).
  mp_modexp_cached_gpu_nocopy(
    rop->numElements, rop->data, exp->data,
    powm_cache->data,
    nsq_dev, p_dev, pp_dev,
    rop->elementSize);
}

#ifndef SIMUL_EXP_TRICK
void GPUCrypto::
modexp(gmpz_array *rop, gmpz_array *base, struct mp_sw *sw_dev) {
  assert(rop->numElements <= base->numElements);
  assert(rop->elementSize == base->elementSize);

  // Use rop to get array parameters (numElements, elementSize). base should
  // not be trusted since it isn't being written to (and can therefore include a
  // larger array that's been cached).
  mp_many_modexp_mont_gpu_nocopy(
    rop->numElements,
    rop->data, base->data, sw_dev,
    nsq_dev, p_dev, pp_dev,
    rop->elementSize);
}

#else
void GPUCrypto::
multi_modexp(gmpz_array *rop, gmpz_array *base, struct mp_multi_sw *multi_sw_dev) {
  assert(rop->numElements == div_roundup(base->numElements, MP_MAX_MULTI_EXP));
  assert(rop->elementSize == base->elementSize);
  //assert(rop->elementSize == public_key_num_limbs);

  // Use rop to get array parameters (numElements, elementSize). base should
  // not be trusted since it isn't being written to (and can therefore include a
  // larger array that's been cached).
  mp_vec_multi_modexp(
    rop->numElements,
    rop->data, base->data, multi_sw_dev,
    nsq_dev, p_dev, pp_dev,
    rop->elementSize);
}
#endif

#ifndef SIMUL_EXP_TRICK
static void
build_sw(struct mp_sw *sw_dev, gmpz_array *exp, int exp_bits) {
  struct mp_sw sw[exp->numElements];
  for (int i=0; i < exp->numElements; i++) {
    mp_get_sw(&sw[i], exp->getElemHost(i), exp_bits / 8 / sizeof(WORD));
  }

  checkCudaErrors(cudaMemcpy(sw_dev, sw, sizeof(struct mp_sw) * exp->numElements, cudaMemcpyHostToDevice));
}

#else
struct sw_state {
  int base;
  int fragment;
  int nextFragLoc;
};

static bool
sw_state_more(sw_state a, sw_state b) {
  return a.nextFragLoc > b.nextFragLoc;
}

static void
mp_get_multi_sw(struct mp_multi_sw *multi_sw, const WORD *bases, int num_bases, int num_limbs) {
  assert(num_bases > 0 && num_bases <= MP_MAX_MULTI_EXP);
  memset(multi_sw, 0, sizeof(*multi_sw));

  vector<sw_state> expStates;
  mp_sw sw[num_bases];
  int exp_bits = num_limbs * BITS_PER_WORD;

  multi_sw->num_bases = num_bases;
  multi_sw->num_fragments = 0;
  for (int i = 0; i < num_bases; i++) {
    mp_get_sw(&sw[i], &bases[i * num_limbs], num_limbs);
    multi_sw->num_fragments += sw[i].num_fragments;
    multi_sw->max_fragment[i] = sw[i].max_fragment;

    sw_state curState = { i, 0, 0 };
    for (int frag = sw[i].num_fragments - 1; frag >= 0; frag--) {
      curState.fragment = sw[i].fragment[frag];
      curState.nextFragLoc += sw[i].length[frag];
      expStates.push_back(curState);
    }
  }
  assert((int) expStates.size() == multi_sw->num_fragments);

  // Use sort to do the hard work even though algorithmically, we don't have to
  // incur the nlogn cost.
  sort(expStates.begin(), expStates.end(), sw_state_more);
  int curFragLoc = 0;
  for (int i = (int) expStates.size() - 1; i >= 0; i--) {
    int base = multi_sw->base[i] = expStates[i].base;
    multi_sw->fragment[i] = expStates[i].fragment;
    multi_sw->length[i] = expStates[i].nextFragLoc - curFragLoc;
    curFragLoc = expStates[i].nextFragLoc;
  }

#if ASSERT_LEVEL >= AL_TESTED
  if (num_bases == 1) {
    assert(multi_sw->num_fragments == sw[0].num_fragments);
    assert(multi_sw->max_fragment[0] == sw[0].max_fragment);
    for (int i = 0; i < multi_sw->num_fragments; i++) {
      assert(multi_sw->base[i] == 0);
      assert(multi_sw->length[i] == sw[0].length[i]);
      assert(multi_sw->fragment[i] == sw[0].fragment[i]);
    }
  } else {
    int curIdx[num_bases];
    int sinceLast[num_bases];
    memset(sinceLast, 0, sizeof(*sinceLast) * num_bases);

    for (int i = 0; i < num_bases; i++)
      curIdx[i] = sw[i].num_fragments - 1;

    for (int i = multi_sw->num_fragments - 1; i >= 0; i--) {
      //cout << "[" << multi_sw->base[i] << "] " << multi_sw->length[i] << " " << multi_sw->fragment[i] << endl;
      int base = multi_sw->base[i];
      assert(base >= 0 && base < num_bases);
      assert(curIdx[base] < sw[base].num_fragments);
      assert(multi_sw->length[i] + sinceLast[base] == sw[base].length[curIdx[base]]);
      assert(multi_sw->fragment[i] == sw[base].fragment[curIdx[base]]);
      curIdx[base]--;

      for (int k = 0; k < num_bases; k++)
        sinceLast[k] = (k != base) ? sinceLast[k] + multi_sw->length[i] : 0;
    }

    for (int i = 0; i < num_bases; i++)
      assert(curIdx[i] == -1);

    //cout << num_bases << endl;
    for (int i = 0; i < num_bases; i++) {
      int sum = 0;
      for (int k = 0; k < sw[i].num_fragments; k++)
        sum += sw[i].length[k];
      assert(sum == exp_bits || (sum == 0 && sw[i].num_fragments == 0));
    }
    int sum = 0;
    for (int k = 0; k < multi_sw->num_fragments; k++)
      sum += multi_sw->length[k];
    assert(sum == exp_bits || (sum == 0 && multi_sw->num_fragments == 0));
  }
#endif
}

static void
build_multi_sw(struct mp_multi_sw *multi_sw_dev, const mpz_t *bases, int size, int num_multi) {
  assert(num_multi > 0 && num_multi <= MP_MAX_MULTI_EXP);
  int num_multi_modexp = div_roundup(size, num_multi);
  struct mp_multi_sw multi_sw[num_multi_modexp];

  int numLimbs = compute_num_limbs(bases, size);
  WORD *rawBases = new WORD[numLimbs * num_multi * size];
  to_gpu_format(rawBases, bases, size, numLimbs);

  for (int i = 0; i < num_multi_modexp; i++) {
    int num_bases = min(num_multi, size - i * num_multi);
    mp_get_multi_sw(&multi_sw[i], &rawBases[i * num_multi * numLimbs], num_bases, numLimbs);
  }

  checkCudaErrors(cudaMemcpy(multi_sw_dev,
                           multi_sw,
                           sizeof(*multi_sw) * num_multi_modexp,
                           cudaMemcpyHostToDevice));

  delete[] rawBases;
}
#endif

void GPUCrypto::
elgamal_enc(mpz_t c1, mpz_t c2, mpz_t plain) {
  // calls the CPU version
  Crypto::elgamal_enc(c1, c2, plain);
}

//#define DBG(n, name) cout << n << ": " << sw[0].name << ", " << sw[1].name << endl;

void GPUCrypto::
elgamal_enc_vec(mpz_t *cipher, mpz_t *plain, int size) {

  if (!is_gpu_enabled()) {
    Crypto::elgamal_enc_vec(cipher, plain, size);
    return;
  }

  Measurement m;
  m.begin_with_init();
  for (int i = 0; i < size; i += maxModexpInstances) {
    int vec_size = min(maxModexpInstances, size - i);

    assert(vec_size <= maxModexpInstances);
    rop.setSize(vec_size);

    // Compute g^{plain}
    exp.fromMPZArray(&plain[i], vec_size);

    modexp(&rop, &exp, &g_cache);
    //build_sw(sw_dev, &exp, public_key_mod_bits);
    //modexp(&rop, &g_array, sw_dev);

    // While the modexp happens, we generate random values.
    for (int j = 0; j < vec_size; j++) {
      // select a random number of size PRIV_KEY_BITS
      // priv_rand_bits should fit in one gmpz_array element.
      prng_private->get_randomb((char*)exp.getElemHost(j), elgamal_priv_rand_bits);
    }

    // Load result of g^{plain}. This will block until the modexp completes.
    rop.toMPZArray(&cipher[2 * i + 1], 2);

    // Compute gr^{rand}
    exp.writeToDevice();
    modexp(&rop, &exp, &gr_cache);
    //build_sw(sw_dev, &exp, elgamal_priv_rand_bits);
    //modexp(&rop, &gr_array, sw_dev);
    rop.toMPZArray(&cipher[2 * i], 2);

    // Compute g^{rand}
    modexp(&rop, &exp, &g_cache);
    //modexp(&rop, &g_array, sw_dev);

    // While the modexp happens, we compute gr^{rand} * g^{plain} mod p
    for (int j = 0; j < vec_size; j++) {
      mpz_mul(cipher[2 * (i + j) + 1],
              cipher[2 * (i + j) + 1],
              cipher[2 * (i + j)]);
      mpz_mod(cipher[2 * (i + j) + 1], cipher[2 * (i + j) + 1], p);
    }

    // Load result of g^{rand}
    rop.toMPZArray(&cipher[2 * i], 2);
  }
  m.end();
  cout<<"Time taken per exponentiation = "<<m.get_papi_elapsed_time()/(3*size)<<" usec"<<endl;
  cout<<"#exps/sec = "<<(3*size*1000.0*1000)/m.get_papi_elapsed_time()<<endl;
}

#ifndef SIMUL_EXP_TRICK
void GPUCrypto::
dot_product_enc_nomulti(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2) {
  const int maxCiphertextVecSize = maxModexpInstances;

  for (uint32_t i = 0; i < size; i += maxCiphertextVecSize) {
    int vec_size = min(maxCiphertextVecSize, (int)(size - i));

    assert(vec_size <= maxModexpInstances);
    rop.setSize(vec_size);

    // Compute q^{d}
    exp.fromMPZArray(&d[i], vec_size);

    // TODO: OPTIMIZE: remove this hardcoded 192 to use computation-provided exponent values
    // VV: Actually, this might not be necessary, since the sliding window
    // algorithm should ignore the most significant zeroes anyways.
    build_sw(sw_dev, &exp, 192);

    for (int comp = 0; comp < expansion_factor; comp++) {
      base.fromMPZArray(&q[expansion_factor * i + comp],
                        vec_size,
                        expansion_factor,
                        true);
      modexp(&rop, &base, sw_dev);

      // Load result of q^{d}.
      // TODO: This blocks until the modexp is done. Consider moving the reduce
      // step above this (would require some extra indexing trickery).
      rop.toMPZArray(result);

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
}
#else
void GPUCrypto::
dot_product_enc_multi(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2) {
  const int maxCiphertextVecSize = maxModexpInstances * MP_MAX_MULTI_EXP;

  for (uint32_t i = 0; i < size; i += maxCiphertextVecSize) {
    int cipher_vec_size = min(maxCiphertextVecSize, (int)(size - i));
    int rop_vec_size = div_roundup(cipher_vec_size, MP_MAX_MULTI_EXP);

    assert(rop_vec_size <= maxModexpInstances);
    rop.setSize(rop_vec_size);

    build_multi_sw(multi_sw_dev, &d[i], cipher_vec_size, MP_MAX_MULTI_EXP);

    for (int comp = 0; comp < expansion_factor; comp++) {
      base.fromMPZArray(&q[expansion_factor * i + comp],
                        cipher_vec_size,
                        expansion_factor,
                        true);
      multi_modexp(&rop, &base, multi_sw_dev);

      // Load result of q^{d}.
      // TODO: This blocks until the modexp is done. Consider moving the reduce
      // step above this (would require some extra indexing trickery).
      rop.toMPZArray(result);

#if ASSERT_LEVEL >= AL_DEBUG
      MPZVector r2(rop_vec_size);
      MPZVector ttt(1);
      for (int k = 0; k < rop_vec_size; k++) {
        mpz_set_ui(r2[k], 1);
        for (int l = 0; l < min(MP_MAX_MULTI_EXP, cipher_vec_size - k * MP_MAX_MULTI_EXP); l++) {
          mpz_powm(ttt[0],
                   q[expansion_factor * (i + k * MP_MAX_MULTI_EXP + l) + comp],
                   d[i + k * MP_MAX_MULTI_EXP + l],
                   p);
          mpz_mul(r2[k], r2[k], ttt[0]);
          mpz_mod(r2[k], r2[k], p);
        }

        if (mpz_cmp(r2[k], result[k]) != 0) {
          cout << "ERROR: " << k << endl;
          for (int l = 0; l < min(MP_MAX_MULTI_EXP, cipher_vec_size - k * MP_MAX_MULTI_EXP); l++)
            gmp_printf("d[%d]: %Zd\n", l, d[i + k * MP_MAX_MULTI_EXP + l]);
        }
      }
#endif

      for (int j = 0; j < rop_vec_size; j++) {
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
}
#endif

void GPUCrypto::
dot_product_enc(uint32_t size, mpz_t *q, mpz_t *d, mpz_t output, mpz_t output2) {
  if (!is_gpu_enabled()) {
    Crypto::dot_product_enc(size, q, d, output, output2);
    return;
  }

  if (crypto_in_use != CRYPTO_ELGAMAL) {
    cerr << "GPUCrypto only supports CRYPTO_ELGAMAL." << endl;
    return;
  }

  mpz_set_ui(output, 1);
  mpz_set_ui(output2, 1);
  Measurement m;
  m.begin_with_init();

#ifdef SIMUL_EXP_TRICK
  dot_product_enc_multi(size, q, d, output, output2);
#else
  dot_product_enc_nomulti(size, q, d, output, output2);
#endif

  m.end();
  cout<<"Time taken per exponentiation = "<<m.get_papi_elapsed_time()/(2*size)<<" usec"<<endl;
  cout<<"#exps/sec = "<<(2*size*1000.0*1000)/m.get_papi_elapsed_time()<<endl;
}
