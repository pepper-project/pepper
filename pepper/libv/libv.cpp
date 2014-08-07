#include <libv/libv.h>

#if USE_GPU == 1
#include <crypto/gpu_crypto.h>
#endif

#if USE_GMP == 0
#include <crypto/cpu_crypto.h>
#endif

Venezia::Venezia(int role, int crypto_in_use, int png_in_use, int field_mod_size) {
  crypto =
#if USE_GPU == 1
    new GPUCrypto(get_crypto_type(role), crypto_in_use, png_in_use, false, field_mod_size);
#else
#if USE_GMP == 1
    new Crypto(get_crypto_type(role), crypto_in_use, png_in_use, false, field_mod_size);
#else
    new CPUCrypto(get_crypto_type(role), crypto_in_use, png_in_use, false, field_mod_size);
#endif
#endif

  init_venezia_state();
}

Venezia::Venezia(int role, int crypto_in_use, int png_in_use, int field_mod_size,
                 int public_key_mod_size) {
  crypto =
#if USE_GPU == 1
    new GPUCrypto(get_crypto_type(role), crypto_in_use, png_in_use, false, field_mod_size,
                  public_key_mod_size);
#else
#if USE_GMP == 1
    new Crypto(get_crypto_type(role), crypto_in_use, png_in_use, false, field_mod_size,
               public_key_mod_size);
#else
    new Crypto(get_crypto_type(role), crypto_in_use, png_in_use, false, field_mod_size,
               public_key_mod_size);
#endif
#endif

  init_venezia_state();
}

Venezia::Venezia(int role, int crypto_in_use, int png_in_use, int field_mod_size,
                 int public_key_mod_size, int elgamal_priv_size) {
  crypto =
#if USE_GPU == 1
    new GPUCrypto(get_crypto_type(role), crypto_in_use, png_in_use, false, field_mod_size,
                  public_key_mod_size, elgamal_priv_size);
#else
#if USE_GMP == 1
    new Crypto(get_crypto_type(role), crypto_in_use, png_in_use, false, field_mod_size,
               public_key_mod_size, elgamal_priv_size);
#else
    new Crypto(get_crypto_type(role), crypto_in_use, png_in_use, false, field_mod_size,
               public_key_mod_size, elgamal_priv_size);
#endif
#endif
  init_venezia_state();
}

Venezia::~Venezia() {
  clear_scalar(temp1);
  clear_scalar(temp2);
  clear_scalar(temp3);
  clear_scalar(temp4);
  clear_scalar(gen);
  clear_scalar(pub_mod);
  clear_scalar(order);

  delete crypto;
#if FAST_FOURIER_INTERPOLATION == 1
  clear_scalar(omega);
  clear_vec(size_omega_set, powers_of_omega);
#endif
}

int Venezia::get_crypto_type(int role) {
  if (role == ROLE_VERIFIER)
    return CRYPTO_TYPE_PRIVATE;
  else
    return CRYPTO_TYPE_PUBLIC;
}

void Venezia::init_venezia_state() {
  mpz_init(temp1);
  mpz_init(temp2);
  mpz_init(temp3);
  mpz_init(temp4);
  mpz_init(gen);
  mpz_init(pub_mod);
  mpz_init(order);
  crypto->elgamal_get_generator(&gen);
  crypto->elgamal_get_public_modulus(&pub_mod);
  crypto->elgamal_get_order(&order);
  mpz_init(omega);
}

void Venezia::create_commitment_query(uint32_t size, mpz_t * r_q,
                                      mpz_t * con_q, mpz_t prime) {
#ifndef DEBUG_MALICIOUS_PROVER
  for (uint32_t i = 0; i < size; i++) {
    crypto->get_random_priv(con_q[i], prime);
  }

  crypto->elgamal_enc_vec(r_q, con_q, size);
#else
  for (uint32_t i = 0; i < 2*size; i++) {
    if (i < size)
      crypto->get_random_priv(con_q[i], prime);
    crypto->get_random_priv(r_q[i], pub_mod);
  }
#endif
}

void Venezia::outer_product(int size_1, mpz_t *q1, int size_2, mpz_t *q2, mpz_t *mask, mpz_t prime, mpz_t *out) {
  int index = 0;
  for (int i = 0; i < size_1; i++) {
    for (int j = 0; j < size_2; j++) {
      mpz_mul(out[index], q1[i], q2[j]);
      mpz_add(out[index], out[index], mask[index]);
      mpz_mod(out[index], out[index], prime);
      index++;
    }
  }
}

void Venezia::compute_set_v(int size, mpz_t *set_v, mpz_t omega, mpz_t prime) {
  mpz_t temp, inv_of_omega_c_1;
  alloc_init_scalar(temp);
  alloc_init_scalar(inv_of_omega_c_1);

  // first compute v_0
  mpz_set_ui(set_v[0], 1);
  for (int i=1; i<size; i++) {
    mpz_sub(temp, powers_of_omega[0], powers_of_omega[i]);
    mpz_mul(set_v[0], set_v[0], temp);
    mpz_mod(set_v[0], set_v[0], prime);
  }
  mpz_invert(set_v[0], set_v[0], prime);
  mpz_invert(inv_of_omega_c_1, powers_of_omega[size-1], prime);

  // next compute the remaining values in the set
  for (int j=0; j<size-1; j++) {
    mpz_mul(set_v[j+1], set_v[j], inv_of_omega_c_1);
    mpz_mod(set_v[j+1], set_v[j+1], prime);
  }
  
  clear_scalar(temp);
  clear_scalar(inv_of_omega_c_1);
}

void Venezia::compute_set_v(int size, mpz_t *set_v, mpz_t prime) {
  // v_j = 1 / \Pi_{0 <= k <= \chi AND k != j} {(\alpha_j - \alpha_k)}
  // and \alpha_i = i
  mpz_t temp;
  alloc_init_scalar(temp);

  // first compute v_0
  mpz_set_ui(set_v[0], 1);
  for (int i=1; i<size; i++) {
    mpz_set_ui(temp, i);
    mpz_neg(temp, temp);
    mpz_mul(set_v[0], set_v[0], temp);
    mpz_mod(set_v[0], set_v[0], prime);
  }
  mpz_invert(set_v[0], set_v[0], prime);

  // next compute the remaining values in the set
  for (int j=1; j<size; j++) {
    mpz_set(set_v[j], set_v[j-1]);

    // divide by (j-0)
    mpz_set_ui(temp, j);
    mpz_invert(temp, temp, prime);
    mpz_mul(set_v[j], set_v[j], temp);
    mpz_mod(set_v[j], set_v[j], prime);

    // multiply by (j-1 - \size-1))
    mpz_set_si(temp, j-size);
    mpz_mul(set_v[j], set_v[j], temp);
    mpz_mod(set_v[j], set_v[j], prime);
  }
  clear_scalar(temp);
}

// creation of PCP queries

void Venezia::update_con_query(mpz_t con, mpz_t beta, mpz_t q, mpz_t prime) {
  mpz_mul(temp1, beta, q);
  mpz_add(con, con, temp1);
  // if (crypto->get_crypto_in_use() != CRYPTO_ELGAMAL)
  // mpz_mod(con, con, prime);
}

void Venezia::update_con_query_vec(uint32_t size, mpz_t * con, mpz_t beta,
                                   mpz_t * q, mpz_t prime) {
  for (uint32_t i = 0; i < size; i++)
    update_con_query(con[i], beta, q[i], prime);
}

void Venezia::create_lin_test_queries(uint32_t size, mpz_t * l1,
                                      mpz_t * l2, mpz_t * l3, mpz_t * con,
                                      int filled, mpz_t * con_coins,
                                      mpz_t prime) {
  
  mpz_t temp1, temp2;
  if (con_coins != NULL) {
    crypto->get_random_priv(con_coins[filled + 1], prime);
    crypto->get_random_priv(con_coins[filled + 2], prime);
    crypto->get_random_priv(con_coins[filled + 3], prime);

    alloc_init_scalar(temp1);
    alloc_init_scalar(temp2);

    // this consistency query creation can be simplified because l3 = l1 + l2
    mpz_add(temp1, con_coins[filled+1], con_coins[filled+3]);
    mpz_add(temp2, con_coins[filled+2], con_coins[filled+3]);
  }

  for (uint32_t i = 0; i < size; i++) {
    crypto->get_random_pub(l1[i], prime);
    crypto->get_random_pub(l2[i], prime);

    // l3[i] = l1[i] + l2[i] % prime
    // We don't really need to realize the third linearity test query
    // since consistency test query can be computed without it
    // mpz_add(l3[i], l1[i], l2[i]);

    // con[i] = con[i] + \con_coins_1 l1[i] + \con_coins_2 \l2[i]
    // \con_coins_3 l3[i]
    if (con != NULL) {
      update_con_query(con[i], temp1, l1[i], prime);
      update_con_query(con[i], temp2, l2[i], prime);
    
      // The next step can be eliminated with the above hack that accounts
      // for incorporating component of l3 in consistency query
      //update_con_query(con[i], con_coins[filled + 3], l3[i], prime);
    }
  }
  
  if (con_coins != NULL) {
    clear_scalar(temp1);
    clear_scalar(temp2);
  }
}


// general version that takes non-symmetric matrices
void Venezia::create_corr_test_queries_vproduct2(uint32_t m, uint32_t n,
    mpz_t * f1_q1,
    mpz_t * f1_q2,
    mpz_t * f_q1, mpz_t * f_q2,
    mpz_t * con, int filled,
    mpz_t * con_coins,
    mpz_t prime) {

  if (con_coins != NULL) {
    crypto->get_random_priv(con_coins[filled + 1], prime);
    crypto->get_random_priv(con_coins[filled + 2], prime);
  }

  for (uint32_t i = 0; i < m * n; i++) {
    crypto->get_random_pub(f1_q1[i], prime);
    crypto->get_random_pub(f1_q2[i], prime);
  }


  int index;
  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < m; j++) {
      for (uint32_t k = 0; k < n; k++) {
        index = (i*m+j)*n+k;
        crypto->get_random_pub(f_q2[index], prime);

        mpz_mul(f_q1[index], f1_q1[i * n + k], f1_q2[k * m + j]);
        mpz_add(f_q1[index], f_q1[index], f_q2[index]);

        // if (crypto->get_crypto_in_use() != CRYPTO_ELGAMAL)
        mpz_mod(f_q1[index], f_q1[index], prime);

        if (con != NULL && con_coins != NULL) {
          update_con_query(con[index], con_coins[filled + 1], f_q1[index],
                           prime);
          update_con_query(con[index], con_coins[filled + 2], f_q2[index],
                           prime);
        }
      }
    }
  }
}

// assumes that the two matrices involved in the vproduct correction
// test are of the same size
void Venezia::create_corr_test_queries_vproduct(uint32_t m, mpz_t * f1_q1,
    mpz_t * f1_q2,
    mpz_t * f_q1, mpz_t * f_q2,
    mpz_t * con, int filled,
    mpz_t * con_coins,
    mpz_t prime) {

  if (con_coins != NULL) {
    crypto->get_random_priv(con_coins[filled + 1], prime);
    //crypto->get_random_priv(con_coins[filled + 2], prime);
  }

  for (uint32_t i = 0; i < m * m; i++) {
    crypto->get_random_pub(f1_q1[i], prime);
    crypto->get_random_pub(f1_q2[i], prime);
  }

  int index = 0;

  for (uint32_t i = 0; i < m; i++) {
    for (uint32_t j = 0; j < m; j++) {
      for (uint32_t k = 0; k < m; k++) {
        //crypto->get_random_priv(f_q2[index], prime);

        mpz_mul(f_q1[index], f1_q1[i * m + k], f1_q2[k * m + j]);
        mpz_add(f_q1[index], f_q1[index], f_q2[index]);

        // if (crypto->get_crypto_in_use() != CRYPTO_ELGAMAL)
        mpz_mod(f_q1[index], f_q1[index], prime);

        if (con != NULL && con_coins != NULL) {
          update_con_query(con[index], con_coins[filled + 1], f_q1[index],
                           prime);
          //update_con_query(con[index], con_coins[filled + 2], f_q2[index],
          //                 prime);
        }
        index++;
      }
    }
  }
}

void Venezia::create_zself_corr_queries(uint32_t size, uint32_t size_1, uint32_t size_2,
                                        mpz_t *q,
                                        mpz_t *c_coins1, int fill1, mpz_t *con_q1,
                                        mpz_t *c_coins2, int fill2, mpz_t *con_q2,
                                        mpz_t prime) {
  if (c_coins1 != NULL)
    crypto->get_random_priv(c_coins1[fill1 + 1], prime);

  if (c_coins2 != NULL)
    crypto->get_random_priv(c_coins2[fill2 + 1], prime);

  this->get_random_vec_pub(size, q, prime);

  for (uint32_t i = 0; i < size_1; i++) {
    if (con_q1 != NULL && c_coins1 != NULL)
      update_con_query(con_q1[i], c_coins1[fill1 + 1], q[i], prime);
  }

  for (uint32_t i = 0; i < size_2; i++) {
    if (con_q2 != NULL && c_coins2 != NULL)
      update_con_query(con_q2[i], c_coins2[fill2 + 1], q[i], prime);
  }
}

void Venezia::evaluate_polynomial_at_random_point(
    int n, int size_2,
    mpz_t tau, mpz_t d_star,
    int num_aij, int num_bij, int num_cij,
    mpz_t *set_v,
    poly_compressed *poly_A, poly_compressed *poly_B, poly_compressed *poly_C,
    mpz_t *eval_poly_A, mpz_t *eval_poly_B, mpz_t *eval_poly_C,
    mpz_t prime) {

  mpz_t temp, l_tau;
  alloc_init_scalar(temp);
  alloc_init_scalar(l_tau);

  crypto->get_random_pub(tau, prime);

  // fill in l_tau = l(tau) = (tau - 0) * (tau - 1) * ... (tau - \chi)
  // size_2 is \chi+1
  mpz_set_ui(l_tau, 1);
  #if FAST_FOURIER_INTERPOLATION == 1
  for (int i=0; i<size_2; i++) {
    mpz_sub(temp, tau, powers_of_omega[i]);
    
    if(mpz_cmp_ui(temp, 0) == 0)
      exit(1);

    mpz_mul(l_tau, l_tau, temp);
    mpz_mod(l_tau, l_tau, prime);
  }
  #else
  for (int i=0; i<size_2; i++) {
    mpz_sub_ui(temp, tau, i);
    mpz_mul(l_tau, l_tau, temp);
    mpz_mod(l_tau, l_tau, prime);
  }
  #endif

  // next fill in eval_poly_A
  for (int i=0; i<num_aij; i++) {
    mpz_mul(temp, set_v[poly_A[i].j], poly_A[i].coefficient);
    mpz_mod(temp, temp, prime);

    #if FAST_FOURIER_INTERPOLATION == 1
    mpz_sub(temp2, tau, powers_of_omega[poly_A[i].j]);
    #else
    mpz_sub_ui(temp2, tau, poly_A[i].j);
    #endif
    mpz_invert(temp2, temp2, prime);

    mpz_mul(temp, temp, temp2);
    mpz_add(eval_poly_A[poly_A[i].i], eval_poly_A[poly_A[i].i], temp);
    mpz_mod(eval_poly_A[poly_A[i].i], eval_poly_A[poly_A[i].i], prime);
  }

  for (int i=0; i<n+1; i++) {
    mpz_mul(eval_poly_A[i], eval_poly_A[i], l_tau);
    mpz_mod(eval_poly_A[i], eval_poly_A[i], prime);
  }

  // fill in eval_poly_B
  for (int i=0; i<num_bij; i++) {
    mpz_mul(temp, set_v[poly_B[i].j], poly_B[i].coefficient);
    mpz_mod(temp, temp, prime);

    #if FAST_FOURIER_INTERPOLATION == 1
    mpz_sub(temp2, tau, powers_of_omega[poly_B[i].j]);
    #else
    mpz_sub_ui(temp2, tau, poly_B[i].j);
    #endif
    mpz_invert(temp2, temp2, prime);

    mpz_mul(temp, temp, temp2);
    mpz_add(eval_poly_B[poly_B[i].i], eval_poly_B[poly_B[i].i], temp);
    mpz_mod(eval_poly_B[poly_B[i].i], eval_poly_B[poly_B[i].i], prime);
  }

  for (int i=0; i<n+1; i++) {
    mpz_mul(eval_poly_B[i], eval_poly_B[i], l_tau);
    mpz_mod(eval_poly_B[i], eval_poly_B[i], prime);
  }

  // fill in eval_poly_C
  for (int i=0; i<num_cij; i++) {
    //assert(poly_C[i].j >= 0 && poly_C[i].j <size_2);

    mpz_mul(temp, set_v[poly_C[i].j], poly_C[i].coefficient);
    mpz_mod(temp, temp, prime);


    #if FAST_FOURIER_INTERPOLATION == 1
    mpz_sub(temp2, tau, powers_of_omega[poly_C[i].j]);
    #else
    mpz_sub_ui(temp2, tau, poly_C[i].j);
    #endif    
    mpz_invert(temp2, temp2, prime);

    mpz_mul(temp, temp, temp2);
    mpz_add(eval_poly_C[poly_C[i].i], eval_poly_C[poly_C[i].i], temp);
    mpz_mod(eval_poly_C[poly_C[i].i], eval_poly_C[poly_C[i].i], prime);
  }

  for (int i=0; i<n+1; i++) {
    mpz_mul(eval_poly_C[i], eval_poly_C[i], l_tau);
    mpz_mod(eval_poly_C[i], eval_poly_C[i], prime);
  }

  // also calculate D(r_star) and store in d_star
  if (d_star != NULL) {
    #if FAST_FOURIER_INTERPOLATION == 1
      mpz_sub_ui(temp, tau, 1);
    #else
      mpz_set(temp, tau);
    #endif
    mpz_invert(d_star, temp, prime);
    mpz_mul(d_star, d_star, l_tau);
    mpz_mod(d_star, d_star, prime);
  }
  clear_scalar(temp);
  clear_scalar(l_tau);
}

void Venezia::create_div_corr_test_queries(int n, uint32_t size_1, uint32_t size_2,
    mpz_t *f1_q1, mpz_t *f1_q2, mpz_t *f1_q3,
    mpz_t *f2_q1,
    mpz_t *c_coins1, int fill1, mpz_t *con_q1,
    mpz_t *c_coins2, int fill2, mpz_t *con_q2,
    mpz_t *corr_q, mpz_t *corr_q2,
    mpz_t d_star,
    int num_aij, int num_bij, int num_cij,
    mpz_t *set_v,
    poly_compressed *poly_A, poly_compressed *poly_B, poly_compressed *poly_C,
    mpz_t *eval_poly_A, mpz_t *eval_poly_B, mpz_t *eval_poly_C,
    mpz_t prime) {

  if (c_coins1 != NULL) {
    crypto->get_random_priv(c_coins1[fill1 + 1], prime);
    crypto->get_random_priv(c_coins1[fill1 + 2], prime);
    crypto->get_random_priv(c_coins1[fill1 + 3], prime);
  }

  if (c_coins2 != NULL)
    crypto->get_random_priv(c_coins2[fill2 + 1], prime);
  
  mpz_t tau;
  alloc_init_scalar(tau);

  evaluate_polynomial_at_random_point(
      n, size_2,
      tau, d_star,
      num_aij, num_bij, num_cij,
      set_v,
      poly_A, poly_B, poly_C,
      eval_poly_A, eval_poly_B, eval_poly_C,
      prime);

  // first, compute q_d
  for (uint32_t i=0; i<size_2; i++) {
    if (i > 0) {
      mpz_mul(f2_q1[i], f2_q1[i-1], tau);
      mpz_mod(f2_q1[i], f2_q1[i], prime);
    } else
      mpz_set_ui(f2_q1[i], 1);
  }

  // add self-correction to all queries
  for (uint32_t i=0; i<size_1; i++) {
    mpz_add(f1_q1[i], eval_poly_A[i+1], corr_q[i]);
    mpz_mod(f1_q1[i], f1_q1[i], prime);

    mpz_add(f1_q2[i], eval_poly_B[i+1], corr_q[i]);
    mpz_mod(f1_q2[i], f1_q2[i], prime);

    mpz_add(f1_q3[i], eval_poly_C[i+1], corr_q[i]);
    mpz_mod(f1_q3[i], f1_q3[i], prime);
  }

  for (uint32_t i=0; i<size_2; i++) {
    mpz_add(f2_q1[i], f2_q1[i], corr_q2[i]);
    mpz_mod(f2_q1[i], f2_q1[i], prime);
  }

  // next, add them to consistency query
  for (uint32_t i = 0; i < size_1; i++) {
    if (con_q1 != NULL && c_coins1 != NULL) {
      update_con_query(con_q1[i], c_coins1[fill1 + 1], f1_q1[i], prime);
      update_con_query(con_q1[i], c_coins1[fill1 + 2], f1_q2[i], prime);
      update_con_query(con_q1[i], c_coins1[fill1 + 3], f1_q3[i], prime);
    }
  }

  for (uint32_t i = 0; i < size_2; i++) {
    if (con_q2 != NULL && c_coins2 != NULL)
      update_con_query(con_q2[i], c_coins2[fill2 + 1], f2_q1[i], prime);
  }
  clear_scalar(tau);
}

// TODO: simplify the signature for this function; too many arguments
void Venezia::create_corr_test_queries_reuse(uint32_t size_1, mpz_t * c1_q,
    uint32_t size_2, mpz_t * c2_q,
    mpz_t * c3_q, mpz_t * r,
    mpz_t * con_q1, mpz_t * con_q2,
    mpz_t * con_q3, int filled1,
    mpz_t * con_coins1, int filled2,
    mpz_t * con_coins2, int filled3,
    mpz_t * con_coins3, mpz_t prime, bool opt) {
  
  if (con_coins3 != NULL) {
    crypto->get_random_priv(con_coins3[filled3 + 1], prime);
  }

  int index = 0;
  for (uint32_t i = 0; i < size_1; i++) {
    for (uint32_t j = 0; j < size_2; j++) {
      mpz_mul(c3_q[index], c1_q[i], c2_q[j]);
      mpz_add(c3_q[index], c3_q[index], r[index]);
      mpz_mod(c3_q[index], c3_q[index], prime);

      if (con_q3 != NULL && con_coins3 != NULL)
        update_con_query(con_q3[index], con_coins3[filled3 + 1], c3_q[index], prime);

      index++;
    }
  }
}

void Venezia::create_corr_test_queries(uint32_t size_1, mpz_t * c1_q,
                                       uint32_t size_2, mpz_t * c2_q,
                                       mpz_t * c3_q, mpz_t * r,
                                       mpz_t * con_q1, mpz_t * con_q2,
                                       mpz_t * con_q3, int filled1,
                                       mpz_t * con_coins1, int filled2,
                                       mpz_t * con_coins2, int filled3,
                                       mpz_t * con_coins3, mpz_t prime, bool opt) {
  if (con_coins1 != NULL && !opt)
    crypto->get_random_priv(con_coins1[filled1 + 1], prime);

  if (con_coins2 != NULL && !opt)
    crypto->get_random_priv(con_coins2[filled2 + 1], prime);

  if (con_coins3 != NULL) {
    crypto->get_random_priv(con_coins3[filled3 + 1], prime);
    crypto->get_random_priv(con_coins3[filled3 + 2], prime);
  }

  for (uint32_t i = 0; i < size_1; i++) {
    if (!opt) {
      crypto->get_random_pub(c1_q[i], prime);
      if (con_q1!= NULL && con_coins1 != NULL)
        update_con_query(con_q1[i], con_coins1[filled1 + 1], c1_q[i], prime);
    }
  }

  for (uint32_t i = 0; i < size_2; i++) {
    if (!opt) {
      crypto->get_random_pub(c2_q[i], prime);
      if (con_q2 != NULL && con_coins2 != NULL)
        update_con_query(con_q2[i], con_coins2[filled2 + 1], c2_q[i], prime);
    }
  }

  int index = 0;

  for (uint32_t i = 0; i < size_1; i++) {
    for (uint32_t j = 0; j < size_2; j++) {
      crypto->get_random_pub(r[index], prime);
      mpz_mul(c3_q[index], c1_q[i], c2_q[j]);
      mpz_add(c3_q[index], c3_q[index], r[index]);

      // if (crypto->get_crypto_in_use() != CRYPTO_ELGAMAL)
      mpz_mod(c3_q[index], c3_q[index], prime);

      if (con_q3 != NULL && con_coins3 != NULL) {
        update_con_query(con_q3[index], con_coins3[filled3 + 1], c3_q[index],
                         prime);
        update_con_query(con_q3[index], con_coins3[filled3 + 2], r[index],
                         prime);
      }
      index++;
    }
  }
}

void Venezia::create_ckt_test_queries(uint32_t size, mpz_t * a,
                                      mpz_t * q_1, mpz_t * q_2,
                                      mpz_t * con_query, int filled,
                                      mpz_t * con_coins, mpz_t prime) {
  if (con_coins != NULL) {
    crypto->get_random_priv(con_coins[filled + 1], prime);
  }

  for (uint32_t i = 0; i < size; i++) {
    mpz_add(q_1[i], q_2[i], a[i]);

    // if (crypto->get_crypto_in_use() != CRYPTO_ELGAMAL)
    mpz_mod(q_1[i], q_1[i], prime);

    if (con_query != NULL && con_coins != NULL) {
      update_con_query(con_query[i], con_coins[filled + 1], q_1[i], prime);
    }
  }
}

bool Venezia::lin_test(mpz_t a1, mpz_t a2, mpz_t a3, mpz_t prime) {
  mpz_add(temp1, a1, a2);
  mpz_mod(temp1, temp1, prime);
  mpz_mod(temp2, a3, prime);
  int temp = mpz_cmp(temp1, temp2);

  if (temp == 0)
    return true;
  else
    return false;
}

bool Venezia::corr_test(mpz_t a1, mpz_t a2, mpz_t a3, mpz_t a4,
                        mpz_t prime) {
  mpz_mul(temp1, a1, a2);
  mpz_mod(temp1, temp1, prime);
  mpz_sub(temp2, a3, a4);
  mpz_mod(temp2, temp2, prime);
  int temp = mpz_cmp(temp1, temp2);

  if (temp == 0)
    return true;
  else
    return false;
}

bool Venezia::ckt_test(uint32_t size, mpz_t * arr, mpz_t c, mpz_t prime) {
  mpz_set(temp2, c);
  for (uint32_t i = 0; i < size / 2; i++) {
    mpz_sub(temp1, arr[2 * i], arr[2 * i + 1]);
    mpz_add(temp2, temp2, temp1);
  }
  mpz_mod(temp2, temp2, prime);
  if (mpz_cmp_ui(temp2, 0) == 0)
    return true;
  else
    return false;
}

void Venezia::elgamal_get_public_key_modulus(mpz_t *mod) {
  crypto->elgamal_get_public_modulus(mod);
}


bool Venezia::consistency_test(uint32_t size, mpz_t con_answer,
                               mpz_t com_answer, mpz_t * answers,
                               mpz_t * con_coins, mpz_t prime) {

  // compute b - <a, \alpha> without applying modp
  mpz_set_ui(temp1, 0);
  for (uint32_t i = 0; i < size; i++) {
    mpz_mul(temp2, con_coins[i], answers[i]);
    mpz_add(temp1, temp1, temp2);
  }

  mpz_sub(temp1, con_answer, temp1);

  // comparison with order of the group
  // checks if b - <a, \alpha> is between [0, q)
  if(mpz_cmp_si(temp1, 0) >= 0 && (mpz_cmp(temp1, order) < 0)) {
  } else {
    return false;
  }
  // compute g^{b - <a, \alpha>}
  //mpz_powm(temp1, gen, temp1, pub_mod);
  crypto->g_fpowm(temp2, temp1);

  // compare s with g^{b - <a, \alpha>}
  if (mpz_cmp(temp2, com_answer) == 0)
    return true;
  else
    return false;
}

void Venezia::dot_product(uint32_t size, mpz_t * q, mpz_t * d,
                          mpz_t output, mpz_t prime) {
  mpz_set_ui(output, 0);
  mpz_t temp;

  mpz_init(temp);

  for (uint32_t i = 0; i < size; i++) {
    mpz_mul(temp, d[i], q[i]);
    mpz_add(output, output, temp);
  }
  clear_scalar(temp);
}

void Venezia::dot_product_par(uint32_t size, mpz_t * q, mpz_t * d,
                              mpz_t output, mpz_t prime, double *par_time_so_far) {
  mpz_set_ui(output, 0);
  int num_threads = NUM_THREADS;
  int sizes[num_threads];
  int sum = 0;
  for (int i=0; i<num_threads-1; i++) {
    sizes[i] = size/num_threads;
    sum += sizes[i];
  }
  sizes[num_threads-1] = size - sum;

  Measurement m;
  m.begin_with_init();
  omp_set_num_threads(num_threads);
  #pragma omp parallel shared(output)
  {
    mpz_t temp, priv_output;
    mpz_init(temp);
    mpz_init_set_ui(priv_output, 0);

    int id = omp_get_thread_num();
    int base = 0;
    for (int i=0; i<id; i++)
      base = base + sizes[i];

    dot_product(sizes[id], &q[base], &d[base], priv_output, prime);
    #pragma omp critical
    {
      mpz_add(output, output, priv_output);
    }
    clear_scalar(temp);
    clear_scalar(priv_output);
  }
  m.end();
  *par_time_so_far += m.get_papi_elapsed_time();
}

void Venezia::add_sign(uint32_t size, mpz_t *vec) {
  #if GEN_SIGNED_INPUTS == 1 
  for (uint32_t i=0; i<size; i++) {
    if (rand()%2 == 0)
      mpz_mul_si(vec[i], vec[i], -1);
  }
  #endif
}

// interface to secure private prng
void Venezia::get_random_vec_priv(uint32_t size, mpz_t * vec, mpz_t n) {
  return crypto->get_random_vec_priv(size, vec, n);
}

void Venezia::get_random_vec_priv(uint32_t size, mpz_t * vec, int nbits) {
  return crypto->get_random_vec_priv(size, vec, nbits);
}

void Venezia::get_random_vec_priv(uint32_t size, mpq_t * vec, int nbits) {
  return crypto->get_random_vec_priv(size, vec, nbits);
}

void Venezia::get_random_priv(mpz_t x, mpz_t p) {
  crypto->get_random_priv(x, p);
}

// with nbits numerators and 2^k where k < nbits denominators.
void Venezia::get_random_rational_vec(uint32_t size, mpq_t *vec, int nbits) {
  for (unsigned i = 0; i < size; i++) {
    crypto->get_randomb_priv(mpq_numref(vec[i]), nbits);
    
    #if GEN_SIGNED_INPUTS == 1
    if (rand()%2 == 0)
      mpz_mul_si(mpq_numref(vec[i]), mpq_numref(vec[i]), -1);
    #endif

    mpz_set_ui(mpq_denref(vec[i]), 1);
    mpz_mul_2exp(mpq_denref(vec[i]), mpq_denref(vec[i]), rand() % nbits);

    mpq_canonicalize(vec[i]);
  }
}

// with n_a- bit numerators and 2^k where k <= n_b denominators.
void Venezia::get_random_rational_vec(uint32_t size, mpq_t *vec, int n_a, int n_b) {
  for (unsigned i = 0; i < size; i++) {
    crypto->get_randomb_priv(mpq_numref(vec[i]), n_a);

    #if GEN_SIGNED_INPUTS == 1
    if (rand()%2 == 0)
      mpz_mul_si(mpq_numref(vec[i]), mpq_numref(vec[i]), -1);
    #endif

    mpz_set_ui(mpq_denref(vec[i]), 1);
    mpz_mul_2exp(mpq_denref(vec[i]), mpq_denref(vec[i]), rand() % (n_b+1));

    mpq_canonicalize(vec[i]);
  }
}

// random Na bit signed integer (min value is -2^{na-1}, max value is  2^{na-1} - 1
void Venezia::get_random_signedint_vec(uint32_t size, mpq_t *vec, int na) {
  for (unsigned i = 0; i < size; i++) {
    crypto->get_randomb_priv(mpq_numref(vec[i]), na-1);

    //Use denominator as a temp int
    #if GEN_SIGNED_INPUTS == 1
    //Flip a coin, if heads, subtract 2^(na-1)
    if (rand()%2 == 0){
      mpz_set_ui(mpq_denref(vec[i]), 1);
      mpz_mul_2exp(mpq_denref(vec[i]), mpq_denref(vec[i]), na - 1);
      mpz_sub(mpq_numref(vec[i]), mpq_numref(vec[i]), mpq_denref(vec[i]));
    }
    #endif

    mpz_set_ui(mpq_denref(vec[i]), 1);

    mpq_canonicalize(vec[i]);
  }
}

// interface to public prng
void Venezia::get_random_vec_pub(uint32_t size, mpz_t * vec, mpz_t n) {
  return crypto->get_random_vec_pub(size, vec, n);
}

void Venezia::get_random_vec_pub(uint32_t size, mpz_t * vec, int nbits) {
  return crypto->get_random_vec_pub(size, vec, nbits);
}

void Venezia::get_random_vec_pub(uint32_t size, mpq_t * vec, int nbits) {
  return crypto->get_random_vec_pub(size, vec, nbits);
}

void Venezia::get_random_pub(mpz_t x, mpz_t p) {
  crypto->get_random_pub(x, p);
}

void Venezia::elgamal_dec(mpz_t plain, mpz_t c1, mpz_t c2) {
  return crypto->elgamal_dec(plain, c1, c2);
}

void Venezia::dot_product_enc(uint32_t size, mpz_t * q, mpz_t * d,
                              mpz_t output, mpz_t output2) {
  return crypto->dot_product_enc(size, q, d, output, output2);
}

void Venezia::add_enc(mpz_t res1, mpz_t res2, mpz_t c1_1, mpz_t c1_2, mpz_t c2_1, mpz_t c2_2) {
  return crypto->elgamal_hadd(res1, res2, c1_1, c1_2, c2_1, c2_2);
}

// TODO: avoid writting wrappers to crypto from libv.cpp
void Venezia::dump_seed_decommit_queries() {
  return crypto->dump_seed_decommit_queries();
}

void Venezia::init_prng_decommit_queries() {
  return crypto->init_prng_decommit_queries();
}

// this generates an nth root of unity deterministically because it uses a PRNG with a null seed
void Venezia::generate_root_of_unity(int n, mpz_t prime) {
  mpz_t x, prime_by_2, prime_by_n, temp;
  mpz_init(x);
  mpz_init(prime_by_2);
  mpz_init(prime_by_n);
  mpz_init(temp);

  // check if prime-1 is divisible by 2^64
  mpz_sub_ui(x, prime, 1);
  mpz_mod_2exp(x, x, 64);
  if(mpz_cmp_ui(x, 0) != 0) {
    cout<<"Generating a root of unity requires 2^64 divide (p-1)"<<endl;
    exit(1);
  }

  gmp_randstate_t r_state;
  gmp_randinit_default(r_state);
  gmp_randseed_ui(r_state, 0);

  // compute (p-1)/2
  mpz_sub_ui(prime_by_2, prime, 1);
  mpz_div_ui(prime_by_2, prime_by_2, 2);

  mpz_sub_ui(prime_by_n, prime, 1);
  mpz_div_ui(prime_by_n, prime_by_n, n);

  //cout<<"Generating a root of unity; make sure your prime is such that 2^64 divides p-1"<<endl;
  while(1) {
    mpz_urandomb(x, r_state, mpz_sizeinbase(prime, 2));
    mpz_mod(x, x, prime); 
    
    if (mpz_cmp_ui(x, 0) == 0)
      continue;
    
    mpz_powm(temp, x, prime_by_2, prime);
    if (mpz_cmp_ui(temp, 1) != 0)
      break;
  }

  mpz_powm(omega, x, prime_by_n, prime);

  mpz_powm_ui(temp, omega, n, prime);
  if (mpz_cmp_ui(temp, 1) != 0) {
    cout<<"Roots of unity generation failed"<<endl;
    exit(0);
  }
  
  size_omega_set = n;
  alloc_init_vec(&powers_of_omega, size_omega_set);
 
  // precompute powers of omega
  mpz_set_ui(powers_of_omega[0], 1);
  for (int i=1; i<size_omega_set; i++) {
    mpz_mul(powers_of_omega[i], powers_of_omega[i-1], omega);
    mpz_mod(powers_of_omega[i], powers_of_omega[i], prime);    
  }

  mpz_clear(x);
  mpz_clear(prime_by_2);
  mpz_clear(prime_by_n);
  mpz_clear(temp);
  gmp_randclear(r_state);
}

void Venezia::get_root_of_unity(mpz_t *arg_omega) {
  mpz_set(*arg_omega, omega);
}
