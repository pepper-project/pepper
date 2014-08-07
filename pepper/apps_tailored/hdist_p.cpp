#include <apps_tailored/hdist_p.h>

HDistProver::
HDistProver(int ph, int b_size, int num_r, int input_size)
  : Prover(ph, b_size, NUM_REPS_PCP, input_size, PROVER_NAME) {
  init_state();
}

void HDistProver::
init_state() {
  num_bits_in_prime = 128;
  num_bits_in_input = 32;
  crypto_in_use = CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;

  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;
  num_local_runs = 1;

  size_f1_vec = m;
  size_f2_vec = m*m;
  size_f3_vec = m*m;
  size_f4_vec = m*m*m;
  size_f5_vec = m*m*m;

  // initialize prover's state
  Prover::init_state();
  mpz_init(neg);

  alloc_init_vec(&F1, m);
  alloc_init_vec(&F2, m*m);
  alloc_init_vec(&F3, m*m);
  alloc_init_vec(&F4, m*m*m);
  alloc_init_vec(&F5, m*m*m);
  num_cons = 2*(m*m+m);
  alloc_init_vec(&alpha, num_cons);

  alloc_init_vec(&f1_commitment, expansion_factor*m);
  alloc_init_vec(&f1_consistency, expansion_factor*m);
  alloc_init_vec(&f1_q1, m);
  alloc_init_vec(&f1_q2, m);
  alloc_init_vec(&f1_q3, m);
  alloc_init_vec(&f1_q4, m);

  alloc_init_vec(&f2_commitment, expansion_factor*m*m);
  alloc_init_vec(&f2_consistency, expansion_factor*m*m);
  alloc_init_vec(&f2_q1, m*m);
  alloc_init_vec(&f2_q2, m*m);
  alloc_init_vec(&f2_q3, m*m);
  alloc_init_vec(&f2_q4, m*m);

  alloc_init_vec(&f3_commitment, expansion_factor*m*m);
  alloc_init_vec(&f3_consistency, expansion_factor*m*m);
  alloc_init_vec(&f3_q1, m*m);
  alloc_init_vec(&f3_q2, m*m);
  alloc_init_vec(&f3_q3, m*m);
  alloc_init_vec(&f3_q4, m*m);

  alloc_init_vec(&f4_commitment, expansion_factor*m*m*m);
  alloc_init_vec(&f4_consistency, expansion_factor*m*m*m);
  alloc_init_vec(&f4_q1, m*m*m);
  alloc_init_vec(&f4_q2, m*m*m);
  alloc_init_vec(&f4_q3, m*m*m);
  alloc_init_vec(&f4_q4, m*m*m);

  alloc_init_vec(&f5_commitment, expansion_factor*m*m*m);
  alloc_init_vec(&f5_consistency, expansion_factor*m*m*m);
  alloc_init_vec(&f5_q1, m*m*m);
  alloc_init_vec(&f5_q2, m*m*m);
  alloc_init_vec(&f5_q3, m*m*m);
  alloc_init_vec(&f5_q4, m*m*m);

  alloc_init_vec(&f_answers, num_lin_pcp_queries);
  alloc_init_vec(&dotp, expansion_factor);
  alloc_init_scalar(answer);

  F_ptrs.clear();
  F_ptrs.push_back(F1);
  F_ptrs.push_back(F2);
  F_ptrs.push_back(F3);
  F_ptrs.push_back(F4);
  F_ptrs.push_back(F5);
  f_q_ptrs.clear();
  f_q_ptrs.push_back(f1_q1);
  f_q_ptrs.push_back(f2_q1);
  f_q_ptrs.push_back(f3_q1);
  f_q_ptrs.push_back(f4_q1);
  f_q_ptrs.push_back(f5_q1);

  f_q2_ptrs.clear();
  f_q2_ptrs.push_back(f1_q2);
  f_q2_ptrs.push_back(f2_q2);
  f_q2_ptrs.push_back(f3_q2);
  f_q2_ptrs.push_back(f4_q2);
  f_q2_ptrs.push_back(f5_q2);

  find_cur_qlengths();

  // initialize computation state
  alloc_init_vec(&B, m*m);
  alloc_init_vec(&Y, m);
}

void HDistProver::
find_cur_qlengths() {
  sizes.clear();
  sizes.push_back(m);
  sizes.push_back(m*m);
  sizes.push_back(m*m);
  sizes.push_back(m*m*m);
  sizes.push_back(m*m*m);

  qquery_sizes.clear();
  //linearity queries
  for(int j=0; j<NUM_REPS_LIN; j++) {
    qquery_sizes.push_back(m);
    qquery_sizes.push_back(m);
    qquery_sizes.push_back(m);
    qquery_sizes.push_back(m*m);
    qquery_sizes.push_back(m*m);
    qquery_sizes.push_back(m*m);
    qquery_sizes.push_back(m*m);
    qquery_sizes.push_back(m*m);
    qquery_sizes.push_back(m*m);
    qquery_sizes.push_back(m*m*m);
    qquery_sizes.push_back(m*m*m);
    qquery_sizes.push_back(m*m*m);
    qquery_sizes.push_back(m*m*m);
    qquery_sizes.push_back(m*m*m);
    qquery_sizes.push_back(m*m*m);
  }
  //other queries
  qquery_sizes.push_back(m*m*m);
  qquery_sizes.push_back(m*m*m);
  qquery_sizes.push_back(m);
  qquery_sizes.push_back(m*m);
  qquery_sizes.push_back(m*m);
  qquery_sizes.push_back(m*m*m);
  qquery_sizes.push_back(m*m*m);

  qquery_f_ptrs.clear();
  for(int j=0; j<NUM_REPS_LIN; j++) {
    for (int i=0; i<3; i++)
      qquery_f_ptrs.push_back(f1_q1);
    for (int i=0; i<3; i++)
      qquery_f_ptrs.push_back(f2_q1);
    for (int i=0; i<3; i++)
      qquery_f_ptrs.push_back(f3_q1);
    for (int i=0; i<3; i++)
      qquery_f_ptrs.push_back(f4_q1);
    for (int i=0; i<3; i++)
      qquery_f_ptrs.push_back(f5_q1);
  }      
  qquery_f_ptrs.push_back(f4_q1);
  qquery_f_ptrs.push_back(f5_q1);
  qquery_f_ptrs.push_back(f1_q1);
  qquery_f_ptrs.push_back(f2_q1);
  qquery_f_ptrs.push_back(f3_q1);
  qquery_f_ptrs.push_back(f4_q1);
  qquery_f_ptrs.push_back(f5_q1);

  qquery_F_ptrs.clear();
  for(int j=0; j<NUM_REPS_LIN; j++) {
    for (int i=0; i<2; i++)
      qquery_F_ptrs.push_back(F1);
    qquery_F_ptrs.push_back(NULL);
    for (int i=0; i<2; i++)
      qquery_F_ptrs.push_back(F2);
    qquery_F_ptrs.push_back(NULL);
    for (int i=0; i<2; i++)
      qquery_F_ptrs.push_back(F3);
    qquery_F_ptrs.push_back(NULL);
    for (int i=0; i<2; i++)
      qquery_F_ptrs.push_back(F4);
    qquery_F_ptrs.push_back(NULL);
    for (int i=0; i<2; i++)
      qquery_F_ptrs.push_back(F5);
    qquery_F_ptrs.push_back(NULL);
  }      
  qquery_F_ptrs.push_back(F4);
  qquery_F_ptrs.push_back(F5);
  qquery_F_ptrs.push_back(F1);
  qquery_F_ptrs.push_back(F2);
  qquery_F_ptrs.push_back(F3);
  qquery_F_ptrs.push_back(F4);
  qquery_F_ptrs.push_back(F5);

  qquery_q_ptrs.clear();
  int query_id = 0;
  for(int j=0; j<NUM_REPS_LIN; j++) {
    for (int k=0; k<NUM_LIN_QUERIES; k++)
      qquery_q_ptrs.push_back(query_id++);
  }
  for (int j=0; j<NUM_DIV_QUERIES; j++)
    qquery_q_ptrs.push_back(query_id++);
}

// COMPUTATION
void HDistProver::
computation() {
  for (int i=0; i<m; i++)
    mpz_set_ui(Y[i], 0);

  for (int i=0; i<m; i++) {
    for (int j=0; j<m; j++) {
      if (mpz_cmp(F1[j], B[i*m+j]) == 0) {
      } else {
        mpz_add_ui(Y[i], Y[i], 1);
      }
    }
  }
}

//PROVER's CODE
void HDistProver::
prover_computation_commitment() {
  // init prover
  load_vector(expansion_factor*m, f1_commitment, (char *)"f1_commitment_query", FOLDER_WWW_DOWNLOAD);
  load_vector(expansion_factor*m*m, f2_commitment, (char *)"f2_commitment_query", FOLDER_WWW_DOWNLOAD);
  load_vector(expansion_factor*m*m, f3_commitment, (char *)"f3_commitment_query", FOLDER_WWW_DOWNLOAD);
  load_vector(expansion_factor*m*m*m, f4_commitment, (char *)"f4_commitment_query", FOLDER_WWW_DOWNLOAD);
  load_vector(expansion_factor*m*m*m, f5_commitment, (char *)"f5_commitment_query", FOLDER_WWW_DOWNLOAD);

  // execute the computation
  m_computation.begin_with_init();
  load_vector(m*m, B, (char *)"input0", FOLDER_WWW_DOWNLOAD);
  m_computation.end();

  for (int i=batch_start; i<=batch_end; i++) {
    m_computation.begin_with_history();
    snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
    load_vector(m, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    computation();

    snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
    dump_vector(m, Y, scratch_str, FOLDER_WWW_DOWNLOAD);
    m_computation.end();

    // compute assignment here
    for (int ii=0; ii<m; ii++)
      mpz_set_ui(Y[ii], 0);

    for (int ii=0; ii<m; ii++) {
      for (int jj=0; jj<m; jj++) {
        if (mpz_cmp(F1[jj], B[ii*m+jj]) == 0) {
          mpz_set_ui(F3[ii*m+jj], 0);
          mpz_set_ui(F2[ii*m+jj], 0);
        } else {
          mpz_add_ui(Y[ii], Y[ii], 1);
          mpz_set_ui(F3[ii*m+jj], 1);
          mpz_sub(F2[ii*m+jj], F1[jj], B[ii*m+jj]);
          mpz_mod(F2[ii*m+jj], F2[ii*m+jj], prime);
          mpz_invert(F2[ii*m+jj], F2[ii*m+jj], prime);
        }
      }
    }

    // start saving the state
    for (int jj=0; jj<m*m; jj++) {
      mpz_mod(F2[jj], F2[jj], prime);
      mpz_mod(F3[jj], F3[jj], prime);
    }
    for (int ii=0; ii<m; ii++) {
      mpz_mod(F1[ii], F1[ii], prime);
      for (int jj=0; jj<m*m; jj++) {
        mpz_mul(F4[ii*m*m+jj], F1[ii], F2[jj]);
        mpz_mod(F4[ii*m*m+jj], F4[ii*m*m+jj], prime);
        mpz_mul(F5[ii*m*m+jj], F1[ii], F3[jj]);
        mpz_mod(F5[ii*m*m+jj], F5[ii*m*m+jj], prime);
      }
    }
    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    dump_vector(m, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", i);
    dump_vector(m*m, F2, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f3_assignment_vector_b_%d", i);
    dump_vector(m*m, F3, scratch_str, FOLDER_WWW_DOWNLOAD);


    snprintf(scratch_str, BUFLEN-1, "f4_assignment_vector_b_%d", i);
    dump_vector(m*m*m, F4, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f5_assignment_vector_b_%d", i);
    dump_vector(m*m*m, F5, scratch_str, FOLDER_WWW_DOWNLOAD);
  }

  for (int i=batch_start; i<=batch_end; i++) {
    if (i == 0)
      m_answer_queries.begin_with_init();
    else
      m_answer_queries.begin_with_history();

    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    load_vector(m, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", i);
    load_vector(m*m, F2, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f3_assignment_vector_b_%d", i);
    load_vector(m*m, F3, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f4_assignment_vector_b_%d", i);
    load_vector(m*m*m, F4, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f5_assignment_vector_b_%d", i);
    load_vector(m*m*m, F5, scratch_str, FOLDER_WWW_DOWNLOAD);


    if (crypto_in_use == CRYPTO_ELGAMAL)
      v->dot_product_enc(m, f1_commitment, F1, dotp[0], dotp[1]);

    if (crypto_in_use == CRYPTO_ELGAMAL)
      v->dot_product_enc(m*m, f2_commitment, F2, Y[0], Y[1]);

    v->add_enc(dotp[0], dotp[1], dotp[0], dotp[1], Y[0], Y[1]);

    if (crypto_in_use == CRYPTO_ELGAMAL)
      v->dot_product_enc(m*m, f3_commitment, F3, Y[0], Y[1]);

    v->add_enc(dotp[0], dotp[1], dotp[0], dotp[1], Y[0], Y[1]);

    if (crypto_in_use == CRYPTO_ELGAMAL)
      v->dot_product_enc(m*m*m, f4_commitment, F4, Y[0], Y[1]);

    v->add_enc(dotp[0], dotp[1], dotp[0], dotp[1], Y[0], Y[1]);

    if (crypto_in_use == CRYPTO_ELGAMAL)
      v->dot_product_enc(m*m*m, f5_commitment, F5, Y[0], Y[1]);

    v->add_enc(dotp[0], dotp[1], dotp[0], dotp[1], Y[0], Y[1]);

    snprintf(scratch_str, BUFLEN-1, "f_commitment_answer_b_%d", i);
    dump_vector(expansion_factor, dotp, scratch_str, FOLDER_WWW_DOWNLOAD);

    m_answer_queries.end();
  }
}

void HDistProver::deduce_queries() {
  int f_con_filled = -1;
  mpz_t *f_con_coins = NULL;
  mpz_t *f1_consistency = NULL;
  mpz_t *f2_consistency = NULL;
  mpz_t *f3_consistency = NULL;
  mpz_t *f4_consistency = NULL;
  mpz_t *f5_consistency = NULL;

  for (int rho=0; rho<num_repetitions; rho++) {
    if (rho == 0) m_plainq.begin_with_init();
    else m_plainq.begin_with_history();

    int query_id = 1;
    for (int i=0; i<NUM_REPS_LIN; i++) {
      v->create_lin_test_queries(size_f1_vec, f1_q1, f1_q2, f1_q3, f1_consistency,
          f_con_filled, f_con_coins, prime);
      f_con_filled += 3;
      
      v->create_lin_test_queries(size_f2_vec, f2_q1, f2_q2, f2_q3, f2_consistency,
          f_con_filled, f_con_coins, prime);
      f_con_filled += 3;
 
      v->create_lin_test_queries(size_f3_vec, f3_q1, f3_q2, f3_q3, f3_consistency,
          f_con_filled, f_con_coins, prime);
      f_con_filled += 3;
      
      v->create_lin_test_queries(size_f4_vec, f4_q1, f4_q2, f4_q3, f4_consistency,
          f_con_filled, f_con_coins, prime);
      f_con_filled += 3;
 
      v->create_lin_test_queries(size_f5_vec, f5_q1, f5_q2, f5_q3, f5_consistency,
          f_con_filled, f_con_coins, prime);
      f_con_filled += 3;
 
      //TODO: can be folded into a function
      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f1_vec, f1_q1, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f1_vec, f1_q2, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f1_vec, f1_q3, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q1, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q2, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q3, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f3_vec, f3_q1, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f3_vec, f3_q2, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f3_vec, f3_q3, scratch_str);
      //send_file(scratch_str);
      
      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f4_vec, f4_q1, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f4_vec, f4_q2, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f4_vec, f4_q3, scratch_str);
      
      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f5_vec, f5_q1, scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f5_vec, f5_q2, scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f5_vec, f5_q3, scratch_str);
      
      // use one of the linearity queries as self correction queries
      if (i == 0) {
        for (int i=0; i<size_f1_vec; i++)
          mpz_set(f1_q4[i], f1_q1[i]);

        for (int i=0; i<size_f2_vec; i++)
          mpz_set(f2_q4[i], f2_q1[i]);
       
        for (int i=0; i<size_f3_vec; i++)
          mpz_set(f3_q4[i], f3_q1[i]);
      
        for (int i=0; i<size_f4_vec; i++)
          mpz_set(f4_q4[i], f4_q1[i]);
        
        for (int i=0; i<size_f5_vec; i++)
          mpz_set(f5_q4[i], f5_q1[i]);
      }
    }

    v->create_corr_test_queries_reuse(m, f1_q4, m*m, f2_q4,
                                f4_q1, f4_q4, f1_consistency, f2_consistency, f4_consistency,
                                f_con_filled, f_con_coins, f_con_filled, f_con_coins,
                                f_con_filled, f_con_coins, prime, false);

    f_con_filled += 1;

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m*m, f4_q1, scratch_str);

    v->create_corr_test_queries_reuse(m, f1_q4, m*m, f3_q4,
                                f5_q1, f5_q4, f1_consistency, f3_consistency, f5_consistency,
                                f_con_filled, f_con_coins, f_con_filled, f_con_coins,
                                f_con_filled, f_con_coins, prime, false);

    f_con_filled += 1;

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m*m, f5_q1, scratch_str);

    // circuit test queries
    snprintf(scratch_str, BUFLEN-1, "input0");
    load_vector(m*m, B, scratch_str);

    // start with generating a random alpha for each constraint
    v->get_random_vec_pub(num_cons, alpha, prime);

    // first, formulate \gamma_1
    for (int i=0; i<m; i++) {
      mpz_set_ui(f1_q2[i], 0);
    }

    mpz_t *A_var = &f1_q2[0];

    // A_j
    int con_id = m*m;
    for (int i=0; i<m; i++) {
      for (int j=0; j<m; j++) {
        mpz_add(A_var[j], A_var[j], alpha[con_id]);
        con_id++;
      }
    }

    // -A_j
    con_id = 2*m*m+m;
    for (int i=0; i<m; i++) {
      mpz_neg(neg, alpha[con_id]);
      mpz_add(A_var[i], A_var[i], neg);
      con_id++;
    }

    // second, formulate \gamma_2
    for (int i=0; i<m*m; i++) {
      mpz_set_ui(f2_q2[i], 0);
    }
    mpz_t *M_var = &f2_q2[0];

    // -b_ij M_ij; there are m^2 such constraints
    con_id = 0;
    for (int i=0; i<m*m; i++) {
      mpz_neg(neg, alpha[con_id]);
      mpz_mul(neg, neg, B[i]);
      mpz_add(M_var[i], M_var[i], neg);
      con_id++;
    }


    // third, formulate \gamma_3
    for (int i=0; i<m*m; i++) {
      mpz_set_ui(f3_q2[i], 0);
    }
    mpz_t *S_var = &f3_q2[0];

    // -S_ij and there are m^2 such constraints
    con_id = 0;
    for (int i=0; i<m*m; i++) {
      mpz_neg(neg, alpha[con_id]);
      mpz_add(S_var[i], S_var[i], neg);
      con_id++;
    }


    // b_ij S_ij; there are m^2 such constraints
    con_id = m*m;
    for (int i=0; i<m*m; i++) {
      mpz_mul(neg, alpha[con_id], B[i]);
      mpz_add(S_var[i], S_var[i], neg);
      con_id++;
    }

    // -sum_{j}{S_ij}; there are m such constraints
    con_id = 2*m*m;
    for (int i=0; i<m; i++) {
      mpz_neg(neg, alpha[con_id]);
      for (int j=0; j<m; j++) {
        mpz_add(S_var[i*m+j], S_var[i*m+j], neg);
      }
      con_id++;
    }


    // fourth, formulate \gamma_4
    for (int i=0; i<m*m*m; i++) {
      mpz_set_ui(f4_q2[i], 0);
    }

    // A_j M_ij and there are m^2 such constraints
    con_id = 0;
    int row_id, col_id;
    for (int i=0; i<m; i++) {
      for (int j=0; j<m; j++) {
        row_id = j; // A_j's position
        col_id = (i*m + j); //M_ij's position
        mpz_add(f4_q2[row_id * m*m + col_id], f4_q2[row_id*m*m + col_id], alpha[con_id]);
        con_id++;
      }
    }


    // fifth, formulate \gamma_5
    for (int i=0; i<m*m*m; i++) {
      mpz_set_ui(f5_q2[i], 0);
    }

    // -A_j S_ij and there are m^2 such constraints
    con_id = m*m;
    for (int i=0; i<m; i++) {
      for (int j=0; j<m; j++) {
        row_id = j; // A_j's position
        col_id = (i*m + j); //S_ij's position
        mpz_neg(neg, alpha[con_id]);
        mpz_add(f5_q2[row_id * m * m + col_id], f5_q2[row_id * m * m + col_id], neg);
        con_id++;
      }
    }

    v->create_ckt_test_queries(m, f1_q2, f1_q3, f1_q4,
                               f1_consistency, f_con_filled, f_con_coins, prime);
    f_con_filled++;
    v->create_ckt_test_queries(m*m, f2_q2, f2_q3, f2_q4,
                               f2_consistency, f_con_filled, f_con_coins, prime);
    f_con_filled++;

    v->create_ckt_test_queries(m*m, f3_q2, f3_q3, f3_q4,
                               f3_consistency, f_con_filled, f_con_coins, prime);
    f_con_filled++;

    v->create_ckt_test_queries(m*m*m, f4_q2, f4_q3, f4_q4,
                               f4_consistency, f_con_filled, f_con_coins, prime);
    f_con_filled++;

    v->create_ckt_test_queries(m*m*m, f5_q2, f5_q3, f5_q4,
                               f5_consistency, f_con_filled, f_con_coins, prime);
    f_con_filled++;


    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m, f1_q3, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m, f2_q3, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m, f3_q3, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m*m, f4_q3, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m*m, f5_q3, scratch_str);
    m_plainq.end();
  }
}
