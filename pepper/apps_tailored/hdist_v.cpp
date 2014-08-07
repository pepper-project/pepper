#include <apps_tailored/hdist_v.h>

HDistVerifier::HDistVerifier(int batch, int reps, int ip_size,
                             int optimize_answers, char *prover_url)
  : Verifier(batch, NUM_REPS_PCP, ip_size, optimize_answers, prover_url, NAME_PROVER) {
  m = ip_size; // move this to base class and change it in every app
  init_state();
}

void HDistVerifier::init_state() {
  num_bits_in_prime = 128;
  num_bits_in_input = 32;
  crypto_in_use = CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;

  // input and output of the computation
  alloc_init_vec(&A, m);
  alloc_init_vec(&B, m*m);

  size_f1_vec = m;
  size_f2_vec = m*m;
  size_f3_vec = m*m;
  size_f4_vec = m*m*m;
  size_f5_vec = m*m*m;

  num_cons = 2*(m*m+m);

  // state of the verifier
  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;
  num_verification_runs = NUM_VERIFICATION_RUNS;
  Verifier::init_state();

  alloc_init_vec(&f1_commitment, expansion_factor*m);
  alloc_init_vec(&f2_commitment, expansion_factor*m*m);
  alloc_init_vec(&f3_commitment, expansion_factor*m*m);
  alloc_init_vec(&f4_commitment, expansion_factor*m*m*m);
  alloc_init_vec(&f5_commitment, expansion_factor*m*m*m);

  alloc_init_vec(&f1_consistency, m);
  alloc_init_vec(&f2_consistency, m*m);
  alloc_init_vec(&f3_consistency, m*m);
  alloc_init_vec(&f4_consistency, m*m*m);
  alloc_init_vec(&f5_consistency, m*m*m);

  alloc_init_vec(&f1_q1, m);
  alloc_init_vec(&f1_q2, m);
  alloc_init_vec(&f1_q3, m);
  alloc_init_vec(&f1_q4, m);

  alloc_init_vec(&f2_q1, m*m);
  alloc_init_vec(&f2_q2, m*m);
  alloc_init_vec(&f2_q3, m*m);
  alloc_init_vec(&f2_q4, m*m);

  alloc_init_vec(&f3_q1, m*m);
  alloc_init_vec(&f3_q2, m*m);
  alloc_init_vec(&f3_q3, m*m);
  alloc_init_vec(&f3_q4, m*m);

  alloc_init_vec(&f4_q1, m*m*m);
  alloc_init_vec(&f4_q2, m*m*m);
  alloc_init_vec(&f4_q3, m*m*m);
  alloc_init_vec(&f4_q4, m*m*m);

  alloc_init_vec(&f5_q1, m*m*m);
  alloc_init_vec(&f5_q2, m*m*m);
  alloc_init_vec(&f5_q3, m*m*m);
  alloc_init_vec(&f5_q4, m*m*m);

  alloc_init_vec(&alpha, num_cons);
  alloc_init_vec(&f_answers, num_repetitions * num_lin_pcp_queries);
  alloc_init_vec(&ckt_answers, 10);

  alloc_init_vec(&temp_arr, expansion_factor);
  alloc_init_vec(&temp_arr2, expansion_factor);
  alloc_init_vec(&temp_arr3, expansion_factor);
  alloc_init_vec(&temp_arr4, expansion_factor);
  alloc_init_vec(&temp_arr5, expansion_factor);

  alloc_init_scalar(c_init_val);
  alloc_init_scalar(a1);
  alloc_init_scalar(a2);
  alloc_init_scalar(a3);
  alloc_init_scalar(a4);
  alloc_init_scalar(a5);
  alloc_init_scalar(f1_s);
  alloc_init_scalar(f2_s);
  alloc_init_scalar(f3_s);
  alloc_init_scalar(f4_s);
  alloc_init_scalar(f5_s);
  mpz_init(temp);
  alloc_init_vec(&output, m);
  mpz_init(neg);
  mpz_init(neg_i);

  // To create consistency and commitment queries.
  commitment_query_sizes.clear();
  commitment_query_sizes.push_back(m);
  commitment_query_sizes.push_back(m*m);
  commitment_query_sizes.push_back(m*m);
  commitment_query_sizes.push_back(m*m*m);
  commitment_query_sizes.push_back(m*m*m);

  f_commitment_ptrs.clear();
  f_commitment_ptrs.push_back(f1_commitment);
  f_commitment_ptrs.push_back(f2_commitment);
  f_commitment_ptrs.push_back(f3_commitment);
  f_commitment_ptrs.push_back(f4_commitment);
  f_commitment_ptrs.push_back(f5_commitment);

  f_consistency_ptrs.clear();
  f_consistency_ptrs.push_back(f1_consistency);
  f_consistency_ptrs.push_back(f2_consistency);
  f_consistency_ptrs.push_back(f3_consistency);
  f_consistency_ptrs.push_back(f4_consistency);
  f_consistency_ptrs.push_back(f5_consistency);
  temp_arr_ptrs.clear();
  temp_arr_ptrs.push_back(temp_arr);
  temp_arr_ptrs.push_back(temp_arr2);
  temp_arr_ptrs.push_back(temp_arr3);
  temp_arr_ptrs.push_back(temp_arr4);
  temp_arr_ptrs.push_back(temp_arr5);

  Q_list.clear();
  int query_id = 0;
  for(int j=0; j<NUM_REPS_LIN; j++) {
    for (int k=0; k<NUM_LIN_QUERIES; k++)
      Q_list.push_back(query_id++);
  }
  for (int j=0; j<NUM_DIV_QUERIES; j++)
    Q_list.push_back(query_id++);
}

void HDistVerifier::create_input() {
  v->get_random_vec_priv(m*m, B, num_bits_in_input);
  dump_vector(m*m, B, (char *)"input0");
  send_file((char *)"input0");

  // as many computations as inputs
  for (int k=0; k<batch_size; k++) {
    v->get_random_vec_priv(m, A, num_bits_in_input);
    snprintf(scratch_str, BUFLEN-1, "input_b_%d", k);
    dump_vector(m, A, scratch_str);
    send_file(scratch_str);
  }
}

void HDistVerifier::create_plain_queries() {
  // keeps track of #filled coins
  int f_con_filled = -1;
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
 
      /*
      //TODO: can be folded into a function
      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f1_vec, f1_q1, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f1_vec, f1_q2, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f1_vec, f1_q3, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q1, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q2, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q3, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f3_vec, f3_q1, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f3_vec, f3_q2, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f3_vec, f3_q3, scratch_str);
      send_file(scratch_str);
      
      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f4_vec, f4_q1, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f4_vec, f4_q2, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f4_vec, f4_q3, scratch_str);
      send_file(scratch_str);
      
      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f5_vec, f5_q1, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f5_vec, f5_q2, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f5_vec, f5_q3, scratch_str);
      send_file(scratch_str);
      */

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

    /*
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m*m, f4_q1, scratch_str);
    send_file(scratch_str);
    */

    v->create_corr_test_queries_reuse(m, f1_q4, m*m, f3_q4,
                                f5_q1, f5_q4, f1_consistency, f3_consistency, f5_consistency,
                                f_con_filled, f_con_coins, f_con_filled, f_con_coins,
                                f_con_filled, f_con_coins, prime, false);

    f_con_filled += 1;
    
    /*
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m*m, f5_q1, scratch_str);
    send_file(scratch_str);
    */

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

    /*
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m, f1_q3, scratch_str);
    send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m, f2_q3, scratch_str);
    send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m, f3_q3, scratch_str);
    send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m*m, f4_q3, scratch_str);
    send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(m*m*m, f5_q3, scratch_str);
    send_file(scratch_str);
    */

    // compute a component of \gamma_0
    con_id = m*m;
    mpz_set_ui(c_init_val, 0);
    for (int i=0; i<m*m; i++) {
      mpz_neg(neg, alpha[con_id]);
      mpz_mul(neg, B[i], neg);
      mpz_add(c_init_val, c_init_val, neg);
      con_id++;
    }
    m_plainq.end();

    if (rho == 0) m_runtests.begin_with_init();
    else m_runtests.begin_with_history();

    // finally compute \gamma_0
    for (int b=0; b<batch_size; b++) {
      snprintf(scratch_str, BUFLEN-1, "input_b_%d", b);
      load_vector(m, A, scratch_str);

      snprintf(scratch_str, BUFLEN-1, "output_b_%d", b);
      load_vector(m, output, scratch_str);

      int c_index = b * num_repetitions + rho;
      mpz_set(c_values[c_index], c_init_val);

      con_id = 2*m*m;
      for (int i=0; i<m; i++) {
        mpz_mul(temp, output[i], alpha[con_id]);
        mpz_add(c_values[c_index], c_values[c_index], temp);
        con_id++;
      }

      con_id = 2*m*m + m;
      for (int i=0; i<m; i++) {
        mpz_mul(temp, A[i], alpha[con_id]);
        mpz_add(c_values[c_index], c_values[c_index], temp);
        con_id++;
      }

      mpz_mod(c_values[c_index], c_values[c_index], prime);
    }
    m_runtests.end();

    //populate_answers(f_answers, rho, num_repetitions, 0);
  }
  dump_vector(m, f1_consistency, (char *)"f1_consistency_query");
  send_file((char *)"f1_consistency_query");

  dump_vector(m*m, f2_consistency, (char *)"f2_consistency_query");
  send_file((char *)"f2_consistency_query");

  dump_vector(m*m, f3_consistency, (char *)"f3_consistency_query");
  send_file((char *)"f3_consistency_query");

  dump_vector(m*m*m, f4_consistency, (char *)"f4_consistency_query");
  send_file((char *)"f4_consistency_query");

  dump_vector(m*m*m, f5_consistency, (char *)"f5_consistency_query");
  send_file((char *)"f5_consistency_query");
}

void HDistVerifier::populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta) {
}

bool HDistVerifier::run_correction_and_circuit_tests(uint32_t beta) {
  bool result = false;

  for (int rho=0; rho<num_repetitions; rho++) {
    
    for (int i=0; i<NUM_REPS_LIN; i++) {
      int base = i*NUM_LIN_QUERIES;

      // linearity test
      bool lin1 = v->lin_test(f_answers[rho*num_lin_pcp_queries + base + Q1],
                              f_answers[rho*num_lin_pcp_queries + base + Q2],
                              f_answers[rho*num_lin_pcp_queries + base + Q3],
                              prime);

      bool lin2 = v->lin_test(f_answers[rho*num_lin_pcp_queries + base + Q4],
                              f_answers[rho*num_lin_pcp_queries + base + Q5],
                              f_answers[rho*num_lin_pcp_queries + base + Q6],
                              prime);

      bool lin3 = v->lin_test(f_answers[rho*num_lin_pcp_queries + base + Q7],
                              f_answers[rho*num_lin_pcp_queries + base + Q8],
                              f_answers[rho*num_lin_pcp_queries + base + Q9],
                              prime);

      bool lin4 = v->lin_test(f_answers[rho*num_lin_pcp_queries + base + Q10],
                              f_answers[rho*num_lin_pcp_queries + base + Q11],
                              f_answers[rho*num_lin_pcp_queries + base + Q12],
                              prime);

      bool lin5 = v->lin_test(f_answers[rho*num_lin_pcp_queries + base + Q13],
                              f_answers[rho*num_lin_pcp_queries + base + Q14],
                              f_answers[rho*num_lin_pcp_queries + base + Q15],
                              prime);

#if VERBOSE == 1
      if (false == lin1 || false == lin2)
        cout<<"LOG: F1, F2 failed the linearity test"<<endl;
      else
        cout<<"LOG: F1, F2 passed the linearity test"<<endl;
#endif
      if (i==0) result = lin1 & lin2 & lin3 & lin4 & lin5;
      else result &= lin1 & lin2 & lin3 & lin4 & lin5;
    }
 
    // Quad Correction test and Circuit test
    bool cor1 = v->corr_test(f_answers[rho*num_lin_pcp_queries + Q1],
                             f_answers[rho*num_lin_pcp_queries + Q4],
                             f_answers[rho*num_lin_pcp_queries + Q16],
                             f_answers[rho*num_lin_pcp_queries + Q10],
                             prime);


    bool cor2 = v->corr_test(f_answers[rho*num_lin_pcp_queries + Q1],
                             f_answers[rho*num_lin_pcp_queries + Q7],
                             f_answers[rho*num_lin_pcp_queries + Q17],
                             f_answers[rho*num_lin_pcp_queries + Q13],
                             prime);

#if VERBOSE == 1
    if (false == cor1 && false == cor2)
      cout<<"LOG: F1, F2, F3, F4, F5 failed the correction test"<<endl;
    else
      cout<<"LOG: F1, F2, F3, F4, F5 passed correction test"<<endl;
#endif

    result &= cor1 & cor2;

    mpz_set(ckt_answers[0], f_answers[rho*num_lin_pcp_queries + Q18]);
    mpz_set(ckt_answers[1], f_answers[rho*num_lin_pcp_queries + Q1]);

    mpz_set(ckt_answers[2], f_answers[rho*num_lin_pcp_queries + Q19]);
    mpz_set(ckt_answers[3], f_answers[rho*num_lin_pcp_queries + Q4]);

    mpz_set(ckt_answers[4], f_answers[rho*num_lin_pcp_queries + Q20]);
    mpz_set(ckt_answers[5], f_answers[rho*num_lin_pcp_queries + Q7]);

    mpz_set(ckt_answers[6], f_answers[rho*num_lin_pcp_queries + Q21]);
    mpz_set(ckt_answers[7], f_answers[rho*num_lin_pcp_queries + Q10]);

    mpz_set(ckt_answers[8], f_answers[rho*num_lin_pcp_queries + Q22]);
    mpz_set(ckt_answers[9], f_answers[rho*num_lin_pcp_queries + Q13]);

    bool ckt2 = v->ckt_test(10, ckt_answers, c_values[beta * num_repetitions + rho], prime);

#if VERBOSE == 1
    if (false == ckt2)
      cout <<"LOG: F1, F2, F3, F4, F5 failed the circuit test"<<endl;
    else
      cout <<"LOG: F1, F2, F3, F4, F5 passed the circuit test"<<endl;
#endif

    result &= ckt2;
  }
  return result;
}
