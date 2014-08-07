#include <apps_tailored/polyeval_d3_v.h>

PolyEvalD3Verifier::PolyEvalD3Verifier(int batch, int reps, int ip_size,
                                       int optimize_answers, char *prover_url)
  : Verifier(batch, NUM_REPS_PCP, ip_size, optimize_answers, prover_url, NAME_PROVER) {
  init_state();
}

void PolyEvalD3Verifier::init_state() {
  num_bits_in_prime = 192;
  num_bits_in_input = 12;
  crypto_in_use= CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;
 
  size_f1_vec = input_size;
  size_f2_vec = size_f1_vec*size_f1_vec;
  size_f3_vec = size_f1_vec*size_f2_vec;

  num_coefficients = (size_f3_vec + 3*size_f2_vec + 2*size_f1_vec)/6 + (size_f1_vec * size_f1_vec + 3*size_f1_vec)/2 + 1;
  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;
  num_verification_runs = NUM_VERIFICATION_RUNS;

  Verifier::init_state();
   
  alloc_init_vec(&f_answers, num_repetitions * num_lin_pcp_queries);
  alloc_init_vec(&coefficients, num_coefficients);
  alloc_init_vec(&input, size_f1_vec);
  alloc_init_vec(&f1_commitment, expansion_factor*size_f1_vec);
  alloc_init_vec(&f2_commitment, expansion_factor*size_f2_vec);
  alloc_init_vec(&f3_commitment, expansion_factor*size_f3_vec);
  alloc_init_vec(&f1_q1, size_f1_vec);
  alloc_init_vec(&f1_q2, size_f1_vec);
  alloc_init_vec(&f1_q3, size_f1_vec);
  alloc_init_vec(&f1_q4, size_f1_vec);
  alloc_init_vec(&f1_q5, size_f1_vec);
  alloc_init_vec(&f2_q1, size_f2_vec);
  alloc_init_vec(&f2_q2, size_f2_vec);
  alloc_init_vec(&f2_q3, size_f2_vec);
  alloc_init_vec(&f2_q4, size_f2_vec);
  alloc_init_vec(&f3_q1, size_f3_vec);
  alloc_init_vec(&f3_q2, size_f3_vec);
  alloc_init_vec(&f3_q3, size_f3_vec);
  alloc_init_vec(&f3_q4, size_f3_vec);
  alloc_init_vec(&f1_consistency, size_f1_vec);
  alloc_init_vec(&f2_consistency, size_f2_vec);
  alloc_init_vec(&f3_consistency, size_f3_vec);

  alloc_init_vec(&alpha, size_f1_vec+1);
  alloc_init_vec(&ckt_answers, 6);
  alloc_init_vec(&temp_arr, expansion_factor);
  alloc_init_vec(&temp_arr2, expansion_factor);
  alloc_init_vec(&temp_arr3, expansion_factor);
  mpz_init(temp);
  mpz_init(output);
  mpz_init(neg);
  mpz_init(neg_i);

  // To create consistency and commitment queries.
  commitment_query_sizes.clear();
  commitment_query_sizes.push_back(size_f1_vec);
  commitment_query_sizes.push_back(size_f2_vec);
  commitment_query_sizes.push_back(size_f3_vec);
  f_commitment_ptrs.clear();
  f_commitment_ptrs.push_back(f1_commitment);
  f_commitment_ptrs.push_back(f2_commitment);
  f_commitment_ptrs.push_back(f3_commitment);
  f_consistency_ptrs.clear();
  f_consistency_ptrs.push_back(f1_consistency);
  f_consistency_ptrs.push_back(f2_consistency);
  f_consistency_ptrs.push_back(f3_consistency);
  
  temp_arr_ptrs.clear();
  temp_arr_ptrs.push_back(temp_arr);
  temp_arr_ptrs.push_back(temp_arr2);
  temp_arr_ptrs.push_back(temp_arr3);
  
  Q_list.clear();
  int query_id = 0;
  for(int j=0; j<NUM_REPS_LIN; j++) {
    for (int k=0; k<NUM_LIN_QUERIES; k++)
      Q_list.push_back(query_id++);
  }
  for (int j=0; j<NUM_DIV_QUERIES; j++)
    Q_list.push_back(query_id++);
}

void PolyEvalD3Verifier::create_input() {
  v->get_random_vec_priv(num_coefficients, coefficients, num_bits_in_input);
  //v->add_sign(num_coefficients, coefficients);
  dump_vector(num_coefficients, coefficients, (char *)"coefficients");
  send_file((char *)"coefficients");

  // as many computations as inputs
  for (int k=0; k<batch_size; k++) {
    v->get_random_vec_priv(size_f1_vec, input, num_bits_in_input);
    v->add_sign(size_f1_vec, input);
    snprintf(scratch_str, BUFLEN-1, "input_b_%d", k);
    dump_vector(size_f1_vec, input, scratch_str);
    send_file(scratch_str);
  }
}

void PolyEvalD3Verifier::create_plain_queries() {
  // keeps track of #filled coins
  int f_con_filled = -1;
  load_vector(num_coefficients, coefficients, (char *)"coefficients");

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
 
      //TODO: can be folded into a function
      /*
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
      */
      // use one of the linearity queries as self correction queries
      if (i == 0) {
        for (int i=0; i<size_f1_vec; i++) {
          mpz_set(f1_q4[i], f1_q1[i]);
          mpz_set(f1_q5[i], f1_q2[i]);
        }

        for (int i=0; i<size_f2_vec; i++)
          mpz_set(f2_q4[i], f2_q1[i]);
       
        for (int i=0; i<size_f3_vec; i++)
          mpz_set(f3_q4[i], f3_q1[i]);
      }
    }

    v->create_corr_test_queries_reuse(size_f1_vec, f1_q4, size_f1_vec, f1_q5,
                                f2_q1, f2_q4, f1_consistency, f1_consistency, f2_consistency,
                                f_con_filled, f_con_coins, f_con_filled, f_con_coins,
                                f_con_filled, f_con_coins, prime, false);
    f_con_filled += 1;
 
    /*
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f2_vec, f2_q1, scratch_str);
    send_file(scratch_str);
    */

    v->create_corr_test_queries_reuse(size_f1_vec, f1_q4,
                                size_f2_vec, f2_q4, f3_q1, f3_q4,
                                f1_consistency, f2_consistency, f3_consistency, f_con_filled,
                                f_con_coins, f_con_filled, f_con_coins, f_con_filled,
                                f_con_coins, prime, false);

    f_con_filled += 1;
    
    /*
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f3_vec, f3_q1, scratch_str);
    send_file(scratch_str);
    */

    // circuit test
    v->get_random_vec_pub(size_f1_vec+1, alpha, prime);

    // formulate a; a = -\alpha_0 \cdot degree-2 coefficients
    mpz_neg(neg, alpha[0]);

    // set the query to zero first
    for (int i=0; i<size_f3_vec; i++) {
      mpz_set_ui(f3_q1[i], 0);
    }

    int index_coefficients = 0;
    int index_query = 0;
    for (int i=0; i<size_f1_vec; i++) {
      for (int j=0; j<=i; j++) {
        for (int k=0; k<=j; k++) {
          index_query = (i*size_f1_vec+j)*size_f1_vec+k;
          mpz_mul(f3_q1[index_query], neg, coefficients[index_coefficients]);
          mpz_mod(f3_q1[index_query], f3_q1[index_query], prime);

          index_coefficients++;
        }
      }
    }

    int offset = (size_f3_vec + 3*size_f2_vec + 2*size_f1_vec)/6;

    // quadratic part of circuit test query
    for (int i=0; i<size_f2_vec; i++) {
      mpz_set_ui(f2_q1[i], 0);
    }

    int index;
    int k = 0;
    for (int i=0; i<size_f1_vec; i++) {
      for (int j=0; j<=i; j++) {
        index = size_f1_vec*i + j;
        mpz_mul(f2_q1[index], neg, coefficients[offset+k]);
        mpz_mod(f2_q1[index], f2_q1[index], prime);
        k++;
      }
    }
    offset += (size_f1_vec * size_f1_vec + size_f1_vec)/2;

    // formulate b; b = -\alpha_0 \cdot degree-1 coefficients + [\alpha_1,
    // \alpha_2, ... , \alpha_{size_f1_vec+1}]
    for (int i=0; i<size_f1_vec; i++) {
      mpz_mul(f1_q2[i], neg, coefficients[offset+i]);

      mpz_add(f1_q2[i], alpha[i+1], f1_q2[i]);
      mpz_mod(f1_q2[i], f1_q2[i], prime);
    }

    v->create_ckt_test_queries(size_f1_vec, f1_q2, f1_q3, f1_q4,
                               f1_consistency, f_con_filled, f_con_coins, prime);
    f_con_filled += 1;
    v->create_ckt_test_queries(size_f2_vec, f2_q1, f2_q3,
                               f2_q4, f2_consistency, f_con_filled, f_con_coins, prime);
    f_con_filled += 1;
    v->create_ckt_test_queries(size_f3_vec, f3_q1,
                               f3_q3, f3_q4, f3_consistency, f_con_filled, f_con_coins, prime);
    f_con_filled += 1;
 
    /*
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f1_vec, f1_q3, scratch_str);
    send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f2_vec, f2_q3, scratch_str);
    send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f3_vec, f3_q3, scratch_str);
    send_file(scratch_str);
    */
    m_plainq.end();

    if (rho == 0) m_runtests.begin_with_init();
    else m_runtests.begin_with_history();
    // compute c
    for (int b=0; b<batch_size; b++) {
      snprintf(scratch_str, BUFLEN-1, "input_b_%d", b);
      load_vector(size_f1_vec, input, scratch_str);

      snprintf(scratch_str, BUFLEN-1, "output_b_%d", b);
      load_scalar(output, scratch_str);

      int c_index = b * num_repetitions + rho;
      mpz_set_ui(c_values[c_index], 0);

      for (int j=0; j<size_f1_vec; j++) {
        mpz_neg(neg_i, alpha[j+1]);
        mpz_mul(temp, neg_i, input[j]);
        mpz_add(c_values[c_index], c_values[c_index], temp);
      }
      mpz_mul(temp, neg, coefficients[num_coefficients-1]);
      mpz_add(c_values[c_index], c_values[c_index], temp);

      mpz_mul(temp, alpha[0], output);
      mpz_add(c_values[c_index], c_values[c_index], temp);
      mpz_mod(c_values[c_index], c_values[c_index], prime);
    }
  }

  dump_vector(size_f1_vec, f1_consistency, (char *)"f1_consistency_query");
  send_file((char *)"f1_consistency_query");

  dump_vector(size_f2_vec, f2_consistency, (char *)"f2_consistency_query");
  send_file((char *)"f2_consistency_query");

  dump_vector(size_f3_vec, f3_consistency, (char *)"f3_consistency_query");
  send_file((char *)"f3_consistency_query");
}

void PolyEvalD3Verifier::populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta) {
}

bool PolyEvalD3Verifier::run_correction_and_circuit_tests(uint32_t beta) {
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

#if VERBOSE == 1
      if (false == lin1 || false == lin2)
        cout<<"LOG: F1, F2 failed the linearity test"<<endl;
      else
        cout<<"LOG: F1, F2 passed the linearity test"<<endl;
#endif
      if (i==0) result = lin1 & lin2 & lin3;
      else result &= lin1 & lin2 & lin3;
    }

    // Quad Correction test and Circuit test
    bool cor1 = v->corr_test(f_answers[rho*num_lin_pcp_queries + Q1],
                             f_answers[rho*num_lin_pcp_queries + Q2],
                             f_answers[rho*num_lin_pcp_queries + Q10],
                             f_answers[rho*num_lin_pcp_queries + Q4], prime);

    bool cor2 = v->corr_test(f_answers[rho*num_lin_pcp_queries + Q1],
                             f_answers[rho*num_lin_pcp_queries + Q4],
                             f_answers[rho*num_lin_pcp_queries + Q11],
                             f_answers[rho*num_lin_pcp_queries + Q7], prime);
#if VERBOSE == 1
    if (false == cor1)
      cout<<"LOG: F1, F2 failed the correction test"<<endl;
    else
      cout<<"LOG: F1, F2 passed correction test"<<endl;

    if (false == cor2)
      cout<<"LOG: F1, F2, F3 failed the correction test"<<endl;
    else
      cout<<"LOG: F1, F2, F3 passed correction test"<<endl;
#endif

    result = result & cor1 & cor2;

    mpz_set(ckt_answers[0], f_answers[rho*num_lin_pcp_queries + Q12]);
    mpz_set(ckt_answers[1], f_answers[rho*num_lin_pcp_queries + Q1]);
    mpz_set(ckt_answers[2], f_answers[rho*num_lin_pcp_queries + Q13]);
    mpz_set(ckt_answers[3], f_answers[rho*num_lin_pcp_queries + Q4]);
    mpz_set(ckt_answers[4], f_answers[rho*num_lin_pcp_queries + Q14]);
    mpz_set(ckt_answers[5], f_answers[rho*num_lin_pcp_queries + Q7]);

    bool ckt2 = v->ckt_test(6, ckt_answers,
                            c_values[beta * num_repetitions + rho], prime);

#if VERBOSE == 1
    if (false == ckt2)
      cout <<"LOG: F1, F2, F3 failed the circuit test"<<endl;
    else
      cout <<"LOG: F1, F2, F3 passed the circuit test"<<endl;
#endif
    
    result &= ckt2;
  }
  return result;
}
