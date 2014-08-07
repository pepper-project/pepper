#include <apps_tailored/matrix_cubicp_q_v.h>

MatrixCubicQVerifier::MatrixCubicQVerifier(int batch, int reps, int
    ip_size, int optimize_answers, char *prover_url)
  : Verifier(batch, NUM_REPS_PCP, ip_size, optimize_answers, prover_url, NAME_PROVER) {
  size_input = 2 * ip_size * ip_size;
  size_output = ip_size * ip_size;
  init_state();
}

void MatrixCubicQVerifier::init_state() {
  num_bits_in_prime = 220;
  num_bits_in_input = 32;
  crypto_in_use = CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;
  cout<<"Running verifier with a prime of size "<<num_bits_in_prime<<endl;

  hadamard_code_size = input_size * input_size * input_size;
  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;
  num_verification_runs = NUM_VERIFICATION_RUNS;

  Verifier::init_state();

  alloc_init_vec(&input, size_input);
  alloc_init_vec(&output, size_output);
  alloc_init_vec(&A, size_input);
  B = &A[size_input/2];
  alloc_init_vec(&A_q, size_input);
  alloc_init_vec(&C, size_output);
  alloc_init_vec(&C_q, size_output);
  alloc_init_vec(&f1_commitment, expansion_factor*hadamard_code_size);
  alloc_init_vec(&f1_consistency, hadamard_code_size);
  alloc_init_vec(&f1_q1, hadamard_code_size);
  alloc_init_vec(&f1_q2, hadamard_code_size);
  alloc_init_vec(&f1_q3, hadamard_code_size);
  alloc_init_vec(&f1_q4, hadamard_code_size);
  alloc_init_vec(&f2_q1, 2*input_size*input_size);
  alloc_init_vec(&f2_q2, 2*input_size*input_size);
  alloc_init_vec(&gamma, input_size*input_size);
  //alloc_init_vec(&f1_con_coins, num_repetitions * NUM_LIN_PCP_QUERIES);

  alloc_init_vec(&f_answers, num_repetitions * num_lin_pcp_queries);
  alloc_init_vec(&f1_answers, num_repetitions * num_lin_pcp_queries);
  alloc_init_vec(&ckt_answers, 2);
  alloc_init_vec(&temp_arr, expansion_factor);

  alloc_init_scalar(a1);
  alloc_init_scalar(a2);
  alloc_init_scalar(a3);
  alloc_init_scalar(temp);
  alloc_init_scalar(temp2);

  // To create consistency and commitment queries.
  commitment_query_sizes.clear();
  commitment_query_sizes.push_back(hadamard_code_size);
  f_commitment_ptrs.clear();
  f_commitment_ptrs.push_back(f1_commitment);
  f_consistency_ptrs.clear();
  f_consistency_ptrs.push_back(f1_consistency);
  con_coins_ptrs.clear();
  con_coins_ptrs.push_back(f1_con_coins);
  temp_arr_ptrs.clear();
  temp_arr_ptrs.push_back(temp_arr);
  answers_rfetch_ptrs.clear();
  answers_rfetch_ptrs.push_back(f1_answers);

  answers_ptrs.clear();
  answers_ptrs.push_back(f1_answers);

  Q_list.clear();
  int query_id = 0;
  for(int j=0; j<NUM_REPS_LIN; j++) {
    Q_list.push_back(query_id++);
    Q_list.push_back(query_id++);
    Q_list.push_back(query_id++);
  }
  query_id +=2;

  Q_list.push_back(query_id++);
  Q_list.push_back(query_id++);
}

void MatrixCubicQVerifier::create_input() {
  // as many computations as inputs
  for (int k=0; k<batch_size; k++) {
    v->get_random_rational_vec(size_input, A_q, num_bits_in_input, num_bits_in_input); 
    snprintf(scratch_str, BUFLEN-1, "input_b_%d", k);
    dump_vector(size_input, A_q, scratch_str);
    send_file(scratch_str);
  }
}

void MatrixCubicQVerifier::create_plain_queries() {
  uint32_t m2 = input_size*input_size;

  // keeps track of #filled coins
  int f1_con_filled = -1;

  for (int rho=0; rho<num_repetitions; rho++) {
    if (rho == 0) m_plainq.begin_with_init();
    else m_plainq.begin_with_history();

    int query_id = 1;
    for (int i=0; i<NUM_REPS_LIN; i++) {
      v->create_lin_test_queries(hadamard_code_size, f1_q1, f1_q2, f1_q3, f1_consistency, f1_con_filled, f_con_coins, prime);

      f1_con_filled += 3;

      //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      //dump_vector(hadamard_code_size, f1_q1, scratch_str);
      //send_file(scratch_str);

      //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      //dump_vector(hadamard_code_size, f1_q2, scratch_str);
      //send_file(scratch_str);

      query_id++;
      //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      //dump_vector(hadamard_code_size, f1_q3, scratch_str);
      //send_file(scratch_str);

      // use one of the linearity queries as self correction queries
      if (i == 0) {
        for (int i=0; i<hadamard_code_size; i++)
          mpz_set(f1_q4[i], f1_q1[i]);
      }
    }

    // f1_q1 = q3
    // f1_q2 = q4
    // f2_q1 = q1
    // f2_q2 = q2
    mpz_set_ui(f_con_coins[f1_con_filled+1], 0);
    mpz_set_ui(f_con_coins[f1_con_filled+2], 0);

    f1_con_filled += 2;
    query_id += 2;
    v->create_corr_test_queries_vproduct(input_size, f2_q1, f2_q2, f1_q1, f1_q4,
                                         f1_consistency, f1_con_filled, f_con_coins, prime);

    f1_con_filled += 1;

    //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    //dump_vector(hadamard_code_size, f1_q1, scratch_str);
    //send_file(scratch_str);

    m_plainq.end();

    // compute answers to f1(q1) and f1(q2) locally now itself
    if (rho == 0) m_runtests.begin_with_init();
    else m_runtests.begin_with_history();
    for (int b=0; b<batch_size; b++) {
      mpz_set_ui(a1, 0);
      mpz_set_ui(a2, 0);
      
      snprintf(scratch_str, BUFLEN-1, "input_b_%d", b);
      load_vector(2*input_size*input_size, A_q, scratch_str);

      snprintf(scratch_str, BUFLEN-1, "input_z_b_%d", b);
      load_vector(2*input_size*input_size, A, scratch_str);

      if (verify_conversion_to_z(2*input_size*input_size, A, A_q, prime) == false) {
        cout<<"Verification failed"<<endl;
        return;
      }

      int index;
      mpz_set_ui(a3, 0);
      for (int k=0; k<input_size; k++) {
        // dot product of k^th row of A with k^th row of f2_q1
        //\sum_{j=1}{m}{A[k][j] \cdot f2_q1[k][j]}
        mpz_set_ui(a1, 0);
        mpz_set_ui(a2, 0);

        for (int j=0; j<input_size; j++) {
          index = j*input_size+k;
          mpz_mul(temp, A[index], f2_q1[index]);
          mpz_add(a1, a1, temp);
        }

        for (int i=0; i<input_size; i++) {
          index = k*input_size+i;
          mpz_mul(temp, B[index], f2_q2[index]);
          mpz_add(a2, a2, temp);
        }

        mpz_mul(temp, a1, a2);
        mpz_add(a3, a3, temp);
        mpz_mod(a3, a3, prime);
      }

      snprintf(scratch_str, BUFLEN-1, "corr_answer_b_%d_r_%d", b, rho);
      dump_scalar(a3, scratch_str);
    }
    m_runtests.end();

    // circuit test
    m_plainq.begin_with_history();
    v->get_random_vec_pub(input_size*input_size, gamma, prime);

    for (int i=0; i<hadamard_code_size; i++)
      mpz_set_ui(f1_q1[i], 0);

    int index, index2;
    for (int i=0; i<input_size; i++) {
      for (int j=0; j<input_size; j++) {
        // add gamma[i*input_size+j] to all the cells in query
        index2 = i*input_size+j;
        for (int k=0; k<input_size; k++) {
          index = index2 * input_size+k;
          mpz_add(f1_q1[index], f1_q1[index], gamma[index2]);
        }
      }
    }

    for (int i=0; i<input_size*input_size*input_size; i++)
      mpz_mod(f1_q1[i], f1_q1[i], prime);

    v->create_ckt_test_queries(hadamard_code_size, f1_q1, f1_q3, f1_q4, f1_consistency,
                               f1_con_filled, f_con_coins, prime);

    //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    //dump_vector(hadamard_code_size, f1_q3, scratch_str);
    //send_file(scratch_str);

    f1_con_filled += 1;

    m_plainq.end();

    m_runtests.begin_with_history();
    // finally compute c
    for (int i=0; i<batch_size; i++) {
      snprintf(scratch_str, BUFLEN-1, "output_z_b_%d", i);
      load_vector(input_size*input_size, C, scratch_str);

      snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
      load_vector(input_size*input_size, C_q, scratch_str);
      
      if (verify_conversion_to_z(input_size*input_size, C, C_q, prime) == false) {
        cout<<"Verification failed"<<endl;
        return;
      }

      int c_index = i * num_repetitions + rho;
      mpz_set_ui(c_values[c_index], 0);

      for (int j=0; j<input_size*input_size; j++) {
        mpz_neg(temp, gamma[j]);
        mpz_mul(temp, temp, C[j]);
        mpz_add(c_values[c_index], c_values[c_index], temp);
      }
    }
    m_runtests.end();
  }

  dump_vector(hadamard_code_size, f1_consistency, (char *)"f1_consistency_query");
  send_file((char *)"f1_consistency_query");
}

void MatrixCubicQVerifier::populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta) {
  //mpz_t *t1 = f1_answers;
  //mpz_t *t = f_answers;
  //uint32_t i = rho * num_lin_pcp_queries;
  //mpz_set(t1[i+0], t[i+Q3]);
  //mpz_set(t1[i+1], t[i+Q4]);
  //mpz_set(t1[i+2], t[i+Q5]);

  //mpz_set(f_con_coins[i+Q3], f1_con_coins[i+0]);
  //mpz_set(f_con_coins[i+Q4], f1_con_coins[i+1]);
  //mpz_set(f_con_coins[i+Q5], f1_con_coins[i+2]);
}

bool MatrixCubicQVerifier::run_correction_and_circuit_tests(uint32_t beta) {
  bool result = true;
  bool lin;

  for (int rho=0; rho<num_repetitions; rho++) {

    for (int i=0; i<NUM_REPS_LIN; i++) {
      int base = i*NUM_LIN_QUERIES;
      lin = v->lin_test(f_answers[rho*num_lin_pcp_queries + base + Q1],
                        f_answers[rho*num_lin_pcp_queries + base + Q2],
                        f_answers[rho*num_lin_pcp_queries + base + Q3],
                        prime);

#if VERBOSE == 1
      if (false == lin)
        cout<<"LOG: F failed the linearity test"<<endl;
      else
        cout<<"LOG: F passed the linearity test"<<endl;
#endif
      result = result & lin;
    }

    // Quad Correction test and Circuit test
    mpz_set_ui(temp, 1);
    mpz_t ans;
    mpz_init_set_ui(ans, 0);
    snprintf(scratch_str, BUFLEN-1, "corr_answer_b_%d_r_%d", beta, rho);
    load_scalar(ans, scratch_str);
    bool cor1 = v->corr_test(ans /*f2_answers[rho*NUM_LIN_PCP_QUERIES + Q1]*/, temp,
                             f_answers[rho*num_lin_pcp_queries + Q6],
                             f_answers[rho*num_lin_pcp_queries + Q1], prime);

#if VERBOSE == 1
    if (false == cor1)
      cout<<"LOG: F1, F2 failed the correction test"<<endl;
    else
      cout<<"LOG: F1, F2 passed correction test"<<endl;
#endif

    result &= cor1;

    mpz_set(ckt_answers[0], f_answers[rho*num_lin_pcp_queries + Q7]);
    mpz_set(ckt_answers[1], f_answers[rho*num_lin_pcp_queries + Q1]);
    bool ckt2 = v->ckt_test(2, ckt_answers,
                            c_values[beta * num_repetitions + rho], prime);

#if VERBOSE == 1
    if (false == ckt2)
      cout <<"LOG: F1 failed the circuit test"<<endl;
    else
      cout <<"LOG: F1 passed the circuit test"<<endl;
#endif

    result &= ckt2;
  }
  return result;
}
