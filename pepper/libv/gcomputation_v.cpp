#include <libv/gcomputation_v.h>

GComputationVerifier::GComputationVerifier(int batch, int reps, int ip_size, int op_size, int optimize_answers, int num_v, int num_c, char *prover_url, char *prover_name, const char *file_name_f1_index)
  : Verifier(batch, NUM_REPS_PCP, ip_size, optimize_answers, prover_url, prover_name) {
  size_f1_vec = num_v;
  size_f2_vec = num_v * num_v;
  num_cons = num_c;
  size_input = ip_size;
  size_output = op_size;
  num_verification_runs = NUM_VERIFICATION_RUNS;
  init_state(file_name_f1_index);
}

GComputationVerifier::~GComputationVerifier() {
  clear_vec(size_input, input_q);
  clear_vec(size_input+size_output, input_output);
  
  clear_scalar(temp);
  clear_scalar(neg);
  clear_scalar(neg_i);
  clear_vec(num_repetitions * num_lin_pcp_queries, f_answers);
  clear_vec(num_cons, alpha);
  clear_vec(4, ckt_answers);
  clear_vec(expansion_factor, temp_arr);
  clear_vec(expansion_factor, temp_arr2);  
  delete[] F1_index;
}

void GComputationVerifier::init_state(const char *file_name_f1_index) {
  string str(file_name_f1_index);
  if (str.find("bisect_sfdl") != std::string::npos) {
    num_bits_in_prime = 220;
  } else if (str.find("pd2_sfdl") != std::string::npos) {
    num_bits_in_prime = 220;
  } else {
    num_bits_in_prime = 128;
  }
  cout<<"LOG: Using a prime of size "<<num_bits_in_prime<<endl;
 
  num_bits_in_input = 32;
  crypto_in_use = CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;

  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;

  Verifier::init_state();

  alloc_init_vec(&ckt_answers, 4);

  alloc_init_vec(&input_output, size_input+size_output);

  input = &input_output[0];
  output = &input_output[size_input];
  alloc_init_vec(&alpha, num_cons);

  alloc_init_vec(&input_q, size_input);

  alloc_init_scalar(temp);
  alloc_init_scalar(neg);
  alloc_init_scalar(neg_i);

  alloc_init_vec(&f_answers, num_repetitions * num_lin_pcp_queries);
  alloc_init_vec(&f1_commitment, expansion_factor*size_f1_vec);
  alloc_init_vec(&f2_commitment, expansion_factor*size_f2_vec);
  alloc_init_vec(&f1_consistency, size_f1_vec);
  alloc_init_vec(&f2_consistency, size_f2_vec);
  alloc_init_vec(&f1_q1, size_f1_vec);
  alloc_init_vec(&f1_q2, size_f1_vec);
  alloc_init_vec(&f1_q3, size_f1_vec);
  alloc_init_vec(&f1_q4, size_f1_vec);
  alloc_init_vec(&f1_q5, size_f1_vec);

  alloc_init_vec(&f2_q1, size_f2_vec);
  alloc_init_vec(&f2_q2, size_f2_vec);
  alloc_init_vec(&f2_q3, size_f2_vec);
  alloc_init_vec(&f2_q4, size_f2_vec);

  alloc_init_vec(&temp_arr, expansion_factor);
  alloc_init_vec(&temp_arr2, expansion_factor);

  F1_index = new uint32_t[size_f1_vec];
  load_vector(size_f1_vec, F1_index, file_name_f1_index);

  commitment_query_sizes.clear();
  commitment_query_sizes.push_back(size_f1_vec);
  commitment_query_sizes.push_back(size_f2_vec);


  f_commitment_ptrs.clear();
  f_commitment_ptrs.push_back(f1_commitment);
  f_commitment_ptrs.push_back(f2_commitment);

  f_consistency_ptrs.clear();
  f_consistency_ptrs.push_back(f1_consistency);
  f_consistency_ptrs.push_back(f2_consistency);

  temp_arr_ptrs.clear();
  temp_arr_ptrs.push_back(temp_arr);
  temp_arr_ptrs.push_back(temp_arr2);

  Q_list.clear();
  for (int i=0; i<NUM_REPS_PCP*NUM_LIN_PCP_QUERIES; i++)
    Q_list.push_back(i);
}

void GComputationVerifier::create_input() {
  // as many computations as inputs
  for (int k=0; k<batch_size; k++) {
    input_creator->create_input(input_q, size_input);

    snprintf(scratch_str, BUFLEN-1, "input_q_b_%d", k);
    dump_vector(size_input, input_q, scratch_str);
    send_file(scratch_str);

    convert_to_z(size_input, input, input_q, prime);

    snprintf(scratch_str, BUFLEN-1, "input_b_%d", k);
    dump_vector(size_input, input, scratch_str);
    send_file(scratch_str);
  }
}

void GComputationVerifier::parse_gamma12(const char *file_name) {
  FILE *fp = fopen(file_name, "r");
  if (fp == NULL) {
    cout<<"Cannot read gamma file"<<endl;
    exit(1);
  }

  // generate one random coefficient per constraint
  v->get_random_vec_pub(num_cons, alpha, prime);

  // empty the gamma_1 and gamma_2
  for (int i=0; i<size_f1_vec; i++)
    mpz_set_ui(f1_q2[i], 0);

  for (int i=0; i<size_f2_vec; i++)
    mpz_set_ui(f2_q1[i], 0);

  char line[BUFLEN];
  mpz_t coefficient;
  alloc_init_scalar(coefficient);
  int gamma1_or2, constraint_id, var_id_x, var_id_y;

  while (fgets(line, sizeof line, fp) != NULL) {
    if (line[0] == '\n')
      continue;
    gmp_sscanf(line, "%d %Zd %d %d %d", &gamma1_or2, coefficient, &constraint_id, &var_id_x, &var_id_y);

    if (gamma1_or2 == 1) {
      // gamma1
      mpz_mul(temp, alpha[constraint_id], coefficient);
      mpz_add(f1_q2[F1_index[var_id_x]], f1_q2[F1_index[var_id_x]], temp);
    } else if (gamma1_or2 == 2) {
      // gamma2
      mpz_mul(temp, alpha[constraint_id], coefficient);
      mpz_add(f2_q1[F1_index[var_id_x] * size_f1_vec + F1_index[var_id_y]], f2_q1[F1_index[var_id_x] * size_f1_vec + F1_index[var_id_y]], temp);
    }
    gamma1_or2 = 0;
  }
  fclose(fp);
  clear_scalar(coefficient);
}

void GComputationVerifier::parse_gamma0(const char *file_name, int rho_i) {
  char line[BUFLEN];
  char G;
  int var_id, constraint_id;

  mpz_t coefficient;
  alloc_init_scalar(coefficient);

  for (int i=0; i<batch_size; i++) {

    snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
    load_vector(size_input, input, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
    load_vector(size_output, output, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "output_q_b_%d", i);
    load_vector(size_output, output_q, scratch_str);
    
    int c_index = i * num_repetitions + rho_i;
    FILE *fp = fopen(file_name, "r");
    if (fp == NULL) {
      cout<<"Cannot read gamma0 file"<<endl;
      exit(1);
    }

    while (fgets(line, sizeof line, fp) != NULL) {
      if (line[0] == '\n')
        continue;
      gmp_sscanf(line, "%c %Zd %d %d", &G, coefficient, &constraint_id, &var_id);
      if (G == 'G') {
        mpz_mul(temp, alpha[constraint_id], coefficient);
        mpz_mul(temp, temp, input_output[var_id]);
        mpz_add(c_values[c_index], c_values[c_index], temp);
      } else if (G == 'C') {
        mpz_mul(temp, alpha[constraint_id], coefficient);
        mpz_add(c_values[c_index], c_values[c_index], temp);
      }
      G = 'X';
    }
    fclose(fp);
  }
  clear_scalar(coefficient);
}

void GComputationVerifier::create_ckt_queries(int rho_i) {
  create_gamma12();

  //f1_q2 stores gamma_1
  //f1_q3 stores the output
  //f1_q4 is self-correction query
  v->create_ckt_test_queries(size_f1_vec, f1_q2, f1_q3, f1_q4,
                             f1_consistency, f_con_filled, f_con_coins, prime);
  f_con_filled += 1;

  //f2_q1 stores gamma_2
  //f2_q3 stores output
  //f2_q4 is self-correction query
  v->create_ckt_test_queries(size_f2_vec, f2_q1, f2_q3,
                             f2_q4, f2_consistency, f_con_filled, f_con_coins, prime);
  f_con_filled += 1;

  snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
  //dump_vector(size_f1_vec, f1_q3, scratch_str);
  //send_file(scratch_str);

  snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
  //dump_vector(size_f2_vec, f2_q3, scratch_str);
  //send_file(scratch_str);

  m_plainq.end();

  if (rho_i == 0) m_runtests.begin_with_init();
  else m_runtests.begin_with_history();

  create_gamma0(rho_i);
  m_runtests.end();
}

void GComputationVerifier::create_qct_queries(int rho_i) {
  // f1_q1 = q1
  // f1_q2 = q2
  // f2_q1 = q3
  // f2_q2 = q4
  // supply the linearity queries are correction queries to the smaller
  // part of proof vector
  v->create_corr_test_queries_reuse(size_f1_vec, f1_q4, size_f1_vec, f1_q5,
                                    f2_q1, f2_q4, f1_consistency, f1_consistency, f2_consistency,
                                    f_con_filled, f_con_coins, f_con_filled, f_con_coins,
                                    f_con_filled, f_con_coins, prime, false);

  f_con_filled += 1;

  snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
  //dump_vector(size_f2_vec, f2_q1, scratch_str);
  //send_file(scratch_str);
}

void GComputationVerifier::create_lin_queries(int rho_i) {
  if (rho_i == 0) m_plainq.begin_with_init();
  else m_plainq.begin_with_history();


  query_id = 1;
  for (int i=0; i<NUM_REPS_LIN; i++) {

    v->create_lin_test_queries(size_f1_vec, f1_q1, f1_q2, f1_q3, f1_consistency,
                               f_con_filled, f_con_coins, prime);
    f_con_filled += 3;

    v->create_lin_test_queries(size_f2_vec, f2_q1, f2_q2, f2_q3, f2_consistency,
                               f_con_filled, f_con_coins, prime);
    f_con_filled += 3;

    //TODO: can be folded into a function
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    //dump_vector(size_f1_vec, f1_q1, scratch_str);
    //send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    //dump_vector(size_f1_vec, f1_q2, scratch_str);
    //send_file(scratch_str);

    query_id++;
    //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    //dump_vector(size_f1_vec, f1_q3, scratch_str);
    //send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    //dump_vector(size_f2_vec, f2_q1, scratch_str);
    //send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    //dump_vector(size_f2_vec, f2_q2, scratch_str);
    //send_file(scratch_str);

    query_id++;
    //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    //dump_vector(size_f2_vec, f2_q3, scratch_str);
    //send_file(scratch_str);

    // use one of the linearity queries as self correction queries
    if (i == 0) {
      for (int i=0; i<size_f1_vec; i++) {
        mpz_set(f1_q4[i], f1_q1[i]);
        mpz_set(f1_q5[i], f1_q2[i]);
      }

      for (int i=0; i<size_f2_vec; i++)
        mpz_set(f2_q4[i], f2_q1[i]);
    }
  }
}

void GComputationVerifier::create_plain_queries() {
  clear_vec(expansion_factor*size_f1_vec, f1_commitment);
  clear_vec(expansion_factor*size_f2_vec, f2_commitment);

  // keeps track of #filled coins
  f_con_filled = -1;
  for (int rho=0; rho<num_repetitions; rho++) {
    create_lin_queries(rho);
    create_qct_queries(rho);
    create_ckt_queries(rho);
    //populate_answers(NULL, rho, num_repetitions, 0);
  }
  
  dump_vector(size_f1_vec, f1_consistency, (char *)"f1_consistency_query");
  send_file((char *)"f1_consistency_query");

  dump_vector(size_f2_vec, f2_consistency, (char *)"f2_consistency_query");
  send_file((char *)"f2_consistency_query");

  clear_vec(size_f1_vec, f1_consistency);
  clear_vec(size_f2_vec, f2_consistency);
  clear_vec(size_f1_vec, f1_q1);
  clear_vec(size_f1_vec, f1_q2);
  clear_vec(size_f1_vec, f1_q3);
  clear_vec(size_f1_vec, f1_q4);
  clear_vec(size_f1_vec, f1_q5);
  clear_vec(size_f2_vec, f2_q1);
  clear_vec(size_f2_vec, f2_q2);
  clear_vec(size_f2_vec, f2_q3);
  clear_vec(size_f2_vec, f2_q4);
}

void GComputationVerifier::populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta) {
}

bool GComputationVerifier::run_correction_and_circuit_tests(uint32_t beta) {
  bool result = true;
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

#if VERBOSE == 1
      if (false == lin1 || false == lin2)
        cout<<"LOG: F1, F2 failed the linearity test"<<endl;
      else
        cout<<"LOG: F1, F2 passed the linearity test"<<endl;
#endif

      result = result & lin1 & lin2;
    }

    // Quad Correction test and Circuit test
    bool cor1 = v->corr_test(f_answers[rho*num_lin_pcp_queries + Q1],
                             f_answers[rho*num_lin_pcp_queries + Q2],
                             f_answers[rho*num_lin_pcp_queries + Q7],
                             f_answers[rho*num_lin_pcp_queries + Q4],
                             prime);

#if VERBOSE == 1
    if (false == cor1)
      cout<<"LOG: F1, F2 failed the correction test"<<endl;
    else
      cout<<"LOG: F1, F2 passed correction test"<<endl;
#endif

    result = result & cor1;

    mpz_set(ckt_answers[0], f_answers[rho*num_lin_pcp_queries + Q8]);
    mpz_set(ckt_answers[1], f_answers[rho*num_lin_pcp_queries + Q1]);
    mpz_set(ckt_answers[2], f_answers[rho*num_lin_pcp_queries + Q9]);
    mpz_set(ckt_answers[3], f_answers[rho*num_lin_pcp_queries + Q4]);

    bool ckt2 = v->ckt_test(4, ckt_answers, c_values[beta * num_repetitions + rho], prime);

#if VERBOSE == 1
    if (false == ckt2)
      cout <<"LOG: F1, F2 failed the circuit test"<<endl;
    else
      cout <<"LOG: F1, F2 passed the circuit test"<<endl;
#endif

    result = result & ckt2;
  }
  return result;
}
