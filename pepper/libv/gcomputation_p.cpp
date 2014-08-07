#include <libv/gcomputation_p.h>

GComputationProver::GComputationProver(int ph, int b_size, int num_r, int i_size,
                                       int o_size, int num_v, int num_c, const char *prover_name, const char *file_name_f1_index)
  : ComputationProver(ph, b_size, num_r, size_input, prover_name) {
  num_vars = num_v;
  size_f1_vec = num_v;
  size_f2_vec = num_v * num_v;
  num_cons = num_c;
  size_output = o_size;
  size_input = i_size;

  num_local_runs = NUM_LOCAL_RUNS;
  init_state(file_name_f1_index);
}

GComputationProver::~GComputationProver() {
  clear_vec(num_lin_pcp_queries, f_answers);
  clear_vec(size_input+size_output, input_output_q);
  clear_vec(size_input+size_output, input_output);
  clear_scalar(temp);
  clear_scalar(temp2);
  clear_scalar(temp_q);
  clear_scalar(temp_q2);
  clear_scalar(temp_q3);
  clear_vec(temp_stack_size, temp_qs);
  clear_scalar(answer);
  clear_scalar(neg);
  clear_vec(expansion_factor, dotp);

  clear_vec(size_f1_vec, F1);
  clear_vec(size_f1_vec, F1_q);
  clear_vec(size_f2_vec, F2);
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
  delete[] F1_index;
  clear_vec(num_cons, alpha);
}

void GComputationProver::init_state(const char *file_name_f1_index) {
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
  temp_stack_size = 16;
  crypto_in_use = CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;

  Prover::init_state();

  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;
  alloc_init_vec(&f_answers, num_lin_pcp_queries);
  
  alloc_init_vec(&alpha, num_cons);

  alloc_init_vec(&input_output_q, size_input+size_output);
  input_q = &input_output_q[0];
  output_q = &input_output_q[size_input];

  alloc_init_vec(&input_output, size_input+size_output);
  input = &input_output[0];
  output = &input_output[size_input];

  alloc_init_scalar(temp);
  alloc_init_scalar(temp2);
  alloc_init_scalar(temp_q);
  alloc_init_scalar(temp_q2);
  alloc_init_scalar(temp_q3);
  alloc_init_vec(&temp_qs, temp_stack_size);
  alloc_init_scalar(answer);
  alloc_init_scalar(neg);
  alloc_init_vec(&dotp, expansion_factor);

  alloc_init_vec(&F1, size_f1_vec);
  alloc_init_vec(&F1_q, size_f1_vec);
  alloc_init_vec(&F2, size_f2_vec);
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

  F1_index = new uint32_t[size_f1_vec];
  load_vector(size_f1_vec, F1_index, file_name_f1_index);

  F_ptrs.clear();
  F_ptrs.push_back(F1);
  F_ptrs.push_back(F2);
  f_q_ptrs.clear();
  f_q_ptrs.push_back(f1_q1);
  f_q_ptrs.push_back(f2_q1);
  f_q2_ptrs.clear();
  f_q2_ptrs.push_back(f1_q2);
  f_q2_ptrs.push_back(f2_q2);
  find_cur_qlengths();
}

void GComputationProver::find_cur_qlengths() {

  sizes.clear();
  sizes.push_back(size_f1_vec);
  sizes.push_back(size_f2_vec);

  qquery_sizes.clear();
  for (int i=0; i<NUM_REPS_LIN; i++) {
    qquery_sizes.push_back(size_f1_vec);
    qquery_sizes.push_back(size_f1_vec);
    qquery_sizes.push_back(size_f1_vec);
    qquery_sizes.push_back(size_f2_vec);
    qquery_sizes.push_back(size_f2_vec);
    qquery_sizes.push_back(size_f2_vec);
  }
  qquery_sizes.push_back(size_f2_vec);
  qquery_sizes.push_back(size_f1_vec);
  qquery_sizes.push_back(size_f2_vec);

  qquery_f_ptrs.clear();
  for (int i=0; i<NUM_REPS_LIN; i++) {
    qquery_f_ptrs.push_back(f1_q1);
    qquery_f_ptrs.push_back(f1_q1);
    qquery_f_ptrs.push_back(f1_q1);
    qquery_f_ptrs.push_back(f2_q1);
    qquery_f_ptrs.push_back(f2_q1);
    qquery_f_ptrs.push_back(f2_q1);
  }

  qquery_f_ptrs.push_back(f2_q1);
  qquery_f_ptrs.push_back(f1_q1);
  qquery_f_ptrs.push_back(f2_q1);

  qquery_F_ptrs.clear();
  for (int i=0; i<NUM_REPS_LIN; i++) {
    qquery_F_ptrs.push_back(F1);
    qquery_F_ptrs.push_back(F1);
    
    //set it to null so that the prover answers this query by using the
    //answers to the previous two queries
    qquery_F_ptrs.push_back(NULL);
    //qquery_F_ptrs.push_back(F1);
    
    qquery_F_ptrs.push_back(F2);
    qquery_F_ptrs.push_back(F2);
    //qquery_F_ptrs.push_back(F2);
    qquery_F_ptrs.push_back(NULL);
  }

  qquery_F_ptrs.push_back(F2);
  qquery_F_ptrs.push_back(F1);
  qquery_F_ptrs.push_back(F2);

  qquery_q_ptrs.clear();
  for (int i=0; i<NUM_REPS_PCP*NUM_LIN_PCP_QUERIES; i++)
    qquery_q_ptrs.push_back(i);
}

void GComputationProver::compute_assignment_vectors() {
  // code to fill in F2 using entries in F1
  int index;
  for (int i=0; i<size_f1_vec; i++) {
    for (int j=0; j<=i; j++) {
      index = i*size_f1_vec+j;
      mpz_mul(F2[index], F1[i], F1[j]);
      mpz_mod(F2[index], F2[index], prime);
    }
  }

  for (int i=0; i<size_f1_vec; i++) {
    for (int j=i+1; j<size_f1_vec; j++) {
      index = i*size_f1_vec+j;
      mpz_set(F2[index], F2[j*size_f1_vec+i]);
    }
  }
}

void GComputationProver::prover_computation_commitment() {

  load_vector(expansion_factor*size_f1_vec, f1_commitment, (char *)"f1_commitment_query", FOLDER_WWW_DOWNLOAD);
  load_vector(expansion_factor*size_f2_vec, f2_commitment, (char *)"f2_commitment_query", FOLDER_WWW_DOWNLOAD);


  // execute the computation
  for (int i=batch_start; i<=batch_end; i++) {
    if (i == batch_start)
      m_computation.begin_with_init();
    else
      m_computation.begin_with_history();

    cout << "Running baseline" << endl;
    for (int g=0; g<num_local_runs; g++) {
      snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
      load_vector(size_input, input, scratch_str, FOLDER_WWW_DOWNLOAD);

      snprintf(scratch_str, BUFLEN-1, "input_q_b_%d", i);
      load_vector(size_input, input_q, scratch_str, FOLDER_WWW_DOWNLOAD);

      exogenous_checker->baseline(input_q, size_input, output_q, size_output);

      snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
      dump_vector(size_output, output_q, scratch_str, FOLDER_WWW_DOWNLOAD);
    }
    m_computation.end();



    if (i == batch_start)
      m_interpret_cons.begin_with_init();
    else
      m_interpret_cons.begin_with_history();

    snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
    load_vector(size_input, input, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "input_q_b_%d", i);
    load_vector(size_input, input_q, scratch_str, FOLDER_WWW_DOWNLOAD);

    for (int j=0; j<size_output; j++) {
      mpz_set_ui(output[j], 0);
      mpq_set_ui(output_q[j], 0, 1);
    }

    interpret_constraints();

    // start saving the state
    snprintf(scratch_str, BUFLEN-1, "output_q_b_%d", i);
    dump_vector(size_output, output_q, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
    dump_vector(size_output, output, scratch_str, FOLDER_WWW_DOWNLOAD);
    m_interpret_cons.end();

    if (i == batch_start)
      m_proofv_creation.begin_with_init();
    else
      m_proofv_creation.begin_with_history();

    compute_assignment_vectors();

    m_proofv_creation.end();

    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    dump_vector(size_f1_vec, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", i);
    dump_vector(size_f2_vec, F2, scratch_str, FOLDER_WWW_DOWNLOAD);

    //May return immediately if exogenous checking is disabled
    bool passed_test = exogenous_checker->exogenous_check(input, input_q, size_input, output, output_q, size_output, prime);
    if (passed_test) {
      cout << "Exogenous check passed." << endl;
    } else {
      cout << "Exogenous check failed." << endl;
    }
  }
  
  for (int i=batch_start; i<=batch_end; i++) {
    if (i == 0)
      m_answer_queries.begin_with_init();
    else
      m_answer_queries.begin_with_history();

    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    load_vector(size_f1_vec, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", i);
    load_vector(size_f2_vec, F2, scratch_str, FOLDER_WWW_DOWNLOAD);

    if (crypto_in_use == CRYPTO_ELGAMAL)
      v->dot_product_enc(size_f1_vec, f1_commitment, F1, dotp[0], dotp[1]);

    if (crypto_in_use == CRYPTO_ELGAMAL)
      v->dot_product_enc(size_f2_vec, f2_commitment, F2, dotp2[0], dotp2[1]);
    v->add_enc(dotp[0], dotp[1], dotp[0], dotp[1], dotp2[0], dotp2[1]);

    snprintf(scratch_str, BUFLEN-1, "f_commitment_answer_b_%d", i);
    dump_vector(expansion_factor, dotp, scratch_str, FOLDER_WWW_DOWNLOAD);
    m_answer_queries.end();
  }
  clear_vec(expansion_factor*size_f1_vec, f1_commitment); 
  clear_vec(expansion_factor*size_f2_vec, f2_commitment); 
}

// TODO: find a way to share the following set of functions with the
// verifier's codebase. Perhaps by creating a new class whose object
// will be created by both the Ginger verifier and the Ginger prover 
void GComputationProver::parse_gamma12(const char *file_name) {
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

void GComputationProver::create_ckt_queries(int rho_i) {
  create_gamma12();

  //f1_q2 stores gamma_1
  //f1_q3 stores the output
  //f1_q4 is self-correction query
  v->create_ckt_test_queries(size_f1_vec, f1_q2, f1_q3, f1_q4,
                             NULL, 0, NULL, prime);
  //f2_q1 stores gamma_2
  //f2_q3 stores output
  //f2_q4 is self-correction query
  v->create_ckt_test_queries(size_f2_vec, f2_q1, f2_q3, f2_q4, 
                             NULL, 0, NULL, prime);

  snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
  dump_vector(size_f1_vec, f1_q3, scratch_str);

  snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
  dump_vector(size_f2_vec, f2_q3, scratch_str);

  m_plainq.end();
}

void GComputationProver::create_qct_queries(int rho_i) {
  // supply the linearity queries as self-correction queries to the smaller
  // part of proof vector
  v->create_corr_test_queries_reuse(size_f1_vec, f1_q4, size_f1_vec, f1_q5,
                                    f2_q1, f2_q4, NULL, NULL, NULL,
                                    0, NULL, 0, NULL,
                                    0, NULL, prime, false);

  snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
  dump_vector(size_f2_vec, f2_q1, scratch_str);
}

void GComputationProver::create_lin_queries(int rho_i) {
  if (rho_i == 0) m_plainq.begin_with_init();
  else m_plainq.begin_with_history();

  query_id = 1;
  for (int i=0; i<NUM_REPS_LIN; i++) {

    v->create_lin_test_queries(size_f1_vec, f1_q1, f1_q2, f1_q3, NULL,
                               0, NULL, prime);

    v->create_lin_test_queries(size_f2_vec, f2_q1, f2_q2, f2_q3, NULL,
                               0, NULL, prime);

    //TODO: can be folded into a function
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    dump_vector(size_f1_vec, f1_q1, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    dump_vector(size_f1_vec, f1_q2, scratch_str);

    query_id++;
    //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    //dump_vector(size_f1_vec, f1_q3, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    dump_vector(size_f2_vec, f2_q1, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    dump_vector(size_f2_vec, f2_q2, scratch_str);

    query_id++;
    //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho_i);
    //dump_vector(size_f2_vec, f2_q3, scratch_str);

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

void GComputationProver::deduce_queries() {
  for (int rho=0; rho<num_repetitions; rho++) {
    create_lin_queries(rho);
    create_qct_queries(rho);
    create_ckt_queries(rho);
  }
}
