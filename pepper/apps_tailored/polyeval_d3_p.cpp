#include <apps_tailored/polyeval_d3_p.h>

PolyEvalD3Prover::
PolyEvalD3Prover(int ph, int b_size, int num_r, int input_size)
  : Prover(ph, b_size, NUM_REPS_PCP, input_size, PROVER_NAME) {
  init_state();
}

void PolyEvalD3Prover::
init_state() {
  num_bits_in_prime = 192;
  num_bits_in_input = 12;
  crypto_in_use= CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;

  Prover::init_state();
  size_f1_vec = m;
  size_f2_vec = m*m;
  size_f3_vec = m*m*m;
  num_coefficients = (size_f3_vec + 3*size_f2_vec + 2*size_f1_vec)/6 + (size_f1_vec * size_f1_vec + 3*size_f1_vec)/2 + 1;
  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;
  num_local_runs = NUM_LOCAL_RUNS;

  alloc_init_vec(&F1, size_f1_vec);
  alloc_init_vec(&F2, size_f2_vec);
  alloc_init_vec(&F3, size_f3_vec);
  alloc_init_vec(&output, expansion_factor);
  alloc_init_vec(&variables, size_f1_vec);
  alloc_init_vec(&coefficients, num_coefficients);
  alloc_init_vec(&f1_q1, size_f1_vec);
  alloc_init_vec(&f1_q2, size_f1_vec);
  alloc_init_vec(&f1_q3, size_f1_vec);
  alloc_init_vec(&f1_q4, size_f1_vec);
  alloc_init_vec(&f1_q5, size_f1_vec);
  alloc_init_vec(&F1, size_f1_vec);
  alloc_init_vec(&f1_commitment, expansion_factor*m);
  alloc_init_vec(&f1_consistency, size_f1_vec);
  alloc_init_vec(&f2_q1, size_f2_vec);
  alloc_init_vec(&f2_q2, size_f2_vec);
  alloc_init_vec(&f2_q3, size_f2_vec);
  alloc_init_vec(&f2_q4, size_f2_vec);
  alloc_init_vec(&F2, size_f2_vec);
  alloc_init_vec(&f2_commitment, expansion_factor*size_f2_vec);
  alloc_init_vec(&f2_consistency, size_f2_vec);
  alloc_init_vec(&f3_q1, size_f3_vec);
  alloc_init_vec(&f3_q2, size_f3_vec);
  alloc_init_vec(&f3_q3, size_f3_vec);
  alloc_init_vec(&f3_q4, size_f3_vec);
  alloc_init_vec(&F3, size_f3_vec);
  alloc_init_vec(&f3_commitment, expansion_factor*size_f3_vec);
  alloc_init_vec(&f3_consistency, size_f3_vec);
  alloc_init_vec(&alpha, size_f1_vec+1);
  alloc_init_vec(&f_answers, num_lin_pcp_queries);
  alloc_init_scalar(neg);
  alloc_init_scalar(answer);
  alloc_init_scalar(temp);
  alloc_init_scalar(temp2);

  F_ptrs.clear();
  F_ptrs.push_back(F1);
  F_ptrs.push_back(F2);
  F_ptrs.push_back(F3);
  
  f_q_ptrs.clear();
  f_q_ptrs.push_back(f1_q1);
  f_q_ptrs.push_back(f2_q1);
  f_q_ptrs.push_back(f3_q1);
  f_q2_ptrs.clear();
  f_q2_ptrs.push_back(f1_q2);
  f_q2_ptrs.push_back(f2_q2);
  f_q2_ptrs.push_back(f3_q2);
  f_q3_ptrs.clear();
  f_q3_ptrs.push_back(f1_q3);
  f_q3_ptrs.push_back(f2_q3);
  f_q3_ptrs.push_back(f3_q3); 

  find_cur_qlengths();
}

void PolyEvalD3Prover::
find_cur_qlengths() {
  num_coefficients = (size_f3_vec + 3*size_f2_vec + 2*size_f1_vec)/6 + (size_f1_vec * size_f1_vec + 3*size_f1_vec)/2 + 1;
  sizes.clear();
  sizes.push_back(size_f1_vec);
  sizes.push_back(size_f2_vec);
  sizes.push_back(size_f3_vec);

  qquery_f_ptrs.clear();
  qquery_F_ptrs.clear();
  qquery_q_ptrs.clear();
  qquery_sizes.clear(); 
 
  int query_id = 0;
  for(int j=0; j<NUM_REPS_LIN; j++) {
    qquery_q_ptrs.push_back(query_id++);
    qquery_q_ptrs.push_back(query_id++);
    qquery_q_ptrs.push_back(query_id++);

    qquery_sizes.push_back(size_f1_vec);
    qquery_sizes.push_back(size_f1_vec);
    qquery_sizes.push_back(size_f1_vec);

    qquery_F_ptrs.push_back(F1);
    qquery_F_ptrs.push_back(F1);
    // pass the next entry as null to use tell the prover to compute the
    // answer to the next query as the sum of the answers to the
    // previous two queries. See how the prover answers queries in
    // libv/prover.cpp for more details 
    // qquery_F_ptrs.push_back(F1);
    qquery_F_ptrs.push_back(NULL);
    
    qquery_f_ptrs.push_back(f1_q1);
    qquery_f_ptrs.push_back(f1_q1);
    qquery_f_ptrs.push_back(f1_q1);
  
    // for \pi_2
    qquery_q_ptrs.push_back(query_id++);
    qquery_q_ptrs.push_back(query_id++);
    qquery_q_ptrs.push_back(query_id++);

    qquery_sizes.push_back(size_f2_vec);
    qquery_sizes.push_back(size_f2_vec);
    qquery_sizes.push_back(size_f2_vec);

    qquery_F_ptrs.push_back(F2);
    qquery_F_ptrs.push_back(F2);
    qquery_F_ptrs.push_back(NULL);

    qquery_f_ptrs.push_back(f2_q1);
    qquery_f_ptrs.push_back(f2_q1);
    qquery_f_ptrs.push_back(f2_q1);
   
    qquery_q_ptrs.push_back(query_id++);
    qquery_q_ptrs.push_back(query_id++);
    qquery_q_ptrs.push_back(query_id++);

    qquery_sizes.push_back(size_f3_vec);
    qquery_sizes.push_back(size_f3_vec);
    qquery_sizes.push_back(size_f3_vec);

    qquery_F_ptrs.push_back(F3);
    qquery_F_ptrs.push_back(F3);
    qquery_F_ptrs.push_back(NULL);

    qquery_f_ptrs.push_back(f3_q1);
    qquery_f_ptrs.push_back(f3_q1);
    qquery_f_ptrs.push_back(f3_q1);
  }

  qquery_q_ptrs.push_back(query_id++);
  qquery_q_ptrs.push_back(query_id++);
  qquery_q_ptrs.push_back(query_id++);
  qquery_q_ptrs.push_back(query_id++);
  qquery_q_ptrs.push_back(query_id++);
  qquery_q_ptrs.push_back(query_id++);
 
  qquery_sizes.push_back(size_f2_vec);
  qquery_sizes.push_back(size_f3_vec);
  qquery_sizes.push_back(size_f1_vec);
  qquery_sizes.push_back(size_f2_vec);
  qquery_sizes.push_back(size_f3_vec);

  qquery_F_ptrs.push_back(F2);
  qquery_F_ptrs.push_back(F3);
  qquery_F_ptrs.push_back(F1);
  qquery_F_ptrs.push_back(F2);
  qquery_F_ptrs.push_back(F3);

  qquery_f_ptrs.push_back(f2_q1);
  qquery_f_ptrs.push_back(f3_q1);
  qquery_f_ptrs.push_back(f1_q1);
  qquery_f_ptrs.push_back(f2_q1);
  qquery_f_ptrs.push_back(f3_q1);
}

// COMPUTATION
void PolyEvalD3Prover::
computation_assignment(mpz_t output) {
  // perform polynomial evaluation
  // first, compute the cubic assignment, F3 = \op{m}{m}{m}
  // then, compute the quadratic assignment, F2 = \op{m}{m}
  int index = 0, index2 = 0, index3 = 0;

  // compute F1 and F2
  for (int i=0; i<m; i++) {
    mpz_set(F1[i], variables[i]);
    mpz_mod(F1[i], F1[i], prime);
    for (int j=0; j<=i; j++) {
      index = m*i+j;
      mpz_mul(F2[index], variables[i], variables[j]);
      mpz_mod(F2[index], F2[index], prime);
    }
  }
  for (int i=0; i<m; i++) {
    for (int j=i+1; j<m; j++) {
      index = m*i + j;
      index2 = m*j+i;
      mpz_set(F2[index], F2[index2]);
    }
  }

  // next compute F3 and partial output
  mpz_set_ui(output, 0);
  index2 = 0;
  index3 = 0;
  for (int i=0; i<m; i++) {
    for (int j=0; j<=i; j++) {
      index = m*i + j;
      for (int k=0; k<=j; k++) {
        mpz_mul(temp, F2[index], variables[k]);
        mpz_mod(temp, temp, prime);
        
        index2 = (i*m+j)*m+k;
        mpz_set(F3[index2], temp);

        index2 = (j*m+i)*m+k;
        mpz_set(F3[index2], temp);

        index2 = (j*m+k)*m+i;
        mpz_set(F3[index2], temp);

        index2 = (k*m+j)*m+i;
        mpz_set(F3[index2], temp);

        index2 = (k*m+i)*m+j;
        mpz_set(F3[index2], temp);

        index2 = (i*m+k)*m+j;
        mpz_set(F3[index2], temp);
      }
    }
  }
}

void PolyEvalD3Prover::
computation_polyeval(mpz_t output) {
  // perform polynomial evaluation
  // first, compute the cubic assignment, F3 = \op{m}{m}{m}
  // then, compute the quadratic assignment, F2 = \op{m}{m}
  int index = 0, index2 = 0, index3 = 0;

  // compute F1 and F2
  for (int i=0; i<m; i++) {
    mpz_set(F1[i], variables[i]);
    for (int j=0; j<=i; j++) {
      index = m*i+j;
      mpz_mul(F2[index], variables[i], variables[j]);
    }
  }
  // next compute F3 and partial output
  mpz_set_ui(output, 0);
  index2 = 0;
  index3 = 0;
  for (int i=0; i<m; i++) {
    for (int j=0; j<=i; j++) {
      index = m*i + j;
      for (int k=0; k<=j; k++) {
        mpz_mul(F3[(i*m+j)*m+k], F2[index], variables[k]);

        // partial output
        mpz_mul(temp2, coefficients[index3], F3[(i*m+j)*m+k]);
        mpz_add(output, output, temp2);
        index3++;
      }
    }
  }

  // now compute the output;
  for (int i=0; i<m; i++) {
    for (int j=0; j<=i; j++) {
      int index = m*i+j;
      mpz_mul(temp, coefficients[index3], F2[index]);
      mpz_add(output, output, temp);
      index3++;
    }
  }

  // now the linear term
  for (int i=0; i<m; i++) {
    mpz_mul(temp, coefficients[index3+i], F1[i]);
    mpz_add(output, output, temp);
  }

  index3 += m;

  // now the constant term
  mpz_add(output, output, coefficients[index3]);
  mpz_mod(output, output, prime);
}

//PROVER's CODE
void PolyEvalD3Prover::
prover_computation_commitment() {
  load_vector(expansion_factor*m, f1_commitment, (char *)"f1_commitment_query", FOLDER_WWW_DOWNLOAD);
  load_vector(expansion_factor*size_f2_vec, f2_commitment, (char *)"f2_commitment_query", FOLDER_WWW_DOWNLOAD);
  load_vector(expansion_factor*size_f3_vec, f3_commitment, (char *)"f3_commitment_query", FOLDER_WWW_DOWNLOAD);

  m_computation.begin_with_init();
  //for (int k=0; k<INNER_LOOP_SMALL; k++)
  load_vector(num_coefficients, coefficients, (char *)"coefficients", FOLDER_WWW_DOWNLOAD);
  m_computation.end();

  for (int i=batch_start; i<=batch_end; i++) {
    m_computation.begin_with_history();

    //for (int k=0; k<INNER_LOOP_SMALL; k++)
    {
      snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
      load_vector(m, variables, scratch_str, FOLDER_WWW_DOWNLOAD);

      for (int j=0; j<expansion_factor; j++)
        mpz_set_ui(output[j], 0);

      computation_polyeval(output[0]);

      // start saving the state
      snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
      dump_scalar(output[0], scratch_str, FOLDER_WWW_DOWNLOAD);
    }
    m_computation.end();
    computation_assignment(output[0]);
    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    dump_vector(m, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", i);
    dump_vector(size_f2_vec, F2, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f3_assignment_vector_b_%d", i);
    dump_vector(size_f3_vec, F3, scratch_str, FOLDER_WWW_DOWNLOAD);
  }

  for (int i=batch_start; i<=batch_end; i++) {
    if (i == 0)
      m_answer_queries.begin_with_init();
    else
      m_answer_queries.begin_with_history();


    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    load_vector(m, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", i);
    load_vector(size_f2_vec, F2, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f3_assignment_vector_b_%d", i);
    load_vector(size_f3_vec, F3, scratch_str, FOLDER_WWW_DOWNLOAD);

    v->dot_product_enc(m, f1_commitment, F1, dotp[0], dotp[1]);
    v->dot_product_enc(size_f2_vec, f2_commitment, F2, output[0], output[1]);
    v->add_enc(dotp[0], dotp[1], dotp[0], dotp[1], output[0], output[1]);
    v->dot_product_enc(size_f3_vec, f3_commitment, F3, output[0], output[1]);
    v->add_enc(dotp[0], dotp[1], dotp[0], dotp[1], output[0], output[1]);

    snprintf(scratch_str, BUFLEN-1, "f_commitment_answer_b_%d", i);
    dump_vector(expansion_factor, dotp, scratch_str, FOLDER_WWW_DOWNLOAD);
    m_answer_queries.end();
  }
}

void PolyEvalD3Prover::deduce_queries() {
  int f_con_filled = -1;
  load_vector(num_coefficients, coefficients, (char *)"coefficients");
  
  for (int rho=0; rho<num_repetitions; rho++) {
    if (rho == 0) m_plainq.begin_with_init();
    else m_plainq.begin_with_history();
    int query_id = 1;

    for (int i=0; i<NUM_REPS_LIN; i++) {
      v->create_lin_test_queries(size_f1_vec, f1_q1, f1_q2, f1_q3, NULL,
          f_con_filled, NULL, prime);
      f_con_filled += 3;
      
      v->create_lin_test_queries(size_f2_vec, f2_q1, f2_q2, f2_q3, NULL,
          f_con_filled, NULL, prime);
      f_con_filled += 3;
 
      v->create_lin_test_queries(size_f3_vec, f3_q1, f3_q2, f3_q3, NULL,
          f_con_filled, NULL, prime);
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
                                f2_q1, f2_q4, NULL, NULL, NULL,
                                f_con_filled, NULL, f_con_filled, NULL,
                                f_con_filled, NULL, prime, false);
    f_con_filled += 1;
 
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f2_vec, f2_q1, scratch_str);
    //send_file(scratch_str);

    v->create_corr_test_queries_reuse(size_f1_vec, f1_q4,
                                size_f2_vec, f2_q4, f3_q1, f3_q4,
                                NULL, NULL, NULL, f_con_filled,
                                NULL, f_con_filled, NULL, f_con_filled,
                                NULL, prime, false);

    f_con_filled += 1;
 
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f3_vec, f3_q1, scratch_str);
    //send_file(scratch_str);


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
                               NULL, f_con_filled, NULL, prime);
    f_con_filled += 1;
    v->create_ckt_test_queries(size_f2_vec, f2_q1, f2_q3,
                               f2_q4, NULL, f_con_filled, NULL, prime);
    f_con_filled += 1;
    v->create_ckt_test_queries(size_f3_vec, f3_q1,
                               f3_q3, f3_q4, NULL, f_con_filled, NULL, prime);
    f_con_filled += 1;
 
    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f1_vec, f1_q3, scratch_str);
    //send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f2_vec, f2_q3, scratch_str);
    //send_file(scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f3_vec, f3_q3, scratch_str);
    //send_file(scratch_str);
    m_plainq.end();

  }
}
