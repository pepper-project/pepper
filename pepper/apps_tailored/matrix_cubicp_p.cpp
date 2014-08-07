#include <apps_tailored/matrix_cubicp_p.h>
#ifdef MATRIX_RATIONAL_SCALED
#define NUM_BITS_PRIME 256
#define NUM_BITS_INPUT 64
#else
#define NUM_BITS_PRIME 128
#define NUM_BITS_INPUT 32
#endif

MatrixCubicProver::
MatrixCubicProver(int ph, int b_size, int num_r, int size_input)
  : Prover(ph, b_size, NUM_REPS_PCP, size_input, PROVER_NAME) {
  init_state();
}

void MatrixCubicProver::
init_state() {
  num_bits_in_prime = NUM_BITS_PRIME;
  num_bits_in_input = NUM_BITS_INPUT;
  crypto_in_use = CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;
  cout<<"Running the prover with a prime of size "<<num_bits_in_prime<<endl;

  Prover::init_state();

  size_f1_vec = m*m*m;
  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;
  num_local_runs = NUM_LOCAL_RUNS;

  alloc_init_vec(&F1, size_f1_vec);
  alloc_init_vec(&A, 2*m*m);
  //alloc_init_vec(&B, m*m);
  B = &A[m*m];
  alloc_init_vec(&C, m*m);
  alloc_init_vec(&output, expansion_factor);
  alloc_init_vec(&f2_q1, m*m);
  alloc_init_vec(&f2_q2, m*m);
  alloc_init_vec(&gamma, m*m);
  alloc_init_vec(&f_answers, num_lin_pcp_queries);
  alloc_init_scalar(answer);
  alloc_init_vec(&f1_q1, size_f1_vec);
  alloc_init_vec(&f1_q2, size_f1_vec);
  alloc_init_vec(&f1_q3, size_f1_vec);
  alloc_init_vec(&f1_q4, size_f1_vec);
  F_ptrs.clear();
  F_ptrs.push_back(F1);

  f_q_ptrs.clear();
  f_q_ptrs.push_back(f1_q1);
  f_q_ptrs.push_back(f1_q2);
  f_q_ptrs.push_back(f1_q3);

  f_q2_ptrs.clear();
  f_q2_ptrs.push_back(f2_q1);
  f_q2_ptrs.push_back(f2_q2);

  find_cur_qlengths();
}

void MatrixCubicProver::
find_cur_qlengths() {
  sizes.clear();
  size_f1_vec = m*m*m;
  sizes.push_back(size_f1_vec);


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
  }
  query_id +=2;

  qquery_q_ptrs.push_back(query_id++);
  qquery_q_ptrs.push_back(query_id++);
  
  qquery_sizes.push_back(size_f1_vec);
  qquery_sizes.push_back(size_f1_vec);

  qquery_F_ptrs.push_back(F1);
  qquery_F_ptrs.push_back(F1);

  qquery_f_ptrs.push_back(f1_q1);
  qquery_f_ptrs.push_back(f1_q1);
}

void MatrixCubicProver::
computation_matrixmult() {
  // perform matrix multiplication
  int index, index2;
  mpz_t temp;
  mpz_init(temp);
  for (int i=0; i<m; i++) {
    for (int j=0; j<m; j++) {
      index = i*m+j;
      mpz_set_ui(C[index], 0);
      for (int k=0; k<m; k++) {
        index2 = index*m+k;
        mpz_mul(temp, A[i*m+k], B[k*m+j]);
        mpz_add(C[index], C[index], temp);
      }
    }
  }
  clear_scalar(temp);
}

void MatrixCubicProver::
computation_assignment() {
  // perform matrix multiplication
  int index, index2;
  mpz_t temp;
  mpz_init(temp);
  for (int i=0; i<m; i++) {
    for (int j=0; j<m; j++) {
      index = i*m+j;
      mpz_set_ui(C[index], 0);
      for (int k=0; k<m; k++) {
        index2 = index*m+k;
        mpz_mul(F1[index2], A[i*m+k], B[k*m+j]);
        mpz_mod(F1[index2], F1[index2], prime);
      }
    }
  }
}

//PROVER's CODE
void MatrixCubicProver::
prover_computation_commitment() {
  // init prover

  for (int i=batch_start; i<=batch_end; i++) {
    if (i == 0)
      m_computation.begin_with_init();
    else
      m_computation.begin_with_history();

    //for (int k=0; k<INNER_LOOP_SMALL; k++)
    {
      snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
      load_vector(2*m*m, A, scratch_str, FOLDER_WWW_DOWNLOAD);

      //snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
      //load_vector(m*m, B, scratch_str, FOLDER_WWW_DOWNLOAD);

      computation_matrixmult();

      // start saving the state
      snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
      dump_vector(m*m, C, scratch_str, FOLDER_WWW_DOWNLOAD);
    }
    m_computation.end();

    computation_assignment();
    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    dump_vector(m*m*m, F1, scratch_str, FOLDER_WWW_DOWNLOAD);
  }

  alloc_init_vec(&f1_commitment, expansion_factor*size_f1_vec);
  load_vector(expansion_factor*size_f1_vec, f1_commitment, (char *)"f1_commitment_query", FOLDER_WWW_DOWNLOAD);
  for (int i=batch_start; i<=batch_end; i++) {
    if (i == 0)
      m_answer_queries.begin_with_init();
    else
      m_answer_queries.begin_with_history();

    
    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    load_vector(m*m*m, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

  
    v->dot_product_enc(m*m*m, f1_commitment, F1, output[0], output[1]);

    snprintf(scratch_str, BUFLEN-1, "f_commitment_answer_b_%d", i);
    dump_vector(expansion_factor, output, scratch_str, FOLDER_WWW_DOWNLOAD);
    m_answer_queries.end();
  }
  clear_vec(expansion_factor*size_f1_vec, f1_commitment);
}

void MatrixCubicProver::deduce_queries() {
// Nothing to deduce.
   cout<<"Inside deduce queries "<<m<<endl; 
   uint32_t m2 = m*m;
 
   for (int rho=0; rho<num_repetitions; rho++) {
     if (rho == 0) m_plainq.begin_with_init();
     else m_plainq.begin_with_history();
 
     int query_id = 1;
     for (int i=0; i<NUM_REPS_LIN; i++) {
       v->create_lin_test_queries(size_f1_vec, f1_q1, f1_q2, f1_q3, NULL, 0, NULL, prime);
 
       snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
       dump_vector(size_f1_vec, f1_q1, scratch_str);
       //send_file(scratch_str);
 
       snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
       dump_vector(size_f1_vec, f1_q2, scratch_str);
       //send_file(scratch_str);
 
       snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
       dump_vector(size_f1_vec, f1_q3, scratch_str);
       //send_file(scratch_str);
 
       // use one of the linearity queries as self correction queries
       if (i == 0) {
         for (int i=0; i<size_f1_vec; i++)
           mpz_set(f1_q4[i], f1_q1[i]);
       }
     }
 
     // f1_q1 = q3
     // f1_q2 = q4
     // f2_q1 = q1
     // f2_q2 = q2
     query_id += 2; //increment since we are not dumping
     v->create_corr_test_queries_vproduct(m, f2_q1, f2_q2, f1_q1, f1_q4,
                                          NULL, 0, NULL, prime);
 
     snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
     dump_vector(size_f1_vec, f1_q1, scratch_str);
     //send_file(scratch_str);
 
     v->get_random_vec_pub(m*m, gamma, prime);
 
     for (int i=0; i<size_f1_vec; i++)
       mpz_set_ui(f1_q1[i], 0);
 
     int index, index2;
     for (int i=0; i<m; i++) {
       for (int j=0; j<m; j++) {
         // add gamma[i*m+j] to all the cells in query
         index2 = i*m+j;
         for (int k=0; k<m; k++) {
           index = index2 * m+k;
           mpz_add(f1_q1[index], f1_q1[index], gamma[index2]);
         }
       }
     }
 
     for (int i=0; i<size_f1_vec; i++)
       mpz_mod(f1_q1[i], f1_q1[i], prime);
 
     v->create_ckt_test_queries(size_f1_vec, f1_q1, f1_q3, f1_q4, NULL,
                                0, NULL, prime);
 
     snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
     dump_vector(size_f1_vec, f1_q3, scratch_str);
     //send_file(scratch_str);
     m_plainq.end();
   } 
}
