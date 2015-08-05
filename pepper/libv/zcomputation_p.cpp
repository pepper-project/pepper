#include <libv/zcomputation_p.h>
#include <storage/ram_impl.h>
#include <fstream>
#include <common/memory.h>

#ifdef USE_LIBSNARK
   using namespace libsnark;
#endif

ZComputationProver::
ZComputationProver(int ph, int b_size, int num_r, int size_input,
                   int size_output, int num_intermediate_variables, int num_constraints,
                   const char *name_prover, int size_aij, int size_bij, int size_cij,
                   const char *file_name_qap,
                   const char *file_name_f1_index) : ComputationProver(ph, b_size, NUM_REPS_PCP, size_input, name_prover) {
  this->size_input = size_input;
  this->size_output = size_output;
  size_constants = 0;

  chi = num_constraints;
  num_aij = size_aij;
  num_bij = size_bij;
  num_cij = size_cij;

  num_vars = num_intermediate_variables;
  n_prime = num_intermediate_variables;
  n = n_prime + size_input + size_output + size_constants;
  
  size_f1_vec = n_prime;
  size_f2_vec = chi+1;
 


  num_local_runs = NUM_LOCAL_RUNS;
  init_state(file_name_f1_index);
  init_qap(file_name_qap);

}

ZComputationProver::
ZComputationProver(int ph, int b_size, int num_r, int size_input,
                   int size_output, int num_intermediate_variables, int num_constraints,
                   const char *name_prover, int size_aij, int size_bij, int size_cij,
                   const char *file_name_qap,
                   const char *file_name_f1_index,
                   const char *_shared_bstore_file_name) : ComputationProver(ph, b_size, NUM_REPS_PCP, size_input, name_prover, _shared_bstore_file_name) {
  this->size_input = size_input;
  this->size_output = size_output;
  size_constants = 0;

  chi = num_constraints;
  num_aij = size_aij;
  num_bij = size_bij;
  num_cij = size_cij;

  num_vars = num_intermediate_variables;
  n_prime = num_intermediate_variables;
  n = n_prime + size_input + size_output + size_constants;

  size_f1_vec = n_prime;
  size_f2_vec = chi+1;

  num_local_runs = NUM_LOCAL_RUNS;
  init_state(file_name_f1_index);
  init_qap(file_name_qap);


}

void ZComputationProver::init_qap(const char *file_name_qap) {
  // log computation-specific information
  if (rank == MPI_COORD_RANK) {
    cout<<"num_constraints "<<chi<<endl;
    cout<<"num_vars "<<num_vars<<endl;
    cout<<"size_f1_vec "<<size_f1_vec<<endl;
    cout<<"size_f2_vec "<<size_f2_vec<<endl;
    cout<<"size_input "<<size_input<<endl;
    cout<<"size_output "<<size_output<<endl; 
  }


  // set the roots
#ifndef USE_LIBSNARK 
  qap_roots.SetLength(size_f2_vec);
  qap_roots2.SetLength(size_f2_vec-1);
  single_root.SetLength(1);
  set_v.SetLength(size_f2_vec);
  v_prime.SetLength(size_f2_vec);
  
  alloc_init_vec(&set_v_z, size_f2_vec);
  
  #if FAST_FOURIER_INTERPOLATION == 1
    mpz_init(omega);
    alloc_init_vec(&powers_of_omega, size_f2_vec);
    
    v->generate_root_of_unity(size_f2_vec, prime);
    v->get_root_of_unity(&omega);
    v->compute_set_v(size_f2_vec, set_v_z, omega, prime);

    mpz_init(omega_inv);
    mpz_invert(omega_inv, omega, prime);
    mpz_set_ui(powers_of_omega[0], 1);
    for (int i=1; i<size_f2_vec; i++) {
      mpz_mul(powers_of_omega[i], powers_of_omega[i-1], omega_inv);
      mpz_mod(powers_of_omega[i], powers_of_omega[i], prime);
    }
  #else
    v->compute_set_v(size_f2_vec, set_v_z, prime);
  #endif

  char str[BUFLEN];
  for (int i=0; i<size_f2_vec; i++) {
    mpz_get_str(str, 10, set_v_z[i]);
    conv(set_v[i], to_ZZ(str));
  }
  
  poly_tree = new ZZ_pX[2*size_f2_vec-1];
  interpolation_tree = new ZZ_pX[2*size_f2_vec-1];
  num_levels = ((log(size_f2_vec)/log(2)));
  
  z_poly_A_c.SetMaxLength(size_f2_vec);
  z_poly_B_c.SetMaxLength(size_f2_vec);
  z_poly_C_c.SetMaxLength(size_f2_vec);
  
  #if FAST_FOURIER_INTERPOLATION == 1
  ZZ_p omega_zz;
  mpz_get_str(str, 10, omega);
  conv(omega_zz, to_ZZ(str));
 
  qap_roots[0] = 1;
  for (int i=1; i<size_f2_vec; i++)
    qap_roots[i] = qap_roots[i-1] * omega_zz;

  for (int i=1; i<size_f2_vec; i++)
    qap_roots2[i-1] = qap_roots[i];
  #else
  for (int i=0; i<size_f2_vec; i++)
    qap_roots[i] = i;

  for (int i=1; i<size_f2_vec; i++)
    qap_roots2[i-1] = i;

  build_poly_tree(num_levels, 0, 0);

  z_poly_A_pv.SetLength(size_f2_vec);
  z_poly_B_pv.SetLength(size_f2_vec);
  z_poly_C_pv.SetLength(size_f2_vec);
  #endif

  
  // step 4 in H(t) business
  BuildFromRoots(z_poly_D_c, qap_roots2);

  // compute these based on values set by the compiler

  
  poly_A = (poly_compressed *) malloc(num_aij * sizeof(poly_compressed));
  poly_B = (poly_compressed *) malloc(num_bij * sizeof(poly_compressed));
  poly_C = (poly_compressed *) malloc(num_cij * sizeof(poly_compressed));

  for (int i=0; i<num_aij; i++)
    alloc_init_scalar(poly_A[i].coefficient);

  for (int i=0; i<num_bij; i++)
    alloc_init_scalar(poly_B[i].coefficient);

  for (int i=0; i<num_cij; i++)
    alloc_init_scalar(poly_C[i].coefficient);

  alloc_init_vec(&poly_A_pv, size_f2_vec);
  alloc_init_vec(&poly_B_pv, size_f2_vec);
  alloc_init_vec(&poly_C_pv, size_f2_vec);

  for (int i=0; i<size_f2_vec; i++) {
    mpz_set_ui(poly_A_pv[i], 0);
    mpz_set_ui(poly_B_pv[i], 0);
    mpz_set_ui(poly_C_pv[i], 0);
  }

  // create vectors to store the evaluations of the polynomial at r_star
  
  
  alloc_init_vec(&eval_poly_A, n+1);
  alloc_init_vec(&eval_poly_B, n+1);
  alloc_init_vec(&eval_poly_C, n+1);

  // the polynomials in the compressed form to be initialized by the COMPILER
  // open the file
  FILE *fp = fopen(file_name_qap, "r");
  if (fp == NULL) {
    cout<<"LOG: Cannot read "<<file_name_qap<<endl;
    exit(1);
  }

  char line[BUFLEN];
  //mpz_t temp;
  //alloc_init_scalar(temp);
  // Measurement m_qap;
  // m_qap.begin_with_init();
  // fill the array of struct: poly_A, poly_B, and poly_C
  int line_num = 0;
  
  while (fgets(line, sizeof line, fp) != NULL) {
    if (line[0] == '\n')
      continue;
    if (line_num < num_aij) {
      gmp_sscanf(line, "%d %d %Zd", &poly_A[line_num].i, &poly_A[line_num].j, poly_A[line_num].coefficient);
    } else if (line_num >= num_aij && line_num < num_aij+num_bij) {
      gmp_sscanf(line, "%d %d %Zd", &poly_B[line_num-num_aij].i, &poly_B[line_num-num_aij].j, poly_B[line_num-num_aij].coefficient);
    } else {
      gmp_sscanf(line, "%d %d %Zd", &poly_C[line_num-num_aij-num_bij].i, &poly_C[line_num-num_aij-num_bij].j, poly_C[line_num-num_aij-num_bij].coefficient);
    }
    line_num++;
  }
  
  fclose(fp);

#endif //USE_LIBSNARK
  //  m_qap.end();
  // cout << "prover time to load qap" << m_qap.get_papi_elapsed_time() << endl;
}

void ZComputationProver::init_state(const char *file_name_f1_index) {
#if NONINTERACTIVE == 1
  num_bits_in_prime = 256;
#else
  string str(file_name_f1_index);
  if (str.find("bisect_sfdl") != std::string::npos) {
    num_bits_in_prime = 220;
  } else if (str.find("pd2_sfdl") != std::string::npos) {
    num_bits_in_prime = 220;
  } else {
    num_bits_in_prime = 128;
  }
  if (rank == MPI_COORD_RANK)
    cout<<"LOG: Using a prime of size "<<num_bits_in_prime<<endl;
#endif

  temp_stack_size = 16;

  crypto_in_use = CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;

  Prover::init_state();

  // sets prime modulus for all NTL's operations
  char prime_str[BUFLEN];
  mpz_get_str(prime_str, 10, prime);
  ZZ z_prime = to_ZZ(prime_str);
  ZZ_p::init(z_prime);

  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;

  alloc_init_vec(&F1, size_f1_vec);
  alloc_init_vec(&F1_q, size_f1_vec);
  alloc_init_vec(&F2, size_f2_vec);

  F1_index = new uint32_t[size_f1_vec];
  load_vector(size_f1_vec, F1_index, file_name_f1_index);

  alloc_init_vec(&input_output_q, size_input+size_output);
  input_q = &input_output_q[0];
  output_q = &input_output_q[size_input];

  alloc_init_vec(&input_output, size_input+size_output);
  input = &input_output[0];
  output = &input_output[size_input];

  alloc_init_vec(&temp_qs, temp_stack_size);

  alloc_init_scalar(temp);
  alloc_init_scalar(temp2);
  alloc_init_scalar(temp_q);
  alloc_init_scalar(temp_q2);

#if NONINTERACTIVE == 1
#if GGPR == 1
  size_answer_G1 = 9;
  size_answer_G2 = 1;

  // init pairing
  init_pairing_from_file(PAIRING_PARAM, prime);

  // init storage for queries
  size_f1_g_Ai_query = size_f1_g_alpha_Ai_query = size_f1_vec;
  size_f1_g_Bi_query = size_f1_h_Bi_query = size_f1_g_Ci_query = size_f1_vec+size_input+size_output;
  size_f1_g_alpha_Bi_query = size_f1_g_alpha_Ci_query = size_f1_vec+size_input+size_output;
  size_f1_g_beta_a_Ai_query = size_f1_g_beta_b_Bi_query = size_f1_g_beta_c_Ci_query = size_f1_vec+size_input+size_output;
  size_f2_g_t_i_query = size_f2_g_alpha_t_i_query = size_f2_vec;

  alloc_init_vec_G1(&f1_g_Ai_query, size_f1_g_Ai_query);
  alloc_init_vec_G1(&f1_g_Bi_query, size_f1_g_Bi_query);
  alloc_init_vec_G2(&f1_h_Bi_query, size_f1_h_Bi_query);
  alloc_init_vec_G1(&f1_g_Ci_query, size_f1_g_Ci_query);
  alloc_init_vec_G1(&f2_g_t_i_query, size_f2_g_t_i_query);

  alloc_init_vec_G1(&f1_g_alpha_Ai_query, size_f1_g_alpha_Ai_query);
  alloc_init_vec_G1(&f1_g_alpha_Bi_query, size_f1_g_alpha_Bi_query);
  alloc_init_vec_G1(&f1_g_alpha_Ci_query, size_f1_g_alpha_Ci_query);
  alloc_init_vec_G1(&f2_g_alpha_t_i_query, size_f2_g_alpha_t_i_query);

  alloc_init_vec_G1(&f1_g_beta_a_Ai_query, size_f1_g_beta_a_Ai_query);
  alloc_init_vec_G1(&f1_g_beta_b_Bi_query, size_f1_g_beta_b_Bi_query);
  alloc_init_vec_G1(&f1_g_beta_c_Ci_query, size_f1_g_beta_c_Ci_query);

#else
  size_answer_G1 = 7;
  size_answer_G2 = 1;

  // init pairing
  init_pairing_from_file(PAIRING_PARAM, prime);

  // init storage for queries
  size_f1_g_a_alpha_a_Ai_query = size_f1_g_a_Ai_query = size_f1_vec;
  size_f1_g_b_Bi_query = size_f1_h_b_Bi_query = size_f1_g_c_Ci_query = size_f1_vec+size_input+size_output;
  size_f1_g_b_alpha_b_Bi_query = size_f1_g_c_alpha_c_Ci_query = size_f1_vec+size_input+size_output;
  //size_f2_h_t_i_query = size_f2_vec;
  size_f2_g_t_i_query = size_f2_vec;
  size_f1_beta_query = size_f1_vec+size_input+size_output;

#if PROTOCOL == PINOCCHIO_ZK
  alloc_init_scalar_G1(g_a_D);
  alloc_init_scalar_G1(g_b_D);
  alloc_init_scalar_G1(g_c_D);

  alloc_init_scalar_G2(h_b_D);

  alloc_init_scalar_G1(g_a_alpha_a_D);
  alloc_init_scalar_G1(g_b_alpha_b_D);
  alloc_init_scalar_G1(g_c_alpha_c_D);
  alloc_init_scalar_G1(g_a_beta_D);
  alloc_init_scalar_G1(g_b_beta_D);
  alloc_init_scalar_G1(g_c_beta_D);

  alloc_init_vec(&delta_abc, 3);
#endif

#endif
  // init storage for answers
  alloc_init_vec_G1(&f_ni_answers_G1, size_answer_G1);
  alloc_init_vec_G2(&f_ni_answers_G2, size_answer_G2);
#else
  alloc_init_vec(&f1_commitment, expansion_factor*size_f1_vec);
  alloc_init_vec(&f1_consistency, expansion_factor*size_f1_vec);
  alloc_init_vec(&f1_q1, size_f1_vec);
  alloc_init_vec(&f1_q2, size_f1_vec);
  alloc_init_vec(&f1_q3, size_f1_vec);
  alloc_init_vec(&f1_q4, size_f1_vec);

  alloc_init_vec(&f2_commitment, expansion_factor*size_f2_vec);
  alloc_init_vec(&f2_consistency, expansion_factor*size_f2_vec);
  alloc_init_vec(&f2_q1, size_f2_vec);
  alloc_init_vec(&f2_q2, size_f2_vec);
  alloc_init_vec(&f2_q3, size_f2_vec);
  alloc_init_vec(&f2_q4, size_f2_vec);

  alloc_init_vec(&f_answers, NUM_REPS_PCP * NUM_LIN_PCP_QUERIES);
  alloc_init_scalar(answer);

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
#endif
}

ZComputationProver::~ZComputationProver() {
#if FAST_FOURIER_INTERPOLATION == 1
  clear_scalar(omega);
  clear_scalar(omega_inv);
  clear_vec(size_f2_vec, powers_of_omega);
#endif
  delete[] F1_index;

  clear_vec(size_f1_vec, F1);
  clear_vec(size_f1_vec, F1_q);

  clear_vec(size_f2_vec, F2);
#ifndef USE_LIBSNARK
  clear_vec(size_f2_vec, set_v_z);
#endif
  clear_vec(size_input + size_output, input_output);
  clear_vec(size_input + size_output, input_output_q);

  clear_scalar(temp);
  clear_scalar(temp2);
  clear_scalar(temp_q);
  clear_scalar(temp_q2);

  clear_vec(temp_stack_size, temp_qs);

#if NONINTERACTIVE == 1
  // clear noninteractive related storage.
#if GGPR == 1
  clear_vec_G1(size_f1_g_Ai_query, f1_g_Ai_query);
  clear_vec_G1(size_f1_g_Bi_query, f1_g_Bi_query);
  clear_vec_G2(size_f1_h_Bi_query, f1_h_Bi_query);
  clear_vec_G1(size_f1_g_Ci_query, f1_g_Ci_query);
  clear_vec_G1(size_f2_g_t_i_query, f2_g_t_i_query);

  clear_vec_G1(size_f1_g_alpha_Ai_query, f1_g_alpha_Ai_query);
  clear_vec_G1(size_f1_g_alpha_Bi_query, f1_g_alpha_Bi_query);
  clear_vec_G1(size_f1_g_alpha_Ci_query, f1_g_alpha_Ci_query);
  clear_vec_G1(size_f2_g_alpha_t_i_query, f2_g_alpha_t_i_query);

  clear_vec_G1(size_f1_g_beta_a_Ai_query, f1_g_beta_a_Ai_query);
  clear_vec_G1(size_f1_g_beta_b_Bi_query, f1_g_beta_b_Bi_query);
  clear_vec_G1(size_f1_g_beta_c_Ci_query, f1_g_beta_c_Ci_query);

#else

#if PROTOCOL == PINOCCHIO_ZK
  clear_scalar_G1(g_a_D);
  clear_scalar_G1(g_b_D);
  clear_scalar_G1(g_c_D);

  clear_scalar_G2(h_b_D);
  
  clear_scalar_G1(g_a_alpha_a_D);
  clear_scalar_G1(g_b_alpha_b_D);
  clear_scalar_G1(g_c_alpha_c_D);
  clear_scalar_G1(g_a_beta_D);
  clear_scalar_G1(g_b_beta_D);
  clear_scalar_G1(g_c_beta_D);

  clear_vec(3, delta_abc);
#endif

#endif
  clear_vec_G1(size_answer_G1, f_ni_answers_G1);
  clear_vec_G2(size_answer_G2, f_ni_answers_G2);
#else
  //clear_vec(expansion_factor*size_f1_vec, f1_commitment);
  clear_vec(expansion_factor*size_f1_vec, f1_consistency);
  clear_vec(size_f1_vec, f1_q1);
  clear_vec(size_f1_vec, f1_q2);
  clear_vec(size_f1_vec, f1_q3);
  clear_vec(size_f1_vec, f1_q4);

  //clear_vec(expansion_factor*size_f2_vec, f2_commitment);
  clear_vec(expansion_factor*size_f2_vec, f2_consistency);
  clear_vec(size_f2_vec, f2_q1);
  clear_vec(size_f2_vec, f2_q2);
  clear_vec(size_f2_vec, f2_q3);
  clear_vec(size_f2_vec, f2_q4);

  clear_vec(NUM_REPS_PCP * NUM_LIN_PCP_QUERIES, f_answers);
  clear_scalar(answer);
#endif
}

void ZComputationProver::
find_cur_qlengths() {
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

  qquery_sizes.push_back(size_f1_vec);
  qquery_sizes.push_back(size_f1_vec);
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

  qquery_f_ptrs.push_back(f1_q1);
  qquery_f_ptrs.push_back(f1_q1);
  qquery_f_ptrs.push_back(f1_q1);
  qquery_f_ptrs.push_back(f2_q1);

  qquery_F_ptrs.clear();
  for (int i=0; i<NUM_REPS_LIN; i++) {
    qquery_F_ptrs.push_back(F1);
    qquery_F_ptrs.push_back(F1);
    
    //setting it to NULL so that the prover will add up answers to
    //previous two queries as answer to the third linearity query. see
    //libv/prover.cpp: prover_answer_queries() function for more details
    //qquery_F_ptrs.push_back(F1);
    qquery_F_ptrs.push_back(NULL);
    
    qquery_F_ptrs.push_back(F2);
    qquery_F_ptrs.push_back(F2);
    //qquery_F_ptrs.push_back(F2);
    qquery_F_ptrs.push_back(NULL);
  }

  qquery_F_ptrs.push_back(F1);
  qquery_F_ptrs.push_back(F1);
  qquery_F_ptrs.push_back(F1);
  qquery_F_ptrs.push_back(F2);

  qquery_q_ptrs.clear();
  for (int i=0; i<NUM_REPS_PCP*NUM_LIN_PCP_QUERIES; i++)
    qquery_q_ptrs.push_back(i);
}

void ZComputationProver::compute_assignment_vectors() {
  // code to compute H(t)
  mpz_t temp;
  alloc_init_scalar(temp);

  // 1. Figure out A_w(t), B_w(t), and C_w(t) at the roots selected for the QAP

  for (int i=0; i<size_f2_vec; i++) {
    mpz_set_ui(poly_A_pv[i], 0);
    mpz_set_ui(poly_B_pv[i], 0);
    mpz_set_ui(poly_C_pv[i], 0);
  }

  int index;
  for (int i=0; i<num_aij; i++) {
    index = poly_A[i].i;
    if (index == 0)
      mpz_set(temp, poly_A[i].coefficient);
    else if (index <= size_f1_vec)
      mpz_mul(temp, poly_A[i].coefficient, F1[index - 1]);
    else if (index > size_f1_vec && index <= size_f1_vec + size_input)
      mpz_mul(temp, poly_A[i].coefficient, input[index - 1 - size_f1_vec]);
    else
      mpz_mul(temp, poly_A[i].coefficient, output[index - 1 - size_input - size_f1_vec]);
    mpz_add(poly_A_pv[poly_A[i].j], poly_A_pv[poly_A[i].j], temp);
    mpz_mod(poly_A_pv[poly_A[i].j], poly_A_pv[poly_A[i].j], prime);
  }

  for (int i=0; i<num_bij; i++) {
    index = poly_B[i].i;
    if (index == 0)
      mpz_set(temp, poly_B[i].coefficient);
    else if (index <= size_f1_vec)
      mpz_mul(temp, poly_B[i].coefficient, F1[index-1]);
    else if (index > size_f1_vec && index <= size_f1_vec + size_input)
      mpz_mul(temp, poly_B[i].coefficient, input[index - 1 - size_f1_vec]);
    else
      mpz_mul(temp, poly_B[i].coefficient, output[index - 1 - size_input - size_f1_vec]);

    mpz_add(poly_B_pv[poly_B[i].j], poly_B_pv[poly_B[i].j], temp);
    mpz_mod(poly_B_pv[poly_B[i].j], poly_B_pv[poly_B[i].j], prime);
  }

  for (int i=0; i<num_cij; i++) {
    index = poly_C[i].i;
    if (index == 0)
      mpz_set(temp, poly_C[i].coefficient);
    else if (index <= size_f1_vec)
      mpz_mul(temp, poly_C[i].coefficient, F1[index-1]);
    else if (index > size_f1_vec && index <= size_f1_vec + size_input)
      mpz_mul(temp, poly_C[i].coefficient, input[index - 1 - size_f1_vec]);
    else
      mpz_mul(temp, poly_C[i].coefficient, output[index - 1 - size_input - size_f1_vec]);

    mpz_add(poly_C_pv[poly_C[i].j], poly_C_pv[poly_C[i].j], temp);
    mpz_mod(poly_C_pv[poly_C[i].j], poly_C_pv[poly_C[i].j], prime);
  }

  // 2. Interpolate them to get them in the coefficients form
  char z_str[BUFLEN];
  ZZ x;
  ZZ_p x_p;
  #if FAST_FOURIER_INTERPOLATION == 1
    mpz_t *poly_A_c = zcomp_fast_interpolate(size_f2_vec, poly_A_pv, omega_inv, prime);
    mpz_set_ui(temp, size_f2_vec);
    mpz_invert(temp, temp, prime);
    for (int i=0; i<size_f2_vec; i++) {
      mpz_mul(poly_A_c[i], poly_A_c[i], temp);
      mpz_mod(poly_A_c[i], poly_A_c[i], prime);
      mpz_get_str(z_str, 10, poly_A_c[i]);
      x = to_ZZ(z_str);
      conv(x_p, x);
      SetCoeff(z_poly_A_c, i, x_p);
    }
    //clear_vec(size_f2_vec, poly_A_c);
    
    mpz_t *poly_B_c = zcomp_fast_interpolate(size_f2_vec, poly_B_pv, omega_inv, prime);
    for (int i=0; i<size_f2_vec; i++) {
      mpz_mul(poly_B_c[i], poly_B_c[i], temp);
      mpz_mod(poly_B_c[i], poly_B_c[i], prime);
      
      mpz_get_str(z_str, 10, poly_B_c[i]);
      x = to_ZZ(z_str);
      conv(x_p, x);
      SetCoeff(z_poly_B_c, i, x_p);
    }
    //clear_vec(size_f2_vec, poly_B_c);
    
    mpz_t *poly_C_c = zcomp_fast_interpolate(size_f2_vec, poly_C_pv, omega_inv, prime);
    for (int i=0; i<size_f2_vec; i++) {
      mpz_mul(poly_C_c[i], poly_C_c[i], temp);
      mpz_mod(poly_C_c[i], poly_C_c[i], prime);
      
      mpz_get_str(z_str, 10, poly_C_c[i]);
      x = to_ZZ(z_str);
      conv(x_p, x);
      SetCoeff(z_poly_C_c, i, x_p);
    }
    //clear_vec(size_f2_vec, poly_C_c);
  #else
    for (int i=0; i<size_f2_vec; i++) {
      mpz_get_str(z_str, 10, poly_A_pv[i]);
      x = to_ZZ(z_str);
      conv(z_poly_A_pv[i], x);

      mpz_get_str(z_str, 10, poly_B_pv[i]);
      x = to_ZZ(z_str);
      conv(z_poly_B_pv[i], x);

      mpz_get_str(z_str, 10, poly_C_pv[i]);
      x = to_ZZ(z_str);
      conv(z_poly_C_pv[i], x);
    }

    // 3. Find P_w(t) = A_w(t) * B_w(t) - C_w(t)
    // old NTL interpolation
    //interpolate(z_poly_A_c, qap_roots, z_poly_A_pv);
    //interpolate(z_poly_B_c, qap_roots, z_poly_B_pv);
    //interpolate(z_poly_C_c, qap_roots, z_poly_C_pv);

    zcomp_interpolate(num_levels, 0, 0, &z_poly_A_pv);
    z_poly_A_c = interpolation_tree[0];
  
    zcomp_interpolate(num_levels, 0, 0, &z_poly_B_pv);
    z_poly_B_c = interpolation_tree[0];

    zcomp_interpolate(num_levels, 0, 0, &z_poly_C_pv);
    z_poly_C_c = interpolation_tree[0];
  #endif

#if PROTOCOL == PINOCCHIO_ZK
  //Add delta_a * D(t) to poly_A, delta_b * D(t) to poly_B, etc.
  ZZ_pX scaledD;
  ZZ z_delta_scalar_;
  ZZ_p z_delta_scalar;

  mpz_get_str(z_str, 10, delta_abc[0]);
  z_delta_scalar_ = to_ZZ(z_str);
  conv(z_delta_scalar, z_delta_scalar_);

  mul(scaledD, z_poly_D_c, z_delta_scalar);
  add(z_poly_A_c, z_poly_A_c, scaledD);

  mpz_get_str(z_str, 10, delta_abc[1]);
  z_delta_scalar_ = to_ZZ(z_str);
  conv(z_delta_scalar, z_delta_scalar_);

  mul(scaledD, z_poly_D_c, z_delta_scalar);
  add(z_poly_B_c, z_poly_B_c, scaledD);

  mpz_get_str(z_str, 10, delta_abc[2]);
  z_delta_scalar_ = to_ZZ(z_str);
  conv(z_delta_scalar, z_delta_scalar_);

  mul(scaledD, z_poly_D_c, z_delta_scalar);
  add(z_poly_C_c, z_poly_C_c, scaledD);
#endif

  mul(z_poly_P_c2, z_poly_A_c, z_poly_B_c);
  add(z_poly_P_c, z_poly_P_c2, z_poly_C_c);

  // 4. Compute D(t)
  // already done

  // 5. Find H_w(t) = P_w(t)/D(t)
  int out = divide(z_poly_H_c, z_poly_P_c, z_poly_D_c);
  
  // 6. Set the coefficients of H_w(t) as F2
  // degree from lower to higher.
  for (int i=0; i<size_f2_vec; i++) {
    x_p = coeff(z_poly_H_c, i);
    stringstream coefficient;
    coefficient<<x_p;
    mpz_set_str(F2[i], (coefficient.str()).c_str(), 10);
    coefficient.clear();
  }
}

void ZComputationProver::prover_do_computation() {
  bool passed_test = true;

  for (int i=batch_start; i<=batch_end; i++) {
    // baseline minimal
    if (exogenous_checker->baseline_minimal_input_size != 0) {
      void* native_input =
        malloc(exogenous_checker->baseline_minimal_input_size);
      void* native_output =
        malloc(exogenous_checker->baseline_minimal_output_size);

      for (uint32_t j = 0; j < exogenous_checker->baseline_minimal_input_size; j++)
        ((char*)native_input)[j] = rand();
      
      snprintf(scratch_str, BUFLEN-1, "input_minimal_b_%d", i);
      dump_vector(exogenous_checker->baseline_minimal_input_size, (char *)native_input, scratch_str, FOLDER_WWW_DOWNLOAD);

      if (i == batch_start)
        m_computation_minimal.begin_with_init();
      else
        m_computation_minimal.begin_with_history();

      for (int g=0; g<num_local_runs; g++) {
        snprintf(scratch_str, BUFLEN-1, "input_minimal_b_%d", i);
        load_vector(exogenous_checker->baseline_minimal_input_size, (char *)native_input, scratch_str, FOLDER_WWW_DOWNLOAD);
        exogenous_checker->baseline_minimal(native_input, native_output);
        snprintf(scratch_str, BUFLEN-1, "output_minimal_b_%d", i);
        dump_vector(exogenous_checker->baseline_minimal_output_size, (char *)native_output, scratch_str, FOLDER_WWW_DOWNLOAD);
      }

      m_computation_minimal.end();
      free(native_input);
      free(native_output);
    }

    // baseline minimal + SHA256
    if (exogenous_checker->baseline_minimal_input_size != 0) {
      void* native_input =
        malloc(exogenous_checker->baseline_minimal_input_size);
      void* native_output =
        malloc(exogenous_checker->baseline_minimal_output_size);
         
      for (uint32_t j = 0; j < exogenous_checker->baseline_minimal_input_size; j++)
          ((char*)native_input)[j] = rand();
      
      snprintf(scratch_str, BUFLEN-1, "input_sha_b_%d", i);
      dump_vector(exogenous_checker->baseline_minimal_input_size, (char *)native_input, scratch_str, FOLDER_WWW_DOWNLOAD);

      if (i == batch_start)
        m_computation_sha.begin_with_init();
      else
        m_computation_sha.begin_with_history();

      for (int g=0; g<num_local_runs; g++) {
        
        // compute a SHA-256 hash of the input
        snprintf(scratch_str, BUFLEN-1, "input_sha_b_%d", i);
        load_vector(exogenous_checker->baseline_minimal_input_size, (char *)native_input, scratch_str, FOLDER_WWW_DOWNLOAD);

        unsigned char hash[32];
        sha256(exogenous_checker->baseline_minimal_input_size, (unsigned char *) native_input, hash);
        
        exogenous_checker->baseline_minimal(native_input, native_output);
        
        // compute a SHA-256 hash of the output
        sha256(exogenous_checker->baseline_minimal_output_size, (unsigned char *) native_output, hash);

        snprintf(scratch_str, BUFLEN-1, "output_sha_b_%d", i);
        dump_vector(exogenous_checker->baseline_minimal_output_size, (char *)native_output, scratch_str, FOLDER_WWW_DOWNLOAD);
      }

      m_computation_sha.end();
      free(native_input);
      free(native_output);
    }

    // baseline minimal + Ajtai
    if (i == batch_start)
      m_computation.begin_with_init();
    else
      m_computation.begin_with_history();

    //The call to init_exo_inputs is NECESSARY to move exogenously created inputs from
    //the verifier's custom input generator to the KyotoCabinet BlockStore for MapReduce computations.
    cout << "LOG: Running baseline" << endl;

    for (int g=0; g<num_local_runs; g++) {
      snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
      load_vector(size_input, input, scratch_str, FOLDER_WWW_DOWNLOAD);

      snprintf(scratch_str, BUFLEN-1, "input_q_b_%d", i);
      load_vector(size_input, input_q, scratch_str, FOLDER_WWW_DOWNLOAD);
      exogenous_checker->init_exo_inputs(input_q, size_input, bstore_file_path, _blockStore);

      exogenous_checker->baseline(input_q, size_input, output_q, size_output);

      snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
      dump_vector(size_output, output_q, scratch_str, FOLDER_WWW_DOWNLOAD);
    }
    m_computation.end();

    if (i==batch_start)
      cout << "LOG: Prover is computing Y, Z from X" << endl;

    //Run Zaatar implementation
    if (size_f1_vec < 10000) {
      num_interpret_runs = num_local_runs;
    } else {
      num_interpret_runs = 1;
    }

    if (i == batch_start)
      m_interpret_cons.begin_with_init();
    else
      m_interpret_cons.begin_with_history();

    for (int g = 0; g < num_interpret_runs; g++) {
      snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
      load_vector(size_input, input, scratch_str, FOLDER_WWW_DOWNLOAD);

      snprintf(scratch_str, BUFLEN-1, "input_q_b_%d", i);
      load_vector(size_input, input_q, scratch_str, FOLDER_WWW_DOWNLOAD);

      interpret_constraints();

      snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
      dump_vector(size_output, output, scratch_str, FOLDER_WWW_DOWNLOAD);

      snprintf(scratch_str, BUFLEN-1, "output_q_b_%d", i);
      dump_vector(size_output, output_q, scratch_str, FOLDER_WWW_DOWNLOAD);

      // export the outputs to different block stores
      exogenous_checker->export_exo_inputs(output_q, size_output, bstore_file_path, _blockStore);

      // shuffle the explicit inputs, in case of MapRed
      //exogenous_checker->run_shuffle_phase(i, FOLDER_STATE);
    }
#ifdef USE_LIBSNARK
     ofstream outfile1("./bin/inputs");
  outfile1 << size_input << endl;
  //cout << size_input<< endl;
  for (int i = 0; i < size_input; i++)
    outfile1 << input[i] << endl;
  outfile1.close();
 
  ofstream outfile2("./bin/outputs");
  //cout << size_output<< endl; 
  outfile2 << size_output << endl;
  for (int i = 0; i < size_output; i++)
    outfile2 << output[i] << endl;
  outfile2.close();
 #endif

    m_interpret_cons.end();
    
    if (i==batch_start)
      cout << "LOG: Running exogenous_check" << endl;

    passed_test &= exogenous_checker->exogenous_check(input, input_q, size_input, output, output_q, size_output, prime);
#ifndef USE_LIBSNARK
    if (i==batch_start)
      cout << "LOG: Prover is filling in the proof vector" << endl;

    if (i == batch_start)
      m_proofv_creation.begin_with_init();
    else
      m_proofv_creation.begin_with_history();

#if PROTOCOL == PINOCCHIO_ZK
    //Choose random delta_a, delta_b, and delta_c for this computation
    //instance
    v->get_random_priv(delta_abc[0], prime);
    v->get_random_priv(delta_abc[1], prime);
    v->get_random_priv(delta_abc[2], prime); 
#endif

#ifndef DEBUG_MALICIOUS_PROVER
      compute_assignment_vectors();
#endif
      
      
  
    m_proofv_creation.end();

#if PROTOCOL == PINOCCHIO_ZK
    //Write delta_a, delta_b, and delta_c to a file
    snprintf(scratch_str, BUFLEN-1, "delta_abc_b_%d", i);
    dump_vector(3, delta_abc, scratch_str, FOLDER_WWW_DOWNLOAD);
#endif

    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    dump_vector(size_f1_vec, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", i);
    dump_vector(size_f2_vec, F2, scratch_str, FOLDER_WWW_DOWNLOAD);

#endif
  }

  if (passed_test)
    cout << endl << "LOG: Output computation passes the exogenous check." << endl << endl;
  else {
    cout << endl << "LOG: Output computation failed the exogenous check and I am killing the prover." << endl << endl;
    exit(1);
  }
#ifndef USE_LIBSNARK
   delete[] poly_tree;
   delete[] interpolation_tree;
#endif
  z_poly_A_pv.SetLength(0);
  z_poly_B_pv.SetLength(0);
  z_poly_C_pv.SetLength(0);

}

//PROVER's CODE
void ZComputationProver::prover_computation_commitment() {

  Measurement mt;
  mt.begin_with_init();
  init_block_store();
 mt.end();
 cout << "init_block_store: " << mt.get_papi_elapsed_time() << endl;
  // execute the computation


  
  mt.begin_with_init();
  prover_do_computation();
 mt.end();
 cout << "p_do_comp: " << mt.get_papi_elapsed_time() << endl;
 
  
#if NONINTERACTIVE == 1
  prover_noninteractive();
 #else
  prover_interactive();
#endif

 cout << "p_current_mem_after_ls_prover " << getCurrentRSS() << endl;
 cout << "p_peak_mem_after_ls_prover " << getPeakRSS() << endl;


}

#if NONINTERACTIVE == 1
#if GGPR == 1
void ZComputationProver::prover_noninteractive_GGPR() {
  G1_t g1_tmp1;

  alloc_init_scalar_G1(g1_tmp1);

  // these vectors are already initialized.
  load_vector_G1(size_f1_g_Ai_query, f1_g_Ai_query, (char *)"f1_g_Ai_query", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(size_f1_g_Bi_query, f1_g_Bi_query, (char *)"f1_g_Bi_query", FOLDER_WWW_DOWNLOAD);
  load_vector_G2(size_f1_h_Bi_query, f1_h_Bi_query, (char *)"f1_h_Bi_query", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(size_f1_g_Ci_query, f1_g_Ci_query, (char *)"f1_g_Ci_query", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(size_f2_g_t_i_query, f2_g_t_i_query, (char *)"f2_g_t_i_query", FOLDER_WWW_DOWNLOAD);

  load_vector_G1(size_f1_g_alpha_Ai_query, f1_g_alpha_Ai_query, (char *)"f1_g_alpha_Ai_query", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(size_f1_g_alpha_Bi_query, f1_g_alpha_Bi_query, (char *)"f1_g_alpha_Bi_query", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(size_f1_g_alpha_Ci_query, f1_g_alpha_Ci_query, (char *)"f1_g_alpha_Ci_query", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(size_f2_g_alpha_t_i_query, f2_g_alpha_t_i_query, (char *)"f2_g_alpha_t_i_query", FOLDER_WWW_DOWNLOAD);

  load_vector_G1(size_f1_g_beta_a_Ai_query, f1_g_beta_a_Ai_query, (char *)"f1_g_beta_a_Ai_query", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(size_f1_g_beta_b_Bi_query, f1_g_beta_b_Bi_query, (char *)"f1_g_beta_b_Bi_query", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(size_f1_g_beta_c_Ci_query, f1_g_beta_c_Ci_query, (char *)"f1_g_beta_c_Ci_query", FOLDER_WWW_DOWNLOAD);

  for (int i=batch_start; i<=batch_end; i++) {
    if (i == 0)
      m_answer_queries.begin_with_init();
    else
      m_answer_queries.begin_with_history();

    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    load_vector(size_f1_vec, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", i);
    load_vector(size_f2_vec, F2, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
    load_vector(size_input, input, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
    load_vector(size_output, output, scratch_str, FOLDER_WWW_DOWNLOAD);

    // right now it is assumed that a symmetric pairing is used.
    // use G1 and G2 accordingly.
    multi_exponentiation_G1(size_f1_g_Ai_query, f1_g_Ai_query, F1, f_ni_answers_G1[0]);

    tri_multi_exponentiation_G1(f1_g_Bi_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[1]);

    tri_multi_exponentiation_G1(f1_g_Ci_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[2]);

    multi_exponentiation_G1(size_f2_g_t_i_query, f2_g_t_i_query, F2, f_ni_answers_G1[3]);

    multi_exponentiation_G1(size_f1_g_alpha_Ai_query, f1_g_alpha_Ai_query, F1, f_ni_answers_G1[4]);

    tri_multi_exponentiation_G1(f1_g_alpha_Bi_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[5]);

    tri_multi_exponentiation_G1(f1_g_alpha_Ci_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[6]);

    multi_exponentiation_G1(size_f2_g_alpha_t_i_query, f2_g_alpha_t_i_query, F2, f_ni_answers_G1[7]);

    tri_multi_exponentiation_G1(f1_g_beta_a_Ai_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[8]);
    tri_multi_exponentiation_G1(f1_g_beta_b_Bi_query, size_f1_vec, F1, size_input, input, size_output, output, g1_tmp1);
    // TODO change
    G1_mul(f_ni_answers_G1[8], f_ni_answers_G1[8], g1_tmp1);
    tri_multi_exponentiation_G1(f1_g_beta_c_Ci_query, size_f1_vec, F1, size_input, input, size_output, output, g1_tmp1);
    G1_mul(f_ni_answers_G1[8], f_ni_answers_G1[8], g1_tmp1);

    // answer queries for f1_h_b_Bi_query
    tri_multi_exponentiation_G2(f1_h_Bi_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G2[0]);

    // fold these into a single file f_answers_b_%d
    snprintf(scratch_str, BUFLEN-1, "f_answers_G1_b_%d", i);
    dump_vector_G1(size_answer_G1, f_ni_answers_G1, scratch_str, FOLDER_WWW_DOWNLOAD);
    snprintf(scratch_str, BUFLEN-1, "f_answers_G2_b_%d", i);
    dump_vector_G2(size_answer_G2, f_ni_answers_G2, scratch_str, FOLDER_WWW_DOWNLOAD);

    m_answer_queries.end();
  }

  clear_scalar_G1(g1_tmp1);
}
#endif

void ZComputationProver::prover_noninteractive() {
#if GGPR == 1
  prover_noninteractive_GGPR();
#else
#ifndef USE_LIBSNARK
#if PROTOCOL == PINOCCHIO_ZK  
  load_vector_G1(1, &g_a_D, (char *)"g_a_D", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(1, &g_b_D, (char *)"g_b_D", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(1, &g_c_D, (char *)"g_c_D", FOLDER_WWW_DOWNLOAD); 
  load_vector_G2(1, &h_b_D, (char *)"h_b_D", FOLDER_WWW_DOWNLOAD);

  //Rest are mentioned in pinocchio
  load_vector_G1(1, &g_a_alpha_a_D, (char *)"g_a_alpha_a_D", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(1, &g_b_alpha_b_D, (char *)"g_b_alpha_b_D", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(1, &g_c_alpha_c_D, (char *)"g_c_alpha_c_D", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(1, &g_a_beta_D, (char *)"g_a_beta_D", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(1, &g_b_beta_D, (char *)"g_b_beta_D", FOLDER_WWW_DOWNLOAD);
  load_vector_G1(1, &g_c_beta_D, (char *)"g_c_beta_D", FOLDER_WWW_DOWNLOAD);
#endif
#endif

  for (int i=batch_start; i<=batch_end; i++) {
    if (i == 0)
      m_answer_queries.begin_with_init();
    else
      m_answer_queries.begin_with_history();
#ifndef USE_LIBSNARK
    snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", i);
    load_vector(size_f1_vec, F1, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", i);
    load_vector(size_f2_vec, F2, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "input_b_%d", i);
    load_vector(size_input, input, scratch_str, FOLDER_WWW_DOWNLOAD);

    snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
    load_vector(size_output, output, scratch_str, FOLDER_WWW_DOWNLOAD);

#if PROTOCOL == PINOCCHIO_ZK
    snprintf(scratch_str, BUFLEN-1, "delta_abc_b_%d", i);
    load_vector(3, delta_abc, scratch_str, FOLDER_WWW_DOWNLOAD);
#endif

    // right now it is assumed that a symmetric pairing is used.
    // use G1 and G2 accordingly.
    alloc_init_vec_G1(&f1_g_a_Ai_query, size_f1_g_a_Ai_query);
    load_vector_G1(size_f1_g_a_Ai_query, f1_g_a_Ai_query, (char *)"f1_g_a_Ai_query", FOLDER_WWW_DOWNLOAD);
    multi_exponentiation_G1(size_f1_g_a_Ai_query, f1_g_a_Ai_query, F1, f_ni_answers_G1[0]);
    clear_vec_G1(size_f1_g_a_Ai_query, f1_g_a_Ai_query);

//    tri_multi_exponentiation_G1(f1_g_b_Bi_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[1]);

    alloc_init_vec_G1(&f1_g_c_Ci_query, size_f1_g_c_Ci_query);
    load_vector_G1(size_f1_g_c_Ci_query, f1_g_c_Ci_query, (char *)"f1_g_c_Ci_query", FOLDER_WWW_DOWNLOAD);
    tri_multi_exponentiation_G1(f1_g_c_Ci_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[2]);
    clear_vec_G1(size_f1_g_c_Ci_query, f1_g_c_Ci_query);

    alloc_init_vec_G1(&f1_g_a_alpha_a_Ai_query, size_f1_g_a_alpha_a_Ai_query);
    load_vector_G1(size_f1_g_a_alpha_a_Ai_query, f1_g_a_alpha_a_Ai_query, (char *)"f1_g_a_alpha_a_Ai_query", FOLDER_WWW_DOWNLOAD);
    multi_exponentiation_G1(size_f1_g_a_alpha_a_Ai_query, f1_g_a_alpha_a_Ai_query, F1, f_ni_answers_G1[3]);
    clear_vec_G1(size_f1_g_a_alpha_a_Ai_query, f1_g_a_alpha_a_Ai_query);

    alloc_init_vec_G1(&f1_g_b_alpha_b_Bi_query, size_f1_g_b_alpha_b_Bi_query);
    load_vector_G1(size_f1_g_b_alpha_b_Bi_query, f1_g_b_alpha_b_Bi_query, (char *)"f1_g_b_alpha_b_Bi_query", FOLDER_WWW_DOWNLOAD);
    tri_multi_exponentiation_G1(f1_g_b_alpha_b_Bi_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[4]);
    clear_vec_G1(size_f1_g_b_alpha_b_Bi_query, f1_g_b_alpha_b_Bi_query);

    alloc_init_vec_G1(&f1_g_c_alpha_c_Ci_query, size_f1_g_c_alpha_c_Ci_query);
    load_vector_G1(size_f1_g_c_alpha_c_Ci_query, f1_g_c_alpha_c_Ci_query, (char *)"f1_g_c_alpha_c_Ci_query", FOLDER_WWW_DOWNLOAD);
    tri_multi_exponentiation_G1(f1_g_c_alpha_c_Ci_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[5]);
    clear_vec_G1(size_f1_g_c_alpha_c_Ci_query, f1_g_c_alpha_c_Ci_query);

    alloc_init_vec_G1(&f1_beta_query, size_f1_beta_query);
    load_vector_G1(size_f1_beta_query, f1_beta_query, (char *)"f1_beta_query", FOLDER_WWW_DOWNLOAD);
    tri_multi_exponentiation_G1(f1_beta_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G1[6]);
    clear_vec_G1(size_f1_beta_query, f1_beta_query);

    // answer queries for f1_h_b_Bi_query
    alloc_init_vec_G2(&f1_h_b_Bi_query, size_f1_h_b_Bi_query);
    load_vector_G2(size_f1_h_b_Bi_query, f1_h_b_Bi_query, (char *)"f1_h_b_Bi_query", FOLDER_WWW_DOWNLOAD);
    tri_multi_exponentiation_G2(f1_h_b_Bi_query, size_f1_vec, F1, size_input, input, size_output, output, f_ni_answers_G2[0]);
    clear_vec_G2(size_f1_h_b_Bi_query, f1_h_b_Bi_query);

    alloc_init_vec_G1(&f2_g_t_i_query, size_f2_g_t_i_query);
    load_vector_G1(size_f2_g_t_i_query, f2_g_t_i_query, (char *)"f2_g_t_i_query", FOLDER_WWW_DOWNLOAD);
    multi_exponentiation_G1(size_f2_g_t_i_query, f2_g_t_i_query, F2, f_ni_answers_G1[1]);
    clear_vec_G1(size_f2_g_t_i_query, f2_g_t_i_query);

#if PROTOCOL == PINOCCHIO_ZK
    //Transformations needed to simulate addition in the exponent of
    //delta_i D(tau) to A(tau), B(tau) and C(tau) wherever they are used

    G1_t g1_tmp;
    G2_t g2_tmp;
    alloc_init_scalar_G1(g1_tmp);
    alloc_init_scalar_G2(g2_tmp);

    G1_exp(g1_tmp, g_a_D, delta_abc[0]);
    G1_mul(f_ni_answers_G1[0], f_ni_answers_G1[0], g1_tmp);
  
//    G1_exp(g1_tmp, g_b_D, delta_abc[1]);
//    G1_mul(f_ni_answers_G1[1], f_ni_answers_G1[1], g1_tmp);

    G1_exp(g1_tmp, g_c_D, delta_abc[2]);
    G1_mul(f_ni_answers_G1[2], f_ni_answers_G1[2], g1_tmp);        

    G1_exp(g1_tmp, g_a_alpha_a_D, delta_abc[0]);
    G1_mul(f_ni_answers_G1[3], f_ni_answers_G1[3], g1_tmp);

    G1_exp(g1_tmp, g_b_alpha_b_D, delta_abc[1]);
    G1_mul(f_ni_answers_G1[4], f_ni_answers_G1[4], g1_tmp);

    G1_exp(g1_tmp, g_c_alpha_c_D, delta_abc[2]);
    G1_mul(f_ni_answers_G1[5], f_ni_answers_G1[5], g1_tmp);

    G1_exp(g1_tmp, g_a_beta_D, delta_abc[0]);
    G1_mul(f_ni_answers_G1[6], f_ni_answers_G1[6], g1_tmp); 
    G1_exp(g1_tmp, g_b_beta_D, delta_abc[1]);
    G1_mul(f_ni_answers_G1[6], f_ni_answers_G1[6], g1_tmp); 
    G1_exp(g1_tmp, g_c_beta_D, delta_abc[2]);
    G1_mul(f_ni_answers_G1[6], f_ni_answers_G1[6], g1_tmp); 

    // update f_ni_answers_G2[0]
    G2_exp(g2_tmp, h_b_D, delta_abc[1]);
    G2_mul(f_ni_answers_G2[0], f_ni_answers_G2[0], g2_tmp);
  
    //f_ni_answers_G2[1] is O.K., we already updated F2 - see
    //computation in compute assignment vectors

    clear_scalar_G1(g1_tmp);
    clear_scalar_G2(g2_tmp);
#endif

    // fold these into a single file f_answers_b_%d
    snprintf(scratch_str, BUFLEN-1, "f_answers_G1_b_%d", i);
    dump_vector_G1(size_answer_G1, f_ni_answers_G1, scratch_str, FOLDER_WWW_DOWNLOAD);
    snprintf(scratch_str, BUFLEN-1, "f_answers_G2_b_%d", i);
    dump_vector_G2(size_answer_G2, f_ni_answers_G2, scratch_str, FOLDER_WWW_DOWNLOAD);
#endif
#ifdef USE_LIBSNARK
    cout << "p_current_mem_before_ls_prover " << getCurrentRSS() << endl;
    cout << "p_peak_mem_before_ls_prover " << getPeakRSS() << endl;
    //    start_profiling();
    typedef Fr<bn128_pp> FieldT;
    bn128_pp::init_public_params();
    string filename = FOLDER_STATE;
    filename += "/libsnark_pk";
    //cout << "FILENAME: " << filename << endl;;
       
    ifstream pkey(filename);
    r1cs_ppzksnark_keypair<bn128_pp> keypair;

    Measurement m_key;
    cout << "reading proving key from file..." << endl;

    m_key.begin_with_init();
    pkey >> keypair.pk;
    m_key.end();
    cout << "m_load_key: " << m_key.get_papi_elapsed_time() << endl;

    r1cs_ppzksnark_primary_input<bn128_pp> primary_input;
    r1cs_ppzksnark_auxiliary_input<bn128_pp> aux_input;


    ifstream inputs("./bin/inputs");
    ifstream outputs("./bin/outputs");
    ifstream variables("./bin/f1vec");
    
    variables >> std::noskipws;
    inputs >> std::noskipws;
    outputs >> std::noskipws;
    char c;

    m_key.begin_with_init();
    int n = 4;
    int numInputs;
    inputs >> numInputs >> c;
    // std::cout << "NUMBER OF INPUTS: " << numInputs << std::endl;
    for (int j = 1; j <= numInputs; j++)
    {
        FieldT currentVar;
        inputs >> currentVar;
        primary_input.push_back(currentVar);
        inputs >> c;
    }

    int num_outputs;
    outputs >> num_outputs >> c;
    // std::cout << "NUMBER OF OUTPUTS: " << num_outputs << std::endl;
    for (int i = 1; i <= num_outputs; i++)
    {
        FieldT currentVar;
        outputs >> currentVar;
        primary_input.push_back(currentVar);
        outputs >> c;
    }

    int num_intermediate_vars;
    variables >> num_intermediate_vars >> c;
    // std::cout << "NUMBER OF VARIABLES: " << num_intermediate_vars << std::endl;
    for (int i = 1; i <= num_intermediate_vars; i++)
    {
        FieldT currentVar;
        variables >> currentVar;
        aux_input.push_back(currentVar);
        variables >> c;
    }

    inputs.close();
    outputs.close();
    variables.close();
   
    m_key.end();
    cout << "p_load_vars " << m_key.get_papi_elapsed_time() << endl;

    m_key.begin_with_init();

    r1cs_ppzksnark_proof<bn128_pp> proof = r1cs_ppzksnark_prover<bn128_pp>(keypair.pk, primary_input, aux_input);

    m_key.end();
    cout << "p_compute_proof " << m_key.get_papi_elapsed_time() << endl;
    filename = FOLDER_STATE;
    filename += "/libsnark_proof";

    m_key.begin_with_init(); 
    ofstream proof_file(filename);
    proof_file << proof; 
    m_key.end();
    cout << "p_write_proof " << m_key.get_papi_elapsed_time() << endl;
   
    proof_file.close();
  
#endif
    m_answer_queries.end();
  }
#endif
}
#else
void ZComputationProver::prover_interactive() {
  // answer commitment query
  load_vector(expansion_factor*size_f1_vec, f1_commitment, (char *)"f1_commitment_query", FOLDER_WWW_DOWNLOAD);
  load_vector(expansion_factor*size_f2_vec, f2_commitment, (char *)"f2_commitment_query", FOLDER_WWW_DOWNLOAD);

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

  clear_vec(size_f1_vec*expansion_factor, f1_commitment);
  clear_vec(size_f2_vec*expansion_factor, f2_commitment);
}
#endif

void ZComputationProver::deduce_queries() {
  int query_id;
  m_plainq.begin_with_init();
  for (int rho=0; rho<num_repetitions; rho++) {
    if (rho == 0) m_plainq.begin_with_init();
    else m_plainq.begin_with_history();

    query_id = 1;
    // create linearity test queries
    for (int i=0; i<NUM_REPS_LIN; i++) {
      v->create_lin_test_queries(size_f1_vec, f1_q1, f1_q2, f1_q3, NULL,
                                 0, NULL, prime);

      v->create_lin_test_queries(size_f2_vec, f2_q1, f2_q2, f2_q3, NULL,
                                 0, NULL, prime);

      //TODO: can be folded into a function
      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f1_vec, f1_q1, scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f1_vec, f1_q2, scratch_str);
      
      // don't dump, but increment query_id
      query_id++;

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q1, scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q2, scratch_str);
      
      query_id++;

      // use one of the linearity queries as self correction queries
      if (i == 0) {
        for (int i=0; i<size_f1_vec; i++)
          mpz_set(f1_q4[i], f1_q1[i]);

        for (int i=0; i<size_f2_vec; i++)
          mpz_set(f2_q4[i], f2_q1[i]);
      }
    }

    for (int i=0; i<n+1; i++) {
      mpz_set_ui(eval_poly_A[i], 0);
      mpz_set_ui(eval_poly_B[i], 0);
      mpz_set_ui(eval_poly_C[i], 0);
    }

    // create zquad correction queries: create q9, q10, q11, and q12
    v->create_div_corr_test_queries(n, size_f1_vec, size_f2_vec,
                                    f1_q1, f1_q2, f1_q3, 
                                    f2_q1, 
                                    NULL, 0, NULL, 
                                    NULL, 0, NULL, 
                                    f1_q4, f2_q4, 
                                    NULL, 
                                    num_aij, num_bij, num_cij, 
                                    set_v_z, 
                                    poly_A, poly_B, poly_C, 
                                    eval_poly_A, eval_poly_B, eval_poly_C,
                                    prime);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f1_vec, f1_q1, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f1_vec, f1_q2, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f1_vec, f1_q3, scratch_str);

    snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
    dump_vector(size_f2_vec, f2_q1, scratch_str);
  }
  m_plainq.end();

  clear_vec(n+1, eval_poly_A);
  clear_vec(n+1, eval_poly_B);
  clear_vec(n+1, eval_poly_C);

  for (int i=0; i<num_aij; i++)
    clear_scalar(poly_A[i].coefficient);

  for (int i=0; i<num_bij; i++)
    clear_scalar(poly_B[i].coefficient);

  for (int i=0; i<num_cij; i++)
    clear_scalar(poly_C[i].coefficient);

  free(poly_A);
  free(poly_B);
  free(poly_C);

  clear_vec(size_f2_vec, poly_A_pv);
  clear_vec(size_f2_vec, poly_B_pv);
  clear_vec(size_f2_vec, poly_C_pv);
}

// Credits: the bitreversal code is from
// http://graphics.stanford.edu/~seander/bithacks.html quoted from
// http://stackoverflow.com/questions/746171/best-algorithm-for-bit-reversal-from-msb-lsb-to-lsb-msb-in-c
// variables are renamed
uint32_t ZComputationProver::zreverse (uint32_t v) {
return ((bit_reversal_table[v & 0xff] << 24) | 
    (bit_reversal_table[(v >> 8) & 0xff] << 16) | 
    (bit_reversal_table[(v >> 16) & 0xff] << 8) |
    (bit_reversal_table[(v >> 24) & 0xff]));
}

mpz_t* ZComputationProver::zcomp_fast_interpolate(int k, mpz_t *A, mpz_t omega_k, mpz_t prime) {
  #ifndef RECURSIVE_FFT
  mpz_t temp, temp2;
  alloc_init_scalar(temp);
  alloc_init_scalar(temp2);

  // bit reversal trick to arrange for in-place FFT
  uint32_t n = (uint32_t) k;
  int num_leading_zeros = ffs(zreverse(k));
  for (uint32_t i=0; i<n; i++) {
    uint32_t j = zreverse(i) >> num_leading_zeros;
    if (j > i) {
      mpz_set(temp, A[i]);
      mpz_set(A[i], A[j]);
      mpz_set(A[j], temp);
    }
  }
    
  mpz_t *m;

  // butterfly updates
  int level = n/2; 
  for (uint32_t L=2; L<=n; L=L+L) {
    int K = level % n;
    for (uint32_t j=0; j<n/L; j++) {
      int k2 = L/2;
      int base = L * j;
      int idx = 0;
      for (int i=0; i<k2; i++) {
        m = &powers_of_omega[idx];
        
        mpz_mul(temp, *m, A[base+k2+i]);
        mpz_set(temp2, A[base+i]);

        mpz_add(A[base+i], A[base+i], temp);
        mpz_sub(A[base+k2+i], temp2, temp);
        if ((L & (4-1)) == 0 || L==n) { 
          //improves performance
          mpz_mod(A[base+i], A[base+i], prime);
          mpz_mod(A[base+k2+i], A[base+k2+i], prime);
        }
        idx = (idx + K)%n;
      }
    }
    level = level/2;
  }
  clear_scalar(temp);
  clear_scalar(temp2);
  return A;
  #else
  mpz_t *A_hat;
  alloc_init_vec(&A_hat, k);
  if (k == 2) {
    mpz_add(A_hat[0], A[0], A[1]);
    mpz_sub(A_hat[1], A[0], A[1]);
    return A_hat;
  }

  mpz_t *A_even;
  mpz_t *A_odd;
  alloc_init_vec(&A_even, k/2);
  alloc_init_vec(&A_odd, k/2);

  for (int i=0; i<k/2; i++) {
    mpz_set(A_even[i], A[2*i]);
    mpz_set(A_odd[i], A[2*i+1]);
  }
  
  mpz_t omega_sqr;
  mpz_init(omega_sqr);
  mpz_mul(omega_sqr, omega_k, omega_k);
  mpz_mod(omega_sqr, omega_sqr, prime);

  mpz_t *A_even_hat = zcomp_fast_interpolate(k/2, A_even, omega_sqr, prime);
  mpz_t *A_odd_hat = zcomp_fast_interpolate(k/2, A_odd, omega_sqr, prime);
  
  mpz_clear(omega_sqr);

  mpz_t m, temp;
  mpz_init_set_ui(m, 1);
  mpz_init(temp);

  int k2 = k/2, i = 0;
  for (int i=0; i<k2; i++) { 
    mpz_mul(temp, m, A_odd_hat[i]);
    
    mpz_add(A_hat[i], A_even_hat[i], temp);
    mpz_mod(A_hat[i], A_hat[i], prime);

    mpz_sub(A_hat[i+k2], A_even_hat[i], temp);
    mpz_mod(A_hat[i+k2], A_hat[i+k2], prime);
    
    mpz_mul(m, m, omega_k);
    mpz_mod(m, m, prime);
  }
 
  mpz_clear(m);
  mpz_clear(temp);
  clear_vec(k2, A_even_hat);
  clear_vec(k2, A_odd_hat);
  return A_hat;
  #endif
}

void ZComputationProver::zcomp_interpolate(int level, int j, int index, vec_ZZ_p *evals) {
  if (level == 0) {
    interpolation_tree[index] = set_v[j] * (*evals)[j];
  } else {
    zcomp_interpolate(level-1, 2*j, 2*index+1, evals);
    zcomp_interpolate(level-1, 2*j+1, 2*index+2, evals);

    mul(interpolation_tree[2*index+1], interpolation_tree[2*index+1], poly_tree[2*index+2]);
    mul(interpolation_tree[2*index+2], interpolation_tree[2*index+2], poly_tree[2*index+1]);

    add(interpolation_tree[index], interpolation_tree[2*index+1], interpolation_tree[2*index+2]);
  }
}

void ZComputationProver::build_poly_tree(int level, int j, int index) {
  if (level == 0) {
    single_root[0] = qap_roots[j];
    BuildFromRoots(poly_tree[index], single_root);
  } else {
    build_poly_tree(level-1, 2*j, 2*index+1);
    build_poly_tree(level-1, 2*j+1, 2*index+2);
    mul(poly_tree[index], poly_tree[2*index+1], poly_tree[2*index+2]);
  }
}
