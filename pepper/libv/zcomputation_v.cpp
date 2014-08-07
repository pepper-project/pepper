#include <libv/zcomputation_v.h>

ZComputationVerifier::ZComputationVerifier(int batch, int reps, int ip_size,
    int out_size, int num_variables, int num_constraints, int optimize_answers,
    char *prover_url, const char *name_prover, int size_aij, int size_bij,
    int size_cij, const char *file_name_qap) : Verifier(batch, NUM_REPS_PCP, ip_size, optimize_answers, prover_url, name_prover) {
  chi = num_constraints;
  n_prime = num_variables;
  num_aij = size_aij;
  num_bij = size_bij;
  num_cij = size_cij;

  n = n_prime + ip_size + out_size;
  size_input = ip_size;
  size_output = out_size;

  size_f1_vec = n_prime;
  size_f2_vec = chi + 1;

  num_verification_runs = NUM_VERIFICATION_RUNS;
  init_qap(file_name_qap);
  init_state();
}

ZComputationVerifier::~ZComputationVerifier() {
#if NONINTERACTIVE == 1
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


  // VK
  clear_scalar_G2(h);
  clear_scalar_G2(h_alpha);
  clear_scalar_G2(h_gamma);
  clear_scalar_G2(h_beta_a_gamma);
  clear_scalar_G2(h_beta_b_gamma);
  clear_scalar_G2(h_beta_c_gamma);
  clear_scalar_G2(h_D);
  clear_scalar_G1(g_A0);
  clear_scalar_G2(h_B0);
  clear_scalar_G1(g_C0);
  clear_vec_G1(size_input+size_output, g_Ai_io);
#else

#if PROTOCOL == PINOCCHIO_ZK
  clear_scalar_G1(g_a_D);
  clear_scalar_G1(g_b_D);
  //clear_scalar_G1(g_c_D);
  clear_scalar_G2(h_b_D);

  clear_scalar_G1(g_a_alpha_a_D);
  clear_scalar_G1(g_b_alpha_b_D);
  clear_scalar_G1(g_c_alpha_c_D);
  clear_scalar_G1(g_a_beta_D);
  clear_scalar_G1(g_b_beta_D);
  clear_scalar_G1(g_c_beta_D);
#endif

  // VK
  clear_scalar_G1(g);
  clear_scalar_G2(h);
  clear_scalar_G2(h_alpha_a);
  clear_scalar_G1(g_alpha_b);
  clear_scalar_G2(h_alpha_b);
  clear_scalar_G2(h_alpha_c);
  clear_scalar_G2(h_gamma);
  clear_scalar_G2(h_beta_gamma);
  clear_scalar_G1(g_gamma);
  clear_scalar_G1(g_beta_gamma);
  clear_scalar_G1(g_c_D);
  clear_scalar_G1(g_a_A0);
  clear_scalar_G2(h_b_B0);
  clear_scalar_G1(g_c_C0);
  clear_vec_G1(size_input+size_output, g_a_Ai_io);
  clear_scalar(r_c_D);
  
  #if PUBLIC_VERIFIER == 0
  clear_vec(size_input+size_output, Ai_io);
  clear_scalar_G1(g_a_base);
  #endif

#endif
  // answers
  clear_vec_G1(size_answer_G1, f_ni_answers_G1);
  clear_vec_G2(size_answer_G2, f_ni_answers_G2);
#else
  clear_scalar(A_tau);
  clear_scalar(B_tau);
  clear_scalar(C_tau);

  clear_vec(num_repetitions*(size_input+size_output+1), A_tau_io);
  clear_vec(num_repetitions*(size_input+size_output+1), B_tau_io);
  clear_vec(num_repetitions*(size_input+size_output+1), C_tau_io);

  clear_vec(size_input+size_output, input);
  clear_vec(size_input+size_output, input_q);
  clear_vec(size_f2_vec, set_v);

  clear_vec(num_repetitions * num_lin_pcp_queries, f_answers);
  clear_vec(expansion_factor, temp_arr);
  clear_vec(expansion_factor, temp_arr2);

  clear_scalar(temp);
  clear_scalar(temp2);
  clear_scalar(temp3);
  clear_scalar(lhs);
  clear_scalar(rhs);
  clear_vec(NUM_REPS_PCP, d_star);
  clear_scalar(omega);
#endif
}

void ZComputationVerifier::init_qap(const char *file_name_qap) {
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

  // create vectors to store the evaluations of the polynomial at tau
  alloc_init_vec(&eval_poly_A, n+1);
  alloc_init_vec(&eval_poly_B, n+1);
  alloc_init_vec(&eval_poly_C, n+1);

  alloc_init_vec(&A_tau_io, num_repetitions*(size_input+size_output+1));
  alloc_init_vec(&B_tau_io, num_repetitions*(size_input+size_output+1));
  alloc_init_vec(&C_tau_io, num_repetitions*(size_input+size_output+1));


  // open the file
  FILE *fp = fopen(file_name_qap, "r");
  if (fp == NULL) {
    cout<<"LOG: Cannot read "<<file_name_qap<<endl;
    exit(1);
  }

  char line[BUFLEN];
  mpz_t temp;
  alloc_init_scalar(temp);

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
  clear_scalar(temp);
  
#if NONINTERACTIVE == 1
  num_bits_in_prime = 256;
#else
  // set prime size based on name of the computation in case of Zaatar
  string str(file_name_qap);
  if (str.find("bisect_sfdl") != std::string::npos) {
    num_bits_in_prime = 220;
  } else if (str.find("pd2_sfdl") != std::string::npos) {
    num_bits_in_prime = 220;
  } else {
    num_bits_in_prime = 128;
  }

  cout<<"LOG: Using a prime of size "<<num_bits_in_prime<<endl;
#endif
}

void ZComputationVerifier::init_state() {
#if NONINTERACTIVE == 1
  num_bits_in_input = 32;

  Verifier::init_state();

  // allocate input and output contiguously
  alloc_init_vec(&input, size_input+size_output);
  output = &input[size_input];
  alloc_init_vec(&input_q, size_input);

  alloc_init_vec(&set_v, size_f2_vec);
  v->compute_set_v(size_f2_vec, set_v, prime);

  // init pairing
  init_pairing_from_file(PAIRING_PARAM, prime);

#if GGPR == 1
  size_vk_G1 = 2 + size_input + size_output;
  size_vk_G2 = 8;

  size_answer_G1 = 9;
  size_answer_G2 = 1;

  // initialize storage for EK
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

  // initialize storage for VK
  //alloc_init_scalar_G1(g);
  alloc_init_scalar_G2(h);
  alloc_init_scalar_G2(h_alpha);
  alloc_init_scalar_G2(h_gamma);
  alloc_init_scalar_G2(h_beta_a_gamma);
  alloc_init_scalar_G2(h_beta_b_gamma);
  alloc_init_scalar_G2(h_beta_c_gamma);
  alloc_init_scalar_G2(h_D);
  alloc_init_scalar_G1(g_A0);
  alloc_init_scalar_G2(h_B0);
  alloc_init_scalar_G1(g_C0);
  alloc_init_vec_G1(&g_Ai_io, size_input+size_output);
#else
  size_vk_G1 = 5 + size_input + size_output;
  size_vk_G2 = 7;

  size_answer_G1 = 7;
  size_answer_G2 = 1;

  // initialize storage for EK
  size_f1_g_a_Ai_query = size_f1_g_a_alpha_a_Ai_query = size_f1_vec;
  size_f1_g_b_Bi_query = size_f1_h_b_Bi_query = size_f1_g_c_Ci_query = size_f1_vec+size_input+size_output;
  size_f1_g_b_alpha_b_Bi_query = size_f1_g_c_alpha_c_Ci_query = size_f1_vec+size_input+size_output;
  //size_f2_h_t_i_query = size_f2_vec;
  size_f2_g_t_i_query = size_f2_vec;
  size_f1_beta_query = size_f1_vec+size_input+size_output;

#if PROTOCOL == PINOCCHIO_ZK
  alloc_init_scalar_G1(g_a_D);
  alloc_init_scalar_G1(g_b_D);
  //alloc_init_scalar_G1(g_c_D);
  alloc_init_scalar_G2(h_b_D);

  alloc_init_scalar_G1(g_a_alpha_a_D);
  alloc_init_scalar_G1(g_b_alpha_b_D);
  alloc_init_scalar_G1(g_c_alpha_c_D);
  alloc_init_scalar_G1(g_a_beta_D);
  alloc_init_scalar_G1(g_b_beta_D);
  alloc_init_scalar_G1(g_c_beta_D);
#endif

  // initialize storage for VK
  alloc_init_scalar_G1(g);
  alloc_init_scalar_G2(h);
  alloc_init_scalar_G2(h_alpha_a);
  alloc_init_scalar_G1(g_alpha_b);
  alloc_init_scalar_G2(h_alpha_b);
  alloc_init_scalar_G2(h_alpha_c);
  alloc_init_scalar_G2(h_gamma);
  alloc_init_scalar_G2(h_beta_gamma);
  alloc_init_scalar_G1(g_gamma);
  alloc_init_scalar_G1(g_beta_gamma);
  alloc_init_scalar_G1(g_c_D);
  alloc_init_scalar_G1(g_a_A0);
  alloc_init_scalar_G2(h_b_B0);
  alloc_init_scalar_G1(g_c_C0);
  alloc_init_vec_G1(&g_a_Ai_io, size_input+size_output);
  alloc_init_scalar(r_c_D);
  
  #if PUBLIC_VERIFIER == 0
  alloc_init_vec(&Ai_io, size_input+size_output);
  alloc_init_scalar_G1(g_a_base);
  #endif

#endif
  // initialize storage for answers
  alloc_init_vec_G1(&f_ni_answers_G1, size_answer_G1);
  alloc_init_vec_G2(&f_ni_answers_G2, size_answer_G2);

#ifdef DEBUG_TEST_ENABLE
  // some test cases
  G1_t g, result1, result2;
  mpz_t a,b,x,y, tmp1, tmp2;

  alloc_init_scalar_G1(g);
  alloc_init_scalar_G1(result1);
  alloc_init_scalar_G1(result2);

  G1_random(g);

  alloc_init_scalar(a);
  alloc_init_scalar(b);
  alloc_init_scalar(x);
  alloc_init_scalar(y);
  alloc_init_scalar(tmp1);
  alloc_init_scalar(tmp2);

  v->get_random_priv(a, prime);
  v->get_random_priv(b, prime);
  v->get_random_priv(x, prime);
  v->get_random_priv(y, prime);

  G1_t* base;
  mpz_t * exp;

  alloc_init_vec_G1(&base, 2);
  alloc_init_vec(&exp, 2);

  G1_exp(base[0], g, a);
  G1_exp(base[1], g, b);

  mpz_set(exp[0], x);
  mpz_set(exp[1], y);

  multi_exponentiation_G1(2, base, exp, result1);

  mpz_mul(tmp1, a, x);
  mpz_mod(tmp1, tmp1, prime);
  mpz_mul(tmp2, b, y);
  mpz_mod(tmp2, tmp2, prime);
  mpz_add(tmp1, tmp1, tmp2);
  mpz_mod(tmp1, tmp1, prime);
  G1_exp(result2, g, tmp1);

  if (G1_cmp(result1, result2)) {
    cout << "BUG: test failed" << endl;
    exit(1);
  }
#endif

#else
//  num_bits_in_prime = 128; this is set in init_qap based on the
//  computation for zaatar
  num_bits_in_input = 32;

  crypto_in_use = CRYPTO_ELGAMAL;
  png_in_use = PNG_CHACHA;

  num_lin_pcp_queries = NUM_LIN_PCP_QUERIES;

  Verifier::init_state();

  // allocate input and output contiguously
  alloc_init_vec(&input, size_input+size_output);
  output = &input[size_input];
  alloc_init_vec(&input_q, size_input+size_output);
  output_q = &input_q[size_input];

  alloc_init_vec(&set_v, size_f2_vec);
  
  #if FAST_FOURIER_INTERPOLATION ==1
    alloc_init_scalar(omega);
    v->generate_root_of_unity(size_f2_vec, prime);
    v->get_root_of_unity(&omega);
    v->compute_set_v(size_f2_vec, set_v, omega, prime);
  #else
    v->compute_set_v(size_f2_vec, set_v, prime);
  #endif

  alloc_init_vec(&f1_commitment, expansion_factor*size_f1_vec);
  alloc_init_vec(&f2_commitment, expansion_factor*size_f2_vec);
  alloc_init_vec(&f1_consistency, size_f1_vec);
  alloc_init_vec(&f2_consistency, size_f2_vec);

  alloc_init_vec(&f1_q1, size_f1_vec);
  alloc_init_vec(&f1_q2, size_f1_vec);
  alloc_init_vec(&f1_q3, size_f1_vec);
  alloc_init_vec(&f1_q4, size_f1_vec);

  alloc_init_vec(&f2_q1, size_f2_vec);
  alloc_init_vec(&f2_q2, size_f2_vec);
  alloc_init_vec(&f2_q3, size_f2_vec);
  alloc_init_vec(&f2_q4, size_f2_vec);

  alloc_init_vec(&f_answers, num_repetitions * num_lin_pcp_queries);
  alloc_init_vec(&temp_arr, expansion_factor);
  alloc_init_vec(&temp_arr2, expansion_factor);

  alloc_init_scalar(temp);
  alloc_init_scalar(temp2);
  alloc_init_scalar(temp3);
  alloc_init_scalar(lhs);
  alloc_init_scalar(rhs);
  alloc_init_vec(&d_star, NUM_REPS_PCP);

  alloc_init_scalar(A_tau);
  alloc_init_scalar(B_tau);
  alloc_init_scalar(C_tau);

  // To create consistency and commitment queries.
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
#endif
}

void ZComputationVerifier::create_input() {
  // as many computations as inputs
  for (int k=0; k<batch_size; k++) {
    int input_size = size_input;
    //v->get_random_vec(size_input, input, num_bits_in_input);
    //v->add_sign(size_input, input);

    input_creator->create_input(input_q, input_size);

    snprintf(scratch_str, BUFLEN-1, "input_q_b_%d", k);
    dump_vector(input_size, input_q, scratch_str);
    send_file(scratch_str);

    convert_to_z(input_size, input, input_q, prime);

    snprintf(scratch_str, BUFLEN-1, "input_b_%d", k);
    dump_vector(size_input, input, scratch_str);
    send_file(scratch_str);
  }
}


#if NONINTERACTIVE == 1
#if GGPR == 1
#ifdef DEBUG_TEST_ENABLE
G1_t g;
G1_t g_alpha;
G1_t g_beta_a, g_beta_b, g_beta_c;

mpz_t t, D;
#endif
void ZComputationVerifier::create_noninteractive_GGPR_query() {
#ifndef DEBUG_TEST_ENABLE
  G1_t g;
  G1_t g_alpha;
  G1_t g_beta_a, g_beta_b, g_beta_c;

  mpz_t t, D;
#endif

  G1_t *vk_G1;
  G2_t *vk_G2;

  mpz_t alpha, beta_a, beta_b, beta_c, gamma;

  //G1_t g1_tmp1, g1_tmp2;
  //mpz_t tmp;

  // initialize local variables.
  alloc_init_scalar_G1(g);
  alloc_init_scalar_G1(g_alpha);
  alloc_init_scalar_G1(g_beta_a);
  alloc_init_scalar_G1(g_beta_b);
  alloc_init_scalar_G1(g_beta_c);

  alloc_init_vec_G1(&vk_G1, size_vk_G1);
  alloc_init_vec_G2(&vk_G2, size_vk_G2);

  alloc_init_scalar(alpha);
  alloc_init_scalar(beta_a);
  alloc_init_scalar(beta_b);
  alloc_init_scalar(beta_c);
  alloc_init_scalar(gamma);

  alloc_init_scalar(t);
  alloc_init_scalar(D);

  //alloc_init_scalar_G1(g1_tmp1);
  //alloc_init_scalar_G1(g1_tmp2);
  //alloc_init_scalar(tmp);

  // choose a generator g, h
  G1_random(g);
  G2_random(h);

  // randomly pick \alpha, \beta_a, \beta_b, \beta_c, \gamma from field.
  // s should be the random point used in the evaluation of the polynomial
  v->get_random_priv(t, prime);
  v->get_random_priv(alpha, prime);
  v->get_random_priv(beta_a, prime);
  v->get_random_priv(beta_b, prime);
  v->get_random_priv(beta_c, prime);
  v->get_random_priv(gamma, prime);

  // compute r_c=r_a*r_b, g_a=g^r_a, g_b=g^r_b, g_c=g^r_c.
  G1_exp(g_alpha, g, alpha);
  G1_exp(g_beta_a, g, beta_a);
  G1_exp(g_beta_b, g, beta_b);
  G1_exp(g_beta_c, g, beta_c);

  for (int i = 0; i< n + 1; i++) {
    mpz_set_ui(eval_poly_A[i], 0);
    mpz_set_ui(eval_poly_B[i], 0);
    mpz_set_ui(eval_poly_C[i], 0);
  }

  v->evaluate_polynomial_at_random_point(
      n, size_f2_vec,
      t, D,
      num_aij, num_bij, num_cij,
      set_v,
      poly_A, poly_B, poly_C,
      eval_poly_A, eval_poly_B, eval_poly_C,
      prime);

  // compute f1_g_Ai_query, f1_g_Bi_query, f1_g_Ci_query
  // eval_poly_A/B/C carries the evaluation of polynomials at a secret point
  // the order of elements inside eval_poly_A is constant, variables, input, output? yes
  //element_pp_t fixed_exp;
  G1_fixed_exp(f1_g_Ai_query, g, eval_poly_A+1, size_f1_g_Ai_query);
  G1_fixed_exp(f1_g_Bi_query, g, eval_poly_B+1, size_f1_g_Bi_query);
  G2_fixed_exp(f1_h_Bi_query, h, eval_poly_B+1, size_f1_h_Bi_query);
  G1_fixed_exp(f1_g_Ci_query, g, eval_poly_C+1, size_f1_g_Ci_query);

  // compute f1_g_a_alpha_Ai_query, f1_g_b_alpha_Bi_query, f1_g_c_alpha_Ci_query
  G1_fixed_exp(f1_g_alpha_Ai_query, g_alpha, eval_poly_A+1, size_f1_g_alpha_Ai_query);
  G1_fixed_exp(f1_g_alpha_Bi_query, g_alpha, eval_poly_B+1, size_f1_g_alpha_Bi_query);
  G1_fixed_exp(f1_g_alpha_Ci_query, g_alpha, eval_poly_C+1, size_f1_g_alpha_Ci_query);

  // compute f1_g_beta_a_Ai_query, f1_g_beta_b_Bi_query, f1_g_beta_c_Ci_query
  G1_fixed_exp(f1_g_beta_a_Ai_query, g_beta_a, eval_poly_A+1, size_f1_g_beta_a_Ai_query);
  G1_fixed_exp(f1_g_beta_b_Bi_query, g_beta_b, eval_poly_B+1, size_f1_g_beta_b_Bi_query);
  G1_fixed_exp(f1_g_beta_c_Ci_query, g_beta_c, eval_poly_C+1, size_f1_g_beta_c_Ci_query);

  G2_geom_fixed_exp(f2_g_t_i_query, g, t, prime, size_f2_g_t_i_query);
  G2_geom_fixed_exp(f2_g_alpha_t_i_query, g_alpha, t, prime, size_f2_g_alpha_t_i_query);

  //element_pp_clear(fixed_exp);

  // create VK
  G2_exp(h_alpha, h, alpha);
  G2_exp(h_gamma, h, gamma);
  G2_exp(h_beta_a_gamma, h_gamma, beta_a);
  G2_exp(h_beta_b_gamma, h_gamma, beta_b);
  G2_exp(h_beta_c_gamma, h_gamma, beta_c);
  G2_exp(h_D, h, D);
  G1_exp(g_A0, g, eval_poly_A[0]);
  G2_exp(h_B0, h, eval_poly_B[0]);
  G1_exp(g_C0, g, eval_poly_C[0]);
  G1_fixed_exp(g_Ai_io, g, eval_poly_A+1+size_f1_g_Ai_query, size_input+size_output);

  // dump the keys into files.
  // dump EK
  dump_vector_G1(size_f1_g_Ai_query, f1_g_Ai_query, (char *)"f1_g_Ai_query");
  dump_vector_G1(size_f1_g_Bi_query, f1_g_Bi_query, (char *)"f1_g_Bi_query");
  dump_vector_G2(size_f1_h_Bi_query, f1_h_Bi_query, (char *)"f1_h_Bi_query");
  dump_vector_G1(size_f1_g_Ci_query, f1_g_Ci_query, (char *)"f1_g_Ci_query");
  dump_vector_G1(size_f2_g_t_i_query, f2_g_t_i_query, (char *)"f2_g_t_i_query");

  dump_vector_G1(size_f1_g_alpha_Ai_query, f1_g_alpha_Ai_query, (char *)"f1_g_alpha_Ai_query");
  dump_vector_G1(size_f1_g_alpha_Bi_query, f1_g_alpha_Bi_query, (char *)"f1_g_alpha_Bi_query");
  dump_vector_G1(size_f1_g_alpha_Ci_query, f1_g_alpha_Ci_query, (char *)"f1_g_alpha_Ci_query");
  dump_vector_G1(size_f2_g_alpha_t_i_query, f2_g_alpha_t_i_query, (char *)"f2_g_alpha_t_i_query");

  dump_vector_G1(size_f1_g_beta_a_Ai_query, f1_g_beta_a_Ai_query, (char *)"f1_g_beta_a_Ai_query");
  dump_vector_G1(size_f1_g_beta_b_Bi_query, f1_g_beta_b_Bi_query, (char *)"f1_g_beta_b_Bi_query");
  dump_vector_G1(size_f1_g_beta_c_Ci_query, f1_g_beta_c_Ci_query, (char *)"f1_g_beta_c_Ci_query");

  // dump VK, first copy elements into the array.
  G1_set(vk_G1[0], g_A0);
  G1_set(vk_G1[1], g_C0);
  
  for (int i = 0; i < size_input+size_output;i++) {
    G1_set(vk_G1[i+2], g_Ai_io[i]);
  }

  G2_set(vk_G2[0], h);
  G2_set(vk_G2[1], h_alpha);
  G2_set(vk_G2[2], h_gamma);
  G2_set(vk_G2[3], h_beta_a_gamma);
  G2_set(vk_G2[4], h_beta_b_gamma);
  G2_set(vk_G2[5], h_beta_c_gamma);
  G2_set(vk_G2[6], h_D);
  G2_set(vk_G2[7], h_B0);

  // dump the verification key in two separate pieces, one for G1 and
  // another for G2.
  dump_vector_G1(size_vk_G1, vk_G1, (char *)"f_verification_key_G1");
  dump_vector_G2(size_vk_G2, vk_G2, (char *)"f_verification_key_G2");

  // clear local variables.
#ifndef DEBUG_TEST_ENABLE
  clear_scalar_G1(g);
  clear_scalar_G1(g_alpha);
  clear_scalar_G1(g_beta_a);
  clear_scalar_G1(g_beta_b);
  clear_scalar_G1(g_beta_c);
  clear_scalar(t);
  clear_scalar(D);
#endif

  clear_vec_G1(size_vk_G1, vk_G1);
  clear_vec_G2(size_vk_G2, vk_G2);

  clear_scalar(alpha);
  clear_scalar(beta_a);
  clear_scalar(beta_b);
  clear_scalar(beta_c);
  clear_scalar(gamma);

  // send the queries
  send_file((char *)"f1_g_Ai_query");
  send_file((char *)"f1_g_Bi_query");
  send_file((char *)"f1_h_Bi_query");
  send_file((char *)"f1_g_Ci_query");
  send_file((char *)"f1_g_alpha_Ai_query");
  send_file((char *)"f1_g_alpha_Bi_query");
  send_file((char *)"f1_g_alpha_Ci_query");
  send_file((char *)"f1_g_beta_a_Ai_query");
  send_file((char *)"f1_g_beta_b_Bi_query");
  send_file((char *)"f1_g_beta_c_Ci_query");
  send_file((char *)"f2_g_t_i_query");
  send_file((char *)"f2_g_alpha_t_i_query");
}
#endif

#if !(GGPR == 1)
#ifdef DEBUG_TEST_ENABLE
G1_t g;
G1_t g_a, g_b, g_c;
G2_t h_b;
G1_t g_a_alpha_a, g_b_alpha_b, g_c_alpha_c;
G1_t g_a_beta, g_b_beta, g_c_beta;
G2_t h_beta;
mpz_t t, D;
#endif
#endif

void ZComputationVerifier::create_noninteractive_query() {
#if GGPR == 1
  create_noninteractive_GGPR_query();
#else
#ifndef DEBUG_TEST_ENABLE
  G1_t g_a, g_b, g_c;
  G2_t h_b;
  G1_t g_a_alpha_a, g_b_alpha_b, g_c_alpha_c;
  G1_t g_a_beta, g_b_beta, g_c_beta;
  G2_t h_beta;
  mpz_t t, D;
#endif
  G1_t *vk_G1;
  G2_t *vk_G2;

  mpz_t r_a, r_b, r_c, alpha_a, alpha_b, alpha_c, beta, gamma;

  // initialize local variables.
  alloc_init_scalar_G1(g_a);
  alloc_init_scalar_G1(g_b);
  alloc_init_scalar_G2(h_b);
  alloc_init_scalar_G1(g_c);
  alloc_init_scalar_G1(g_a_alpha_a);
  alloc_init_scalar_G1(g_b_alpha_b);
  alloc_init_scalar_G1(g_c_alpha_c);
  alloc_init_scalar_G1(g_a_beta);
  alloc_init_scalar_G1(g_b_beta);
  alloc_init_scalar_G1(g_c_beta);
  alloc_init_scalar_G2(h_beta);

  alloc_init_vec_G1(&vk_G1, size_vk_G1);
  alloc_init_vec_G2(&vk_G2, size_vk_G2);

  alloc_init_scalar(r_a);
  alloc_init_scalar(r_b);
  alloc_init_scalar(r_c);
  alloc_init_scalar(alpha_a);
  alloc_init_scalar(alpha_b);
  alloc_init_scalar(alpha_c);
  alloc_init_scalar(beta);
  alloc_init_scalar(gamma);

  alloc_init_scalar(t);
  alloc_init_scalar(D);

  // choose a generator g, h
  G1_random(g);
  G2_random(h);

  // randomly pick r_a, r_b, s, \alpha_a, \alpha_b, \alpah_c, \beta, \gamma from field.
  v->get_random_priv(r_a, prime);
  v->get_random_priv(r_b, prime);
  // s should be the random point used in the evaluation of the polynomial
  v->get_random_priv(t, prime);
  v->get_random_priv(alpha_a, prime);
  v->get_random_priv(alpha_b, prime);
  v->get_random_priv(alpha_c, prime);
  v->get_random_priv(beta, prime);
  v->get_random_priv(gamma, prime);

  // compute r_c=r_a*r_b, g_a=g^r_a, g_b=g^r_b, g_c=g^r_c.
  mpz_mul(r_c, r_a, r_b);
  mpz_mod(r_c, r_c, prime);
  G1_exp(g_a, g, r_a);
  G1_exp(g_b, g, r_b);
  G2_exp(h_b, h, r_b);
  G1_exp(g_c, g, r_c);
  G1_exp(g_a_alpha_a, g_a, alpha_a);
  G1_exp(g_b_alpha_b, g_b, alpha_b);
  G1_exp(g_c_alpha_c, g_c, alpha_c);
  G1_exp(g_a_beta, g_a, beta);
  G1_exp(g_b_beta, g_b, beta);
  G1_exp(g_c_beta, g_c, beta);

  for (int i = 0; i< n + 1; i++) {
    mpz_set_ui(eval_poly_A[i], 0);
    mpz_set_ui(eval_poly_B[i], 0);
    mpz_set_ui(eval_poly_C[i], 0);
  }

  v->evaluate_polynomial_at_random_point(
      n, size_f2_vec,
      t, D,
      num_aij, num_bij, num_cij,
      set_v,
      poly_A, poly_B, poly_C,
      eval_poly_A, eval_poly_B, eval_poly_C,
      prime);

  // compute f1_g_a_Ai_query, f1_g_b_Bi_query, f1_g_c_Ci_query
  // eval_poly_A/B/C carries the evaluation of polynomials at a secret point
  // the order of entries inside eval_poly_A is constant, variables, input, output? yes
  alloc_init_vec_G1(&f1_g_a_Ai_query, size_f1_g_a_Ai_query);
  G1_fixed_exp(f1_g_a_Ai_query, g_a, eval_poly_A+1, size_f1_g_a_Ai_query);
  dump_vector_G1(size_f1_g_a_Ai_query, f1_g_a_Ai_query, (char *)"f1_g_a_Ai_query");
  clear_vec_G1(size_f1_g_a_Ai_query, f1_g_a_Ai_query);

  alloc_init_vec_G1(&f1_g_b_Bi_query, size_f1_g_b_Bi_query);
  G1_fixed_exp(f1_g_b_Bi_query, g_b, eval_poly_B+1, size_f1_g_b_Bi_query);
  dump_vector_G1(size_f1_g_b_Bi_query, f1_g_b_Bi_query, (char *)"f1_g_b_Bi_query");
  clear_vec_G1(size_f1_g_b_Bi_query, f1_g_b_Bi_query);

  alloc_init_vec_G2(&f1_h_b_Bi_query, size_f1_h_b_Bi_query);
  G2_fixed_exp(f1_h_b_Bi_query, h_b, eval_poly_B+1, size_f1_h_b_Bi_query);
  dump_vector_G2(size_f1_h_b_Bi_query, f1_h_b_Bi_query, (char *)"f1_h_b_Bi_query");
  clear_vec_G2(size_f1_h_b_Bi_query, f1_h_b_Bi_query);

  alloc_init_vec_G1(&f1_g_c_Ci_query, size_f1_g_c_Ci_query);
  G1_fixed_exp(f1_g_c_Ci_query, g_c, eval_poly_C+1, size_f1_g_c_Ci_query);
  dump_vector_G1(size_f1_g_c_Ci_query, f1_g_c_Ci_query, (char *)"f1_g_c_Ci_query");
  clear_vec_G1(size_f1_g_c_Ci_query, f1_g_c_Ci_query);

  // compute f1_g_a_alpha_a_Ai_query, f1_g_b_alpha_b_Bi_query, f1_g_c_alpha_c_Ci_query
  alloc_init_vec_G1(&f1_g_a_alpha_a_Ai_query, size_f1_g_a_alpha_a_Ai_query);
  G1_fixed_exp(f1_g_a_alpha_a_Ai_query, g_a_alpha_a, eval_poly_A+1, size_f1_g_a_alpha_a_Ai_query);
  dump_vector_G1(size_f1_g_a_alpha_a_Ai_query, f1_g_a_alpha_a_Ai_query, (char *)"f1_g_a_alpha_a_Ai_query");
  clear_vec_G1(size_f1_g_a_alpha_a_Ai_query, f1_g_a_alpha_a_Ai_query);

  alloc_init_vec_G1(&f1_g_b_alpha_b_Bi_query, size_f1_g_b_alpha_b_Bi_query);
  G1_fixed_exp(f1_g_b_alpha_b_Bi_query, g_b_alpha_b, eval_poly_B+1, size_f1_g_b_alpha_b_Bi_query);
  dump_vector_G1(size_f1_g_b_alpha_b_Bi_query, f1_g_b_alpha_b_Bi_query, (char *)"f1_g_b_alpha_b_Bi_query");
  clear_vec_G1(size_f1_g_b_alpha_b_Bi_query, f1_g_b_alpha_b_Bi_query);

  alloc_init_vec_G1(&f1_g_c_alpha_c_Ci_query, size_f1_g_c_alpha_c_Ci_query);
  G1_fixed_exp(f1_g_c_alpha_c_Ci_query, g_c_alpha_c, eval_poly_C+1, size_f1_g_c_alpha_c_Ci_query);
  dump_vector_G1(size_f1_g_c_alpha_c_Ci_query, f1_g_c_alpha_c_Ci_query, (char *)"f1_g_c_alpha_c_Ci_query");
  clear_vec_G1(size_f1_g_c_alpha_c_Ci_query, f1_g_c_alpha_c_Ci_query);

  alloc_init_vec_G1(&f2_g_t_i_query, size_f2_g_t_i_query);
  G1_geom_fixed_exp(f2_g_t_i_query, g, t, prime, size_f2_g_t_i_query);
  dump_vector_G1(size_f2_g_t_i_query, f2_g_t_i_query, (char *)"f2_g_t_i_query");
  clear_vec_G1(size_f2_g_t_i_query, f2_g_t_i_query);

  // compute f1_beta_query
  alloc_init_vec_G1(&f1_beta_query, size_f1_beta_query);
  G1_fixed_exp(f1_beta_query, g_a_beta, eval_poly_A+1, size_f1_beta_query);
  G1_mul_fixed_exp(f1_beta_query, g_b_beta, eval_poly_B+1, size_f1_beta_query);
  G1_mul_fixed_exp(f1_beta_query, g_c_beta, eval_poly_C+1, size_f1_beta_query);
  dump_vector_G1(size_f1_beta_query, f1_beta_query, (char *)"f1_beta_query");
  clear_vec_G1(size_f1_beta_query, f1_beta_query);

#if PROTOCOL == PINOCCHIO_ZK
  G1_exp(g_a_D, g_a, D);
  G1_exp(g_b_D, g_b, D);
  //alloc_init_scalar_G1(g_c_D);
  G2_exp(h_b_D, h_b, D);

  G1_exp(g_a_alpha_a_D, g_a_alpha_a, D);
  G1_exp(g_b_alpha_b_D, g_b_alpha_b, D);
  G1_exp(g_c_alpha_c_D, g_c_alpha_c, D);
  G1_exp(g_a_beta_D, g_a_beta, D);
  G1_exp(g_b_beta_D, g_b_beta, D);
  G1_exp(g_c_beta_D, g_c_beta, D);
#endif

  // create VK
  G2_exp(h_alpha_a, h, alpha_a);
  G2_exp(h_alpha_b, h, alpha_b);
  G1_exp(g_alpha_b, g, alpha_b);
  G2_exp(h_alpha_c, h, alpha_c);
  G2_exp(h_beta, h, beta);
  G2_exp(h_gamma, h, gamma);
  G2_exp(h_beta_gamma, h_gamma, beta);
  G1_exp(g_gamma, g, gamma);
  G1_exp(g_beta_gamma, g_gamma, beta);
  G1_exp(g_c_D, g_c, D);
  G1_exp(g_a_A0, g_a, eval_poly_A[0]);
  G2_exp(h_b_B0, h_b, eval_poly_B[0]);
  G1_exp(g_c_C0, g_c, eval_poly_C[0]);
 
  #if PUBLIC_VERIFIER == 1 
  G1_fixed_exp(g_a_Ai_io, g_a, eval_poly_A+1+size_f1_g_a_Ai_query, size_input+size_output);
  #endif

  mpz_mul(r_c_D, r_c, D);
  mpz_mod(r_c_D, r_c_D, prime);

  // dump the keys into files.
#if PROTOCOL == PINOCCHIO_ZK
  dump_vector_G1(1, &g_a_D, (char *)"g_a_D");
  dump_vector_G1(1, &g_b_D, (char *)"g_b_D");
  dump_vector_G2(1, &h_b_D, (char *)"h_b_D");
  dump_vector_G1(1, &g_c_D, (char *)"g_c_D");

  dump_vector_G1(1, &g_a_alpha_a_D, (char *)"g_a_alpha_a_D");
  dump_vector_G1(1, &g_b_alpha_b_D, (char *)"g_b_alpha_b_D");
  dump_vector_G1(1, &g_c_alpha_c_D, (char *)"g_c_alpha_c_D");
  dump_vector_G1(1, &g_a_beta_D, (char *)"g_a_beta_D");
  dump_vector_G1(1, &g_b_beta_D, (char *)"g_b_beta_D");
  dump_vector_G1(1, &g_c_beta_D, (char *)"g_c_beta_D");
#endif

  // dump VK, first copy elements into the array.
  G1_set(vk_G1[0], g_c_D);
  G1_set(vk_G1[1], g_a_A0);
  G1_set(vk_G1[2], g_c_C0);
  G1_set(vk_G1[3], g_alpha_b);
  G1_set(vk_G1[4], g_beta_gamma);
  
  #if PUBLIC_VERIFIER == 1
  for (int i = 0; i < size_input+size_output;i++) {
    G1_set(vk_G1[i+5], g_a_Ai_io[i]);
  }
  #endif

  G2_set(vk_G2[0], h);
  G2_set(vk_G2[1], h_alpha_a);
  G2_set(vk_G2[2], h_alpha_b);
  G2_set(vk_G2[3], h_alpha_c);
  G2_set(vk_G2[4], h_gamma);
  G2_set(vk_G2[5], h_beta_gamma);
  G2_set(vk_G2[6], h_b_B0);

  // dump the verification key in two separate pieces.
  #if PUBLIC_VERIFIER == 1
  dump_vector_G1(size_vk_G1, vk_G1, (char *)"f_verification_key_G1");
  #else
  dump_vector_G1(size_vk_G1-(size_input+size_output), vk_G1, (char *)"f_verification_key_G1");
  dump_vector_G1(1, &g_a, "f_verification_key_base");
  dump_vector(size_input+size_output, eval_poly_A+1+size_f1_g_a_Ai_query, (char *)"f_verification_key_IO");
  #endif
  dump_vector_G2(size_vk_G2, vk_G2, (char *)"f_verification_key_G2");
  dump_vector(1, &r_c_D, (char *)"f_verification_key_mpz_t");

  cout << "v_verification_key_size " << stat_size("f_verification_key_G1") + stat_size("f_verification_key_G2") + stat_size("f_verification_key_mpz_t") + stat_size("f_verification_key_IO")  << endl;

  // clear local variables.
#ifndef DEBUG_TEST_ENABLE
  clear_scalar_G1(g_a);
  clear_scalar_G1(g_b);
  clear_scalar_G1(g_c);
  clear_scalar_G2(h_b);
  clear_scalar_G1(g_a_alpha_a);
  clear_scalar_G1(g_b_alpha_b);
  clear_scalar_G1(g_c_alpha_c);
  clear_scalar_G1(g_a_beta);
  clear_scalar_G1(g_b_beta);
  clear_scalar_G1(g_c_beta);
  clear_scalar_G2(h_beta);
  clear_scalar(t);
  clear_scalar(D);
#endif
  clear_vec_G1(size_vk_G1, vk_G1);
  clear_vec_G2(size_vk_G2, vk_G2);

  clear_scalar(r_a);
  clear_scalar(r_b);
  clear_scalar(r_c);
  clear_scalar(alpha_a);
  clear_scalar(alpha_b);
  clear_scalar(alpha_c);
  clear_scalar(beta);
  clear_scalar(gamma);

  // send the queries
  send_file((char *)"f1_g_a_Ai_query");
  send_file((char *)"f1_g_b_Bi_query");
  send_file((char *)"f1_h_b_Bi_query");
  send_file((char *)"f1_g_c_Ci_query");
  send_file((char *)"f1_g_a_alpha_a_Ai_query");
  send_file((char *)"f1_g_b_alpha_b_Bi_query");
  send_file((char *)"f1_g_c_alpha_c_Ci_query");
  //send_file((char *)"f2_h_t_i_query");
  send_file((char *)"f2_g_t_i_query");
  send_file((char *)"f1_beta_query");

#if PROTOCOL == PINOCCHIO_ZK
  send_file((char *)"g_a_D");
  send_file((char *)"g_b_D");
  send_file((char *)"g_c_D");
  send_file((char *)"h_b_D");

  send_file((char *)"g_a_alpha_a_D");
  send_file((char *)"g_b_alpha_b_D");
  send_file((char *)"g_c_alpha_c_D");
  send_file((char *)"g_a_beta_D");
  send_file((char *)"g_b_beta_D");
  send_file((char *)"g_c_beta_D");
#endif
#endif
}
#endif

void ZComputationVerifier::create_plain_queries() {
#if NONINTERACTIVE == 0
  clear_vec(size_f1_vec*expansion_factor, f1_commitment);
  clear_vec(size_f2_vec*expansion_factor, f2_commitment);

  m_plainq.begin_with_init();
  // keeps track of #filled coins
  int f_con_filled = -1;
  int query_id;

  for (int rho=0; rho<num_repetitions; rho++) {
    if (rho == 0) m_plainq.begin_with_init();
    else m_plainq.begin_with_history();

    // create linearity test queries
    query_id = 1;
    for (int i=0; i<NUM_REPS_LIN; i++) {
      v->create_lin_test_queries(size_f1_vec, f1_q1, f1_q2, f1_q3, f1_consistency,
                                 f_con_filled, f_con_coins, prime);

      f_con_filled += 3;

      v->create_lin_test_queries(size_f2_vec, f2_q1, f2_q2, f2_q3, f2_consistency,
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
      
      // don't dump, but increment query_id
      query_id++;
      //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      //dump_vector(size_f1_vec, f1_q3, scratch_str);
      //send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q1, scratch_str);
      send_file(scratch_str);

      snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      dump_vector(size_f2_vec, f2_q2, scratch_str);
      send_file(scratch_str);
      
      query_id++;
      //snprintf(scratch_str, BUFLEN-1, "q%d_qquery_r_%d", query_id++, rho);
      //dump_vector(size_f2_vec, f2_q3, scratch_str);
      //send_file(scratch_str);
      */

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
    v->create_div_corr_test_queries(n, size_f1_vec, size_f2_vec, f1_q1, f1_q2, f1_q3, f2_q1, f_con_coins, f_con_filled, f1_consistency, f_con_coins, f_con_filled+3, f2_consistency, f1_q4, f2_q4, d_star[rho], num_aij, num_bij, num_cij, set_v, poly_A, poly_B, poly_C, eval_poly_A, eval_poly_B, eval_poly_C, prime);

    f_con_filled += 4;

    int base = rho * (1 + size_input + size_output);
    mpz_set(A_tau_io[base+0], eval_poly_A[0]);
    mpz_set(B_tau_io[base+0], eval_poly_B[0]);
    mpz_set(C_tau_io[base+0], eval_poly_C[0]);

    for (int i=0; i<size_input+size_output; i++) {
      mpz_set(A_tau_io[base+1+i], eval_poly_A[1+n_prime+i]);
      mpz_set(B_tau_io[base+1+i], eval_poly_B[1+n_prime+i]);
      mpz_set(C_tau_io[base+1+i], eval_poly_C[1+n_prime+i]);
    }

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
    */
  }

  dump_vector(size_f1_vec, f1_consistency, (char *)"f1_consistency_query");
  send_file((char *)"f1_consistency_query");

  dump_vector(size_f2_vec, f2_consistency, (char *)"f2_consistency_query");
  send_file((char *)"f2_consistency_query");
  m_plainq.end();

  // cleanup time
  clear_vec(n+1, eval_poly_A);
  clear_vec(n+1, eval_poly_B);
  clear_vec(n+1, eval_poly_C);
  clear_vec(size_f1_vec, f1_consistency);
  clear_vec(size_f2_vec, f2_consistency);

  for (int i=0; i<num_aij; i++)
    clear_scalar(poly_A[i].coefficient);

  for (int i=0; i<num_bij; i++)
    clear_scalar(poly_B[i].coefficient);

  for (int i=0; i<num_cij; i++)
    clear_scalar(poly_C[i].coefficient);

  free(poly_A);
  free(poly_B);
  free(poly_C);

  clear_vec(size_f1_vec, f1_q1);
  clear_vec(size_f1_vec, f1_q2);
  clear_vec(size_f1_vec, f1_q3);
  clear_vec(size_f1_vec, f1_q4);
  clear_vec(size_f2_vec, f2_q1);
  clear_vec(size_f2_vec, f2_q2);
  clear_vec(size_f2_vec, f2_q3);
  clear_vec(size_f2_vec, f2_q4);

  // Ginger's codebase did some run test part of work in plain query
  // creation; so the base class calls with history;
  m_runtests.reset();
#endif
}

void ZComputationVerifier::populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta) { }

#if NONINTERACTIVE == 1
void ZComputationVerifier::test_noninteractive_protocol(uint32_t beta) {
#ifdef DEBUG_TEST_ENABLE
#if GGPR == 1
  G1_t g_A, g_B, g_C, g_H;
  G2_t h_B;
  G1_t g_alpha_A, g_alpha_B, g_alpha_C, g_alpha_H;
  G1_t g_Beta;
  G1_t g_A_io;

  mpz_t *F1, *F2;

  alloc_init_scalar_G1(g_A);
  alloc_init_scalar_G1(g_B);
  alloc_init_scalar_G2(h_B);
  alloc_init_scalar_G1(g_C);
  alloc_init_scalar_G1(g_H);
  alloc_init_scalar_G1(g_alpha_A);
  alloc_init_scalar_G1(g_alpha_B);
  alloc_init_scalar_G1(g_alpha_C);
  alloc_init_scalar_G1(g_alpha_H);
  alloc_init_scalar_G1(g_Beta);
  alloc_init_scalar_G1(g_A_io);

  alloc_init_vec(&F1, size_f1_vec);
  alloc_init_vec(&F2, size_f2_vec);

  snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", beta);
  load_vector(size_f1_vec, F1, scratch_str);

  snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", beta);
  load_vector(size_f2_vec, F2, scratch_str);

  mpz_t expA, expB, expC, expH;
  mpz_t tmp;

  alloc_init_scalar(expA);
  alloc_init_scalar(expB);
  alloc_init_scalar(expC);
  alloc_init_scalar(expH);
  alloc_init_scalar(tmp);

  G1_t g1_debug_tmp1, g1_debug_tmp2, g1_debug_tmp3;
  G2_t g2_debug_tmp1, g2_debug_tmp2, g2_debug_tmp3;

  alloc_init_scalar_G1(g1_debug_tmp1);
  alloc_init_scalar_G1(g1_debug_tmp2);
  alloc_init_scalar_G1(g1_debug_tmp3);
  alloc_init_scalar_G2(g2_debug_tmp1);
  alloc_init_scalar_G2(g2_debug_tmp2);
  alloc_init_scalar_G2(g2_debug_tmp3);

  snprintf(scratch_str, BUFLEN-1, "f_answers_G1_b_%d", beta);
  load_vector_G1(size_answer_G1, f_ni_answers_G1, scratch_str);

  snprintf(scratch_str, BUFLEN-1, "f_answers_G2_b_%d", beta);
  load_vector_G2(size_answer_G2, f_ni_answers_G2, scratch_str);

  G1_set(g_A, f_ni_answers_G1[0]);
  G1_set(g_B, f_ni_answers_G1[1]);
  G1_set(g_C, f_ni_answers_G1[2]);
  G1_set(g_H, f_ni_answers_G1[3]);
  G1_set(g_alpha_A, f_ni_answers_G1[4]);
  G1_set(g_alpha_B, f_ni_answers_G1[5]);
  G1_set(g_alpha_C, f_ni_answers_G1[6]);
  G1_set(g_alpha_H, f_ni_answers_G1[7]);
  G1_set(g_Beta, f_ni_answers_G1[8]);

  G2_set(h_B, f_ni_answers_G2[0]);
  // input/output are properly loaded at this point.

  // compute values needed for test.
  multi_exponentiation_G1(size_input, g_Ai_io, input, g1_debug_tmp1);
  multi_exponentiation_G1(size_output, g_Ai_io + size_input, output, g_A_io);
  G1_mul(g_A_io, g1_debug_tmp1, g_A_io);

  // first compute the exponent.
  mpz_set_ui(expA, 0);
  for (int i = 0; i < size_f1_g_Ai_query; i++) {
    mpz_mul(tmp, eval_poly_A[1+i], F1[i]);
    mpz_add(expA, expA, tmp);
    mpz_mod(expA, expA, prime);
  }
  G1_exp(g1_debug_tmp1, g, expA);

  // raise base to first exponent and another, how the prover works.
  G1_set1(g1_debug_tmp2);
  for (int i = 0; i < size_f1_g_Ai_query; i++) {
    G1_exp(g1_debug_tmp3, g, eval_poly_A[1+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, F1[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);
  }

  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2))
    cout << "BUG: test failed " << endl;
  if (G1_cmp(g1_debug_tmp1, g_A))
    cout << "BUG: test failed" << endl;
  if (G1_cmp(g1_debug_tmp2, g_A))
    cout << "BUG: test failed" << endl;

  for (int i = 0; i < size_input; i++) {
    mpz_mul(tmp, eval_poly_A[1+size_f1_g_Ai_query+i], input[i]);
    mpz_add(expA, expA, tmp);
    mpz_mod(expA, expA, prime);
  }
  for (int i = 0; i < size_output; i++) {
    mpz_mul(tmp, eval_poly_A[1+size_f1_vec+size_input+i], output[i]);
    mpz_add(expA, expA, tmp);
    mpz_mod(expA, expA, prime);
  }
  mpz_add(expA, expA, eval_poly_A[0]);
  mpz_mod(expA, expA, prime);

  G1_exp(g1_debug_tmp1, g, expA);
  G1_mul(g1_debug_tmp2, g_A0, g_A_io);
  G1_mul(g1_debug_tmp2, g1_debug_tmp2, g_A);
  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2))
    cout << "BUG: test failed" << endl;

  mpz_set_ui(expB, 0);
  for (int i = 0; i < size_f1_vec; i++) {
    mpz_mul(tmp, eval_poly_B[1+i], F1[i]);
    mpz_add(expB, expB, tmp);
    mpz_mod(expB, expB, prime);
  }
  for (int i = 0; i < size_input; i++) {
    mpz_mul(tmp, eval_poly_B[1+size_f1_vec+i], input[i]);
    mpz_add(expB, expB, tmp);
    mpz_mod(expB, expB, prime);
  }
  for (int i = 0; i < size_output; i++) {
    mpz_mul(tmp, eval_poly_B[1+size_f1_vec+size_input+i], output[i]);
    mpz_add(expB, expB, tmp);
    mpz_mod(expB, expB, prime);
  }
  G1_exp(g1_debug_tmp1, g, expB);
  G2_exp(g2_debug_tmp1, h, expB);

  // raise base to first exponent and another, how the prover works.
  G1_set1(g1_debug_tmp2);
  for (int i = 0; i < size_f1_vec; i++) {
    G1_exp(g1_debug_tmp3, g, eval_poly_B[1+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, F1[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);

    G2_exp(g2_debug_tmp3, h, eval_poly_B[1+i]);
    G2_exp(g2_debug_tmp3, g2_debug_tmp3, F1[i]);
    G2_mul(g2_debug_tmp2, g2_debug_tmp2, g2_debug_tmp3);
  }

  for (int i = 0; i < size_input; i++) {
    G1_exp(g1_debug_tmp3, g, eval_poly_B[1+size_f1_vec+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, input[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);

    G2_exp(g2_debug_tmp3, h, eval_poly_B[1+size_f1_vec+i]);
    G2_exp(g2_debug_tmp3, g2_debug_tmp3, input[i]);
    G2_mul(g2_debug_tmp2, g2_debug_tmp2, g2_debug_tmp3);
  }
  for (int i = 0; i < size_output; i++) {
    G1_exp(g1_debug_tmp3, g, eval_poly_B[1+size_f1_vec+size_input+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, output[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);

    G2_exp(g2_debug_tmp3, h, eval_poly_B[1+size_f1_vec+size_input+i]);
    G2_exp(g2_debug_tmp3, g2_debug_tmp3, output[i]);
    G2_mul(g2_debug_tmp2, g2_debug_tmp2, g2_debug_tmp3);
  }

  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2))
    cout << "BUG: test failed" << endl;
  if (G2_cmp(g2_debug_tmp1, g2_debug_tmp2))
    cout << "BUG: test failed" << endl;
  if (G1_cmp(g1_debug_tmp1, g_B))
    cout << "BUG: test failed" << endl;

  if (G2_cmp(g2_debug_tmp1, h_B))
    cout << "BUG: test failed" << endl;
  if (G1_cmp(g1_debug_tmp2, g_B))
    cout << "BUG: test failed" << endl;
  if (G2_cmp(g2_debug_tmp2, h_B))
    cout << "BUG: test failed" << endl;

  mpz_add(expB, expB, eval_poly_B[0]);
  mpz_mod(expB, expB, prime);

  G2_exp(g2_debug_tmp1, h, expB);
  G2_mul(g2_debug_tmp2, h_B0, h_B);
  if(G2_cmp(g2_debug_tmp1, g2_debug_tmp2))
    cout << "BUG: test failed" << endl;

  mpz_set_ui(expC, 0);
  for (int i = 0; i < size_f1_vec; i++) {
    mpz_mul(tmp, eval_poly_C[1+i], F1[i]);
    mpz_add(expC, expC, tmp);
    mpz_mod(expC, expC, prime);
  }
  for (int i = 0; i < size_input; i++) {
    mpz_mul(tmp, eval_poly_C[1+size_f1_vec+i], input[i]);
    mpz_add(expC, expC, tmp);
    mpz_mod(expC, expC, prime);
  }
  for (int i = 0; i < size_output; i++) {
    mpz_mul(tmp, eval_poly_C[1+size_f1_vec+size_input+i], output[i]);
    mpz_add(expC, expC, tmp);
    mpz_mod(expC, expC, prime);
  }
  G1_exp(g1_debug_tmp1, g, expC);

  G1_set1(g1_debug_tmp2);
  for (int i = 0; i < size_f1_vec; i++) {
    G1_exp(g1_debug_tmp3, g, eval_poly_C[1+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, F1[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);
  }
  for (int i = 0; i < size_input; i++) {
    G1_exp(g1_debug_tmp3, g, eval_poly_C[1+size_f1_vec+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, input[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);
  }
  for (int i = 0; i < size_output; i++) {
    G1_exp(g1_debug_tmp3, g, eval_poly_C[1+size_f1_vec+size_input+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, output[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);
  }

  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2)) {
    cout << "BUG C(1)a************************************************" << endl;
  }
  if (G1_cmp(g1_debug_tmp1, g_C)) {
    cout << "BUG C(1)b************************************************" << endl;
  }
  if (G1_cmp(g1_debug_tmp2, g_C)) {
    cout << "BUG C(1)c************************************************" << endl;
  }

  mpz_add(expB, expB, eval_poly_B[0]);
  mpz_mod(expB, expB, prime);

  G1_exp(g1_debug_tmp1, g, expC);
  G1_mul(g1_debug_tmp2, g_C0, g_C);
  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2)) {
    cout << "BUG C************************************************" << endl;
  }

  mpz_set_ui(expH, 0);
  for (int i = 0; i < size_f2_g_t_i_query; i++) {
    mpz_powm_ui(tmp, t, i, prime);
    mpz_mul(tmp, tmp, F2[i]);
    mpz_add(expH, expH, tmp);
    mpz_mod(expH, expH, prime);
  }

  G1_exp(g1_debug_tmp1, g, expH);
  if (G1_cmp(g1_debug_tmp1, g_H)) {
    cout << "BUG H************************************************" << endl;
  }
  mpz_mul(tmp, expA, expB);
  mpz_mod(tmp, tmp, prime);
  mpz_add(tmp, tmp, expC);
  mpz_mod(tmp, tmp, prime);
  mpz_t tmp1;
  alloc_init_scalar(tmp1);
  mpz_mul(tmp1, D, expH);
  mpz_mod(tmp1, tmp1, prime);
  if (mpz_cmp(tmp, tmp1)) {
    cout << "BUG DIV************************************************" << endl;
  }

  // clear local variables
  clear_scalar_G1(g_A);
  clear_scalar_G1(g_B);
  clear_scalar_G2(h_B);
  clear_scalar_G1(g_C);
  clear_scalar_G1(g_H);

  clear_scalar_G1(g_alpha_A);
  clear_scalar_G1(g_alpha_B);
  clear_scalar_G1(g_alpha_C);
  clear_scalar_G1(g_alpha_H);

  clear_scalar_G1(g_Beta);
  clear_scalar_G1(g_A_io);

  clear_vec(size_f1_vec, F1);
  clear_vec(size_f2_vec, F2);

  clear_scalar(expA);
  clear_scalar(expB);
  clear_scalar(expC);
  clear_scalar(expH);
  clear_scalar(tmp);
  clear_scalar(tmp1);

  clear_scalar_G1(g1_debug_tmp1);
  clear_scalar_G1(g1_debug_tmp2);
  clear_scalar_G1(g1_debug_tmp3);
  clear_scalar_G2(g2_debug_tmp1);
  clear_scalar_G2(g2_debug_tmp2);
  clear_scalar_G2(g2_debug_tmp3);
#else
  G1_t g_a_A, g_b_B, g_c_C;
  G2_t h_b_B;
  G1_t g_a_alpha_a_A, g_b_alpha_b_B, g_c_alpha_c_C;
  G1_t g_Beta;
  G1_t g_a_A_io;

  mpz_t *F1, *F2;

  alloc_init_scalar_G1(g_a_A);
  alloc_init_scalar_G1(g_b_B);
  alloc_init_scalar_G2(h_b_B);
  alloc_init_scalar_G1(g_c_C);
  alloc_init_scalar_G1(g_a_alpha_a_A);
  alloc_init_scalar_G1(g_b_alpha_b_B);
  alloc_init_scalar_G1(g_c_alpha_c_C);
  alloc_init_scalar_G1(g_Beta);
  alloc_init_scalar_G1(g_a_A_io);

  alloc_init_vec(&F1, size_f1_vec);
  alloc_init_vec(&F2, size_f2_vec);

  snprintf(scratch_str, BUFLEN-1, "f1_assignment_vector_b_%d", beta);
  load_vector(size_f1_vec, F1, scratch_str);

  snprintf(scratch_str, BUFLEN-1, "f2_assignment_vector_b_%d", beta);
  load_vector(size_f2_vec, F2, scratch_str);

  mpz_t expA, expB, expC, expH;
  mpz_t tmp;

  alloc_init_scalar(expA);
  alloc_init_scalar(expB);
  alloc_init_scalar(expC);
  alloc_init_scalar(expH);
  alloc_init_scalar(tmp);

  G1_t g1_debug_tmp1, g1_debug_tmp2, g1_debug_tmp3;
  G2_t g2_debug_tmp1, g2_debug_tmp2, g2_debug_tmp3;

  alloc_init_scalar_G1(g1_debug_tmp1);
  alloc_init_scalar_G1(g1_debug_tmp2);
  alloc_init_scalar_G1(g1_debug_tmp3);
  alloc_init_scalar_G2(g2_debug_tmp1);
  alloc_init_scalar_G2(g2_debug_tmp2);
  alloc_init_scalar_G2(g2_debug_tmp3);

  snprintf(scratch_str, BUFLEN-1, "f_answers_G1_b_%d", beta);
  load_vector_G1(size_answer_G1, f_ni_answers_G1, scratch_str);

  snprintf(scratch_str, BUFLEN-1, "f_answers_G2_b_%d", beta);
  load_vector_G2(size_answer_G2, f_ni_answers_G2, scratch_str);

  G1_set(g_a_A, f_ni_answers_G1[0]);
  G1_set(g_b_B, f_ni_answers_G1[1]);
  G1_set(g_c_C, f_ni_answers_G1[2]);
  G1_set(g_a_alpha_a_A, f_ni_answers_G1[3]);
  G1_set(g_b_alpha_b_B, f_ni_answers_G1[4]);
  G1_set(g_c_alpha_c_C, f_ni_answers_G1[5]);
  G1_set(g_Beta, f_ni_answers_G1[6]);

  G2_set(h_b_B, f_ni_answers_G2[0]);
  // input/output are properly loaded at this point.

  // compute values needed for test.
  G1_set1(g_a_A_io);
  for (int i = 0; i < size_input; i++) {
    G1_exp(g1_debug_tmp1, g_a_Ai_io[i], input[i]);
    G1_mul(g_a_A_io, g_a_A_io, g1_debug_tmp1);
  }
  for (int i = 0; i < size_output; i++) {
    G1_exp(g1_debug_tmp1, g_a_Ai_io[size_input+i], output[i]);
    G1_mul(g_a_A_io, g_a_A_io, g1_debug_tmp1);
  }

  // first compute the exponent.
  mpz_set_ui(expA, 0);
  for (int i = 0; i < size_f1_g_a_Ai_query; i++) {
    mpz_mul(tmp, eval_poly_A[1+i], F1[i]);
    mpz_add(expA, expA, tmp);
    mpz_mod(expA, expA, prime);
  }
  G1_exp(g1_debug_tmp1, g_a, expA);

  // raise base to first exponent and another, how the prover works.
  G1_set1(g1_debug_tmp2);
  for (int i = 0; i < size_f1_g_a_Ai_query; i++) {
    G1_exp(g1_debug_tmp3, g_a, eval_poly_A[1+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, F1[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);
  }

  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2)) {
    cout << "BUG A(1)a************************************************" << endl;
  }
  if (G1_cmp(g1_debug_tmp1, g_a_A)) {
    cout << "BUG A(1)b************************************************" << endl;
  }
  if (G1_cmp(g1_debug_tmp2, g_a_A)) {
    cout << "BUG A(1)c************************************************" << endl;
  }

  for (int i = 0; i < size_input; i++) {
    mpz_mul(tmp, eval_poly_A[1+size_f1_g_a_Ai_query+i], input[i]);
    mpz_add(expA, expA, tmp);
    mpz_mod(expA, expA, prime);
  }
  for (int i = 0; i < size_output; i++) {
    mpz_mul(tmp, eval_poly_A[1+size_f1_vec+size_input+i], output[i]);
    mpz_add(expA, expA, tmp);
    mpz_mod(expA, expA, prime);
  }
  mpz_add(expA, expA, eval_poly_A[0]);
  mpz_mod(expA, expA, prime);

  G1_exp(g1_debug_tmp1, g_a, expA);
  G1_mul(g1_debug_tmp2, g_a_A0, g_a_A_io);
  G1_mul(g1_debug_tmp2, g1_debug_tmp2, g_a_A);
  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2)) {
    cout << "BUG A************************************************" << endl;
  }


  mpz_set_ui(expB, 0);

  for (int i = 0; i < size_f1_vec; i++) {
    mpz_mul(tmp, eval_poly_B[1+i], F1[i]);
    mpz_add(expB, expB, tmp);
    mpz_mod(expB, expB, prime);
  }
  for (int i = 0; i < size_input; i++) {
    mpz_mul(tmp, eval_poly_B[1+size_f1_vec+i], input[i]);
    mpz_add(expB, expB, tmp);
    mpz_mod(expB, expB, prime);
  }
  for (int i = 0; i < size_output; i++) {
    mpz_mul(tmp, eval_poly_B[1+size_f1_vec+size_input+i], output[i]);
    mpz_add(expB, expB, tmp);
    mpz_mod(expB, expB, prime);
  }

  G1_exp(g1_debug_tmp1, g_b, expB);
  G2_exp(g2_debug_tmp1, h_b, expB);

  // raise base to first exponent and another, how the prover works.
  G1_set1(g1_debug_tmp2);
  G2_set1(g2_debug_tmp2);

  for (int i = 0; i < size_f1_vec; i++) {
    G1_exp(g1_debug_tmp3, g_b, eval_poly_B[1+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, F1[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);

    G2_exp(g2_debug_tmp3, h_b, eval_poly_B[1+i]);
    G2_exp(g2_debug_tmp3, g2_debug_tmp3, F1[i]);
    G2_mul(g2_debug_tmp2, g2_debug_tmp2, g2_debug_tmp3);
  }
  for (int i = 0; i < size_input; i++) {
    G1_exp(g1_debug_tmp3, g_b, eval_poly_B[1+size_f1_vec+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, input[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);

    G2_exp(g2_debug_tmp3, h_b, eval_poly_B[1+size_f1_vec+i]);
    G2_exp(g2_debug_tmp3, g2_debug_tmp3, input[i]);
    G2_mul(g2_debug_tmp2, g2_debug_tmp2, g2_debug_tmp3);
  }
  for (int i = 0; i < size_output; i++) {
    G1_exp(g1_debug_tmp3, g_b, eval_poly_B[1+size_f1_vec+size_input+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, output[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);

    G2_exp(g2_debug_tmp3, h_b, eval_poly_B[1+size_f1_vec+size_input+i]);
    G2_exp(g2_debug_tmp3, g2_debug_tmp3, output[i]);
    G2_mul(g2_debug_tmp2, g2_debug_tmp2, g2_debug_tmp3);
  }

  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2)) {
    cout << "BUG B(1)a************************************************" << endl;
  }
  if (G2_cmp(g2_debug_tmp1, g2_debug_tmp2)) {
    cout << "BUG B(2)a************************************************" << endl;
  }
  if (G1_cmp(g1_debug_tmp1, g_b_B)) {
    cout << "BUG B(1)b************************************************" << endl;
  }
  if (G2_cmp(g2_debug_tmp1, h_b_B)) {
    cout << "BUG B(2)b************************************************" << endl;
  }
  if (G1_cmp(g1_debug_tmp2, g_b_B)) {
    cout << "BUG B(1)c************************************************" << endl;
  }
  if (G2_cmp(g2_debug_tmp2, h_b_B)) {
    cout << "BUG B(2)c************************************************" << endl;
  }

  mpz_add(expB, expB, eval_poly_B[0]);
  mpz_mod(expB, expB, prime);

  G2_exp(g2_debug_tmp1, h_b, expB);
  G2_mul(g2_debug_tmp2, h_b_B0, h_b_B);
  if(G2_cmp(g2_debug_tmp1, g2_debug_tmp2)){
    cout << "BUG B************************************************" << endl;
  }

  mpz_set_ui(expC, 0);
  for (int i = 0; i < size_f1_vec; i++) {
    mpz_mul(tmp, eval_poly_C[1+i], F1[i]);
    mpz_add(expC, expC, tmp);
    mpz_mod(expC, expC, prime);
  }
  for (int i = 0; i < size_input; i++) {
    mpz_mul(tmp, eval_poly_C[1+size_f1_vec+i], input[i]);
    mpz_add(expC, expC, tmp);
    mpz_mod(expC, expC, prime);
  }
  for (int i = 0; i < size_output; i++) {
    mpz_mul(tmp, eval_poly_C[1+size_f1_vec+size_input+i], output[i]);
    mpz_add(expC, expC, tmp);
    mpz_mod(expC, expC, prime);
  }
  G1_exp(g1_debug_tmp1, g_c, expC);

  G1_set1(g1_debug_tmp2);
  for (int i = 0; i < size_f1_vec; i++) {
    G1_exp(g1_debug_tmp3, g_c, eval_poly_C[1+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, F1[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);
  }
  for (int i = 0; i < size_input; i++) {
    G1_exp(g1_debug_tmp3, g_c, eval_poly_C[1+size_f1_vec+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, input[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);
  }
  for (int i = 0; i < size_output; i++) {
    G1_exp(g1_debug_tmp3, g_c, eval_poly_C[1+size_f1_vec+size_input+i]);
    G1_exp(g1_debug_tmp3, g1_debug_tmp3, output[i]);
    G1_mul(g1_debug_tmp2, g1_debug_tmp2, g1_debug_tmp3);
  }

  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2)) {
    cout << "BUG C(1)a************************************************" << endl;
  }
  if (G1_cmp(g1_debug_tmp1, g_c_C)) {
    cout << "BUG C(1)b************************************************" << endl;
  }
  if (G1_cmp(g1_debug_tmp2, g_c_C)) {
    cout << "BUG C(1)c************************************************" << endl;
  }

  mpz_add(expC, expC, eval_poly_C[0]);
  mpz_mod(expC, expC, prime);

  G1_exp(g1_debug_tmp1, g_c, expC);
  G1_mul(g1_debug_tmp2, g_c_C0, g_c_C);
  if (G1_cmp(g1_debug_tmp1, g1_debug_tmp2)) {
    cout << "BUG C************************************************" << endl;
  }

  mpz_set_ui(expH, 0);
  for (int i = 0; i < size_f2_g_t_i_query; i++) {
    mpz_powm_ui(tmp, t, i, prime);
    mpz_mul(tmp, tmp, F2[i]);
    mpz_add(expH, expH, tmp);
    mpz_mod(expH, expH, prime);
  }

  /*
  G2_exp(g2_debug_tmp1, h, expH);
  if (G2_cmp(g2_debug_tmp1, h_H)) {
    cout << "BUG H************************************************" << endl;
  }
  */
  mpz_mul(tmp, expA, expB);
  mpz_mod(tmp, tmp, prime);
  mpz_add(tmp, tmp, expC);
  mpz_mod(tmp, tmp, prime);
  mpz_t tmp1;
  alloc_init_scalar(tmp1);
  mpz_mul(tmp1, D, expH);
  mpz_mod(tmp1, tmp1, prime);
  if (mpz_cmp(tmp, tmp1)) {
    cout << "BUG DIV************************************************" << endl;
  }

  // clear local variables
  clear_scalar_G1(g_a_A);
  clear_scalar_G1(g_b_B);
  clear_scalar_G2(h_b_B);
  clear_scalar_G1(g_c_C);

  clear_scalar_G1(g_a_alpha_a_A);
  clear_scalar_G1(g_b_alpha_b_B);
  clear_scalar_G1(g_c_alpha_c_C);
  //clear_scalar_G2(h_H);
  clear_scalar_G1(g_Beta);
  clear_scalar_G1(g_a_A_io);

  clear_vec(size_f1_vec, F1);
  clear_vec(size_f2_vec, F2);

  clear_scalar(expA);
  clear_scalar(expB);
  clear_scalar(expC);
  clear_scalar(expH);
  clear_scalar(tmp);
  clear_scalar(tmp1);

  clear_scalar_G1(g1_debug_tmp1);
  clear_scalar_G1(g1_debug_tmp2);
  clear_scalar_G1(g1_debug_tmp3);
  clear_scalar_G2(g2_debug_tmp1);
  clear_scalar_G2(g2_debug_tmp2);
  clear_scalar_G2(g2_debug_tmp3);
#endif
#endif
}

void ZComputationVerifier::prepare_noninteractive_answers(uint32_t beta) {
  // refactor this to some upper layer because it remains the same for the batch.
  if (beta == 0) {
    G1_t *vk_G1;
    G2_t *vk_G2;
    alloc_init_vec_G1(&vk_G1, size_vk_G1);
    alloc_init_vec_G2(&vk_G2, size_vk_G2);
    
#if GGPR == 1
    load_vector_G1(size_vk_G1, vk_G1, (char *)"f_verification_key_G1");
    load_vector_G2(size_vk_G2, vk_G2, (char *)"f_verification_key_G2");
    
    G1_set(g_A0, vk_G1[0]);
    G1_set(g_C0, vk_G1[1]);
    for (int i = 0; i < size_input+size_output;i++) {
      G1_set(g_Ai_io[i], vk_G1[i+2]);
    }

    G2_set(h, vk_G2[0]);
    G2_set(h_alpha, vk_G2[1]);
    G2_set(h_gamma, vk_G2[2]);
    G2_set(h_beta_a_gamma, vk_G2[3]);
    G2_set(h_beta_b_gamma, vk_G2[4]);
    G2_set(h_beta_c_gamma, vk_G2[5]);
    G2_set(h_D, vk_G2[6]);
    G2_set(h_B0, vk_G2[7]);
#else
    #if PUBLIC_VERIFIER == 1
    load_vector_G1(size_vk_G1, vk_G1, (char *)"f_verification_key_G1");
    #else
    load_vector_G1(size_vk_G1-(size_input+size_output), vk_G1, (char *)"f_verification_key_G1");
    #endif

    load_vector_G2(size_vk_G2, vk_G2, (char *)"f_verification_key_G2");
    
    G1_set(g_c_D, vk_G1[0]);
    G1_set(g_a_A0, vk_G1[1]);
    G1_set(g_c_C0, vk_G1[2]);
    G1_set(g_alpha_b, vk_G1[3]);
    G1_set(g_beta_gamma, vk_G1[4]);
    
    #if PUBLIC_VERIFIER == 1
    for (int i = 0; i < size_input+size_output;i++) {
      G1_set(g_a_Ai_io[i], vk_G1[i+5]);
    }
    #else
    load_vector(size_input+size_output, Ai_io, (char*)"f_verification_key_IO");
    load_vector_G1(1, &g_a_base, (char*)"f_verification_key_base");
    #endif

    G2_set(h, vk_G2[0]);
    G2_set(h_alpha_a, vk_G2[1]);
    G2_set(h_alpha_b, vk_G2[2]);
    G2_set(h_alpha_c, vk_G2[3]);
    G2_set(h_gamma, vk_G2[4]);
    G2_set(h_beta_gamma, vk_G2[5]);
    G2_set(h_b_B0, vk_G2[6]);
    load_vector(1, &r_c_D, (char *)"f_verification_key_mpz_t");
    
#endif

    // clear vk_G1 and vk_G2
    clear_vec_G1(size_vk_G1, vk_G1);
    clear_vec_G2(size_vk_G2, vk_G2);
  }

  // read input/output and answers
  snprintf(scratch_str, BUFLEN-1, "input_b_%d", beta);
  load_vector(size_input, input, scratch_str);

  snprintf(scratch_str, BUFLEN-1, "output_b_%d", beta);
  load_vector(size_output, output, scratch_str);

  snprintf(scratch_str, BUFLEN-1, "f_answers_G1_b_%d", beta);
  load_vector_G1(size_answer_G1, f_ni_answers_G1, scratch_str);

  snprintf(scratch_str, BUFLEN-1, "f_answers_G2_b_%d", beta);
  load_vector_G2(size_answer_G2, f_ni_answers_G2, scratch_str);
}

#if GGPR == 1
bool ZComputationVerifier::run_noninteractive_GGPR_tests(uint32_t beta) {
  G1_t g_A, g_B, g_C, g_H;
  G2_t h_B;
  G1_t g_alpha_A, g_alpha_B, g_alpha_C, g_alpha_H;
  G1_t g_Beta;
  G1_t g_A_io;

  G1_t g1_tmp1;
  GT_t gt_tmp1;
  G1_t in1;
  G2_t in2;
  GT_t lhs, rhs;

  bool result = true;

  // init local variables.
  alloc_init_scalar_G1(g_A);
  alloc_init_scalar_G1(g_B);
  alloc_init_scalar_G2(h_B);
  alloc_init_scalar_G1(g_C);
  alloc_init_scalar_G1(g_H);
  alloc_init_scalar_G1(g_alpha_A);
  alloc_init_scalar_G1(g_alpha_B);
  alloc_init_scalar_G1(g_alpha_C);
  alloc_init_scalar_G1(g_alpha_H);
  alloc_init_scalar_G1(g_Beta);
  alloc_init_scalar_G1(g_A_io);

  alloc_init_scalar_G1(g1_tmp1);
  alloc_init_scalar_GT(gt_tmp1);

  alloc_init_scalar_G1(in1);
  alloc_init_scalar_G2(in2);

  alloc_init_scalar_GT(lhs);
  alloc_init_scalar_GT(rhs);

  G1_set(g_A, f_ni_answers_G1[0]);
  G1_set(g_B, f_ni_answers_G1[1]);
  G1_set(g_C, f_ni_answers_G1[2]);
  G1_set(g_H, f_ni_answers_G1[3]);
  G1_set(g_alpha_A, f_ni_answers_G1[4]);
  G1_set(g_alpha_B, f_ni_answers_G1[5]);
  G1_set(g_alpha_C, f_ni_answers_G1[6]);
  G1_set(g_alpha_H, f_ni_answers_G1[7]);
  G1_set(g_Beta, f_ni_answers_G1[8]);

  G2_set(h_B, f_ni_answers_G2[0]);

  multi_exponentiation_G1(size_input, g_Ai_io, input, g1_tmp1);
  multi_exponentiation_G1(size_output, g_Ai_io + size_input, output, g_A_io);
  G1_mul(g_A_io, g1_tmp1, g_A_io);

  // divisibility test
  // g^A0*g^A_io*g^A
  G1_mul(in1, g_A0, g_A_io);
  G1_mul(in1, in1, g_A);
  // h^B0*g^B
  G2_mul(in2, h_B0, h_B);
  do_pairing(lhs, in1, in2);

  // g^C0*g^C
  GT_mul(in1, g_C0, g_C);
  do_pairing(gt_tmp1, in1, h);
  GT_mul(lhs, lhs, gt_tmp1);

  do_pairing(rhs, g_H, h_D);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the divisibility test" << endl;
#endif
  }
  // linearity test
  do_pairing(lhs, g_alpha_A, h);
  do_pairing(rhs, g_A, h_alpha);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the linearity test for A" << endl;
#endif
  }
  do_pairing(lhs, g_alpha_B, h);
  do_pairing(rhs, g_B, h_alpha);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the linearity test for B" << endl;
#endif
  }
  do_pairing(lhs, g_alpha_C, h);
  do_pairing(rhs, g_C, h_alpha);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the linearity test for C" << endl;
#endif
  }
  do_pairing(lhs, g_alpha_H, h);
  do_pairing(rhs, g_H, h_alpha);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the linearity test for H" << endl;
#endif
  }
  // consistency test
  do_pairing(lhs, g_Beta, h_gamma);
  G1_mul(g1_tmp1, g_A_io, g_A);
  do_pairing(rhs, g1_tmp1, h_beta_a_gamma);
  do_pairing(gt_tmp1, g_B, h_beta_b_gamma);
  GT_mul(rhs, rhs, gt_tmp1);
  do_pairing(gt_tmp1, g_C, h_beta_c_gamma);
  GT_mul(rhs, rhs, gt_tmp1);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the consistency test" << endl;
#endif
  }

#if VERBOSE == 1
  if (false == result) {
    cout << "LOG: F failed the test" << endl;
  }
  else
    cout << "LOG: F passed the test" << endl;
#endif
  test_noninteractive_protocol(beta);

  // clear local variables.
  clear_scalar_G1(g_A);
  clear_scalar_G1(g_B);
  clear_scalar_G2(h_B);
  clear_scalar_G1(g_C);
  clear_scalar_G1(g_H);

  clear_scalar_G1(g_alpha_A);
  clear_scalar_G1(g_alpha_B);
  clear_scalar_G1(g_alpha_C);
  clear_scalar_G1(g_alpha_H);

  clear_scalar_G1(g_Beta);

  clear_scalar_G1(g1_tmp1);
  clear_scalar_GT(gt_tmp1);

  clear_scalar_G1(g_A_io);
  clear_scalar_G1(in1);
  clear_scalar_G2(in2);
  clear_scalar_GT(lhs);
  clear_scalar_GT(rhs);

  return result;
}
#endif

bool ZComputationVerifier::run_noninteractive_tests(uint32_t beta) {
#if GGPR == 1
  return run_noninteractive_GGPR_tests(beta);
#else
  G1_t g_a_A, /*g_b_B,*/ g_c_C;
  G2_t h_b_B;
  G1_t g_a_alpha_a_A, g_b_alpha_b_B, g_c_alpha_c_C;
  //G2_t h_H;
  G1_t g_H;
  G1_t g_Beta;
  G1_t g_a_A_io;

  G1_t g1_tmp1;
  GT_t gt_tmp1;
  G1_t in1;
  G2_t in2;
  GT_t lhs, rhs;

  bool result = true;

  // init local variables.
  alloc_init_scalar_G1(g_a_A);
//  alloc_init_scalar_G1(g_b_B);
  alloc_init_scalar_G2(h_b_B);
  alloc_init_scalar_G1(g_c_C);
  alloc_init_scalar_G1(g_a_alpha_a_A);
  alloc_init_scalar_G1(g_b_alpha_b_B);
  alloc_init_scalar_G1(g_c_alpha_c_C);
  //alloc_init_scalar_G2(h_H);
  alloc_init_scalar_G1(g_H);
  alloc_init_scalar_G1(g_Beta);
  alloc_init_scalar_G1(g_a_A_io);

  alloc_init_scalar_G1(g1_tmp1);
  alloc_init_scalar_GT(gt_tmp1);

  alloc_init_scalar_G1(in1);
  alloc_init_scalar_G2(in2);

  alloc_init_scalar_GT(lhs);
  alloc_init_scalar_GT(rhs);

  G1_set(g_a_A, f_ni_answers_G1[0]);
  //G1_set(g_b_B, f_ni_answers_G1[1]);
  G1_set(g_H, f_ni_answers_G1[1]);
  G1_set(g_c_C, f_ni_answers_G1[2]);
  G1_set(g_a_alpha_a_A, f_ni_answers_G1[3]);
  G1_set(g_b_alpha_b_B, f_ni_answers_G1[4]);
  G1_set(g_c_alpha_c_C, f_ni_answers_G1[5]);
  G1_set(g_Beta, f_ni_answers_G1[6]);
  //G1_set(g_H, f_ni_answers_G1[7]);

  G2_set(h_b_B, f_ni_answers_G2[0]);
  //G2_set(h_H, f_ni_answers_G2[1]);

  #if PUBLIC_VERIFIER == 1
  multi_exponentiation_G1(size_input, g_a_Ai_io, input, g1_tmp1);
  multi_exponentiation_G1(size_output, g_a_Ai_io + size_input, output, g_a_A_io);
  G1_mul(g_a_A_io, g1_tmp1, g_a_A_io);
  #else
  mpz_t dotp, temp;
  alloc_init_scalar(dotp);
  alloc_init_scalar(temp);
  mpz_set_ui(dotp, 0);
  int i;
  for (i=0; i<size_input; i++) {
    mpz_mul(temp, Ai_io[i], input[i]);
    mpz_add(dotp, dotp, temp);
    mpz_mod(dotp, dotp, prime);
  }

  for (; i<size_input+size_output; i++) {
    mpz_mul(temp, Ai_io[i], output[i-size_input]);
    mpz_add(dotp, dotp, temp);
    mpz_mod(dotp, dotp, prime);
  }
  G1_exp(g_a_A_io, g_a_base, dotp); 
  clear_scalar(dotp);
  clear_scalar(temp);
  #endif
  
  // divisibility test
  // g_a^A0*g_a^A_io*g_a^A
  G1_mul(in1, g_a_A0, g_a_A_io);
  G1_mul(in1, in1, g_a_A);
  // g_b^B0*g_b^B
  G2_mul(in2, h_b_B0, h_b_B);
  do_pairing(lhs, in1, in2);

  // g_c^C0*g_c^C
  G1_mul(in1, g_c_C0, g_c_C);
  do_pairing(gt_tmp1, in1, h);
  GT_mul(lhs, lhs, gt_tmp1);

  //do_pairing(rhs, g_c_D, h_H);
  //G2_exp(in2, h_H, r_c_D);
  //do_pairing(rhs, g, in2);
  G1_exp(in1, g_H, r_c_D);
  do_pairing(rhs, in1, h);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the divisibility test" << endl;
#endif
  }
  // linearity test
  do_pairing(lhs, g_a_alpha_a_A, h);
  do_pairing(rhs, g_a_A, h_alpha_a);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the linearity test for A" << endl;
#endif
  }
  do_pairing(lhs, g_b_alpha_b_B, h);
  //do_pairing(rhs, g_b_B, h_alpha_b);
  do_pairing(rhs, g_alpha_b, h_b_B);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the linearity test for B" << endl;
#endif
  }
  do_pairing(lhs, g_c_alpha_c_C, h);
  do_pairing(rhs, g_c_C, h_alpha_c);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the linearity test for C" << endl;
#endif
  }
  // consistency test
  do_pairing(lhs, g_Beta, h_gamma);
  G1_mul(g1_tmp1, g_a_A_io, g_a_A);
  //G1_mul(g1_tmp1, g1_tmp1, g_b_B);
  G1_mul(g1_tmp1, g1_tmp1, g_c_C);
  do_pairing(gt_tmp1, g1_tmp1, h_beta_gamma);
  //gt_tmp1 holds beta gamma (A + C)
  do_pairing(rhs, g_beta_gamma, h_b_B);

  GT_mul(rhs, rhs, gt_tmp1);
  //element_printf("lhs = %B\n", lhs);
  //element_printf("rhs = %B\n", rhs);
  if (GT_cmp(lhs, rhs) != 0) {
    result = false;
#if VERBOSE == 1
    cout << "LOG: F failed the consistency test" << endl;
#endif
  }

#if VERBOSE == 1
  if (false == result) {
    cout << "LOG: F failed the test" << endl;
  }
  else
    cout << "LOG: F passed the test" << endl;
#endif
  test_noninteractive_protocol(beta);

  // clear local variables.
  clear_scalar_G1(g_a_A);
//  clear_scalar_G1(g_b_B);
  clear_scalar_G2(h_b_B);
  clear_scalar_G1(g_c_C);

  clear_scalar_G1(g_a_alpha_a_A);
  clear_scalar_G1(g_b_alpha_b_B);
  clear_scalar_G1(g_c_alpha_c_C);
  clear_scalar_G1(g_H);
  clear_scalar_G1(g_Beta);

  clear_scalar_G1(g1_tmp1);
  clear_scalar_GT(gt_tmp1);

  clear_scalar_G1(g_a_A_io);
  clear_scalar_G1(in1);
  clear_scalar_G2(in2);
  clear_scalar_GT(lhs);
  clear_scalar_GT(rhs);

  return result;
#endif
}
#else
bool ZComputationVerifier::run_interactive_tests(uint32_t beta) {
  bool lin1, lin2, corr;
  bool result = true;

  for (int rho=0; rho<num_repetitions; rho++) {
    // linearity test
    for (int i=0; i<NUM_REPS_LIN; i++) {
      int base = i*NUM_LIN_QUERIES;
      lin1 = v->lin_test(f_answers[rho*num_lin_pcp_queries + base +
QUERY1],
                         f_answers[rho*num_lin_pcp_queries + base + QUERY2],
                         f_answers[rho*num_lin_pcp_queries + base + QUERY3],
                         prime);

      lin2 = v->lin_test(f_answers[rho*num_lin_pcp_queries + base + QUERY4],
                         f_answers[rho*num_lin_pcp_queries + base + QUERY5],
                         f_answers[rho*num_lin_pcp_queries + base + QUERY6],
                         prime);

#if VERBOSE == 1
      if (false == lin1 || false == lin2)
        cout<<"LOG: F1, F2 failed the linearity test"<<endl;
      else
        cout<<"LOG: F1, F2 passed the linearity test"<<endl;
#endif

      result = result & lin1 & lin2;
    }

    // divisibilty correction test
    mpz_set(lhs, f_answers[rho*num_lin_pcp_queries + QUERY10]);
    mpz_sub(lhs, lhs, f_answers[rho*num_lin_pcp_queries + QUERY4]);

    mpz_mul(lhs, lhs, d_star[rho]);
    mpz_mod(lhs, lhs, prime);

    int base = rho * (1 + size_input + size_output);
    mpz_set(A_tau, A_tau_io[base+0]);
    mpz_set(B_tau, B_tau_io[base+0]);
    mpz_set(C_tau, C_tau_io[base+0]);

    // note: input and output are contiguous; so we can do this in one
    // loop
    for (int i=0; i<size_input+size_output; i++) {
      mpz_mul(rhs, input[i], A_tau_io[base+1+i]);
      mpz_add(A_tau, A_tau, rhs);

      mpz_mul(rhs, input[i], B_tau_io[base+1+i]);
      mpz_add(B_tau, B_tau, rhs);

      mpz_mul(rhs, input[i], C_tau_io[base+1+i]);
      mpz_add(C_tau, C_tau, rhs);
    }


    mpz_sub(rhs, f_answers[rho*num_lin_pcp_queries + QUERY7], f_answers[rho*num_lin_pcp_queries + QUERY1]);
    mpz_add(A_tau, A_tau, rhs);

    mpz_sub(rhs, f_answers[rho*num_lin_pcp_queries + QUERY8], f_answers[rho*num_lin_pcp_queries + QUERY1]);
    mpz_add(B_tau, B_tau, rhs);

    mpz_sub(rhs, f_answers[rho*num_lin_pcp_queries + QUERY9], f_answers[rho*num_lin_pcp_queries + QUERY1]);
    mpz_add(C_tau, C_tau, rhs);

    mpz_mul(rhs, A_tau, B_tau);
    mpz_add(rhs, rhs, C_tau);
    mpz_mod(rhs, rhs, prime);
    corr = mpz_cmp(lhs, rhs);
#if VERBOSE == 1
    if (0 == corr)
      cout <<"LOG: F1, F2 passed the divisibility correction test"<<endl;
    else
      cout <<"LOG: F1, F2 failed the divisibility correction test"<<endl;
#endif
    if (0 == corr)
      result = result & true;
    else
      result = result & false;
  }
  return result;
}
#endif

bool ZComputationVerifier::run_correction_and_circuit_tests(uint32_t beta) {
#if NONINTERACTIVE == 1
  return run_noninteractive_tests(beta);
#else
  return run_interactive_tests(beta);
#endif
}
