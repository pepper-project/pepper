#ifndef CODE_PEPPER_APPS_ZCOMPUTATION_V_H_
#define CODE_PEPPER_APPS_ZCOMPUTATION_V_H_

#include <libv/verifier.h>
#include <libv/input_creator.h>
#include <libv/zpcp.h>
#include <common/utility.h>
#if NONINTERACTIVE == 1
#include <common/pairing_util.h>
#endif

#ifdef INTERFACE_MPI
#include <libv/zcomputation_p.h>
#endif

class ZComputationVerifier : public Verifier {
  protected:
    int chi, n;
    int n_prime;

    mpz_t omega;
    poly_compressed *poly_A, *poly_B, *poly_C;
    mpz_t *eval_poly_A, *eval_poly_B, *eval_poly_C;
    int num_aij, num_bij, num_cij;
    mpz_t A_tau, B_tau, C_tau;

    mpz_t *d_star, *A_tau_io, *B_tau_io, *C_tau_io;
    mpz_t *set_v;
#if NONINTERACTIVE == 0
    mpz_t *f1_q1, *f1_q2, *f1_q3, *f1_q4, *f1_commitment, *f1_consistency;
    mpz_t *f2_q1, *f2_q2, *f2_q3, *f2_q4, *f2_commitment, *f2_consistency;
    mpz_t temp, temp2, temp3, lhs, rhs;
    mpz_t *temp_arr, *temp_arr2;
#endif
    InputCreator* input_creator;

#if NONINTERACTIVE == 1
    int size_vk_G1, size_vk_G2;
    int size_answer_G1, size_answer_G2;

#if GGPR == 1
    // EK
    G1_t *f1_g_Ai_query, *f1_g_Bi_query, *f1_g_Ci_query, *f2_g_t_i_query;
    G2_t *f1_h_Bi_query;
    G1_t *f1_g_alpha_Ai_query, *f1_g_alpha_Bi_query, *f1_g_alpha_Ci_query, *f2_g_alpha_t_i_query;
    G1_t *f1_g_beta_a_Ai_query, *f1_g_beta_b_Bi_query, *f1_g_beta_c_Ci_query;

    int size_f1_g_Ai_query, size_f1_g_Bi_query, size_f1_h_Bi_query, size_f1_g_Ci_query, size_f2_g_t_i_query;
    int size_f1_g_alpha_Ai_query, size_f1_g_alpha_Bi_query, size_f1_g_alpha_Ci_query, size_f2_g_alpha_t_i_query;
    int size_f1_g_beta_a_Ai_query, size_f1_g_beta_b_Bi_query, size_f1_g_beta_c_Ci_query;

    // VK
    G2_t h;
    G2_t h_alpha, h_gamma;
    G2_t h_beta_a_gamma, h_beta_b_gamma, h_beta_c_gamma;
    G2_t h_D;
    G1_t g_A0;
    G2_t h_B0;
    G1_t g_C0;
    G1_t* g_Ai_io;
#else
    // EK
    G1_t *f1_g_a_Ai_query, *f1_g_b_Bi_query, *f1_g_c_Ci_query;
    G2_t *f1_h_b_Bi_query;
    G1_t *f1_g_a_alpha_a_Ai_query, *f1_g_b_alpha_b_Bi_query, *f1_g_c_alpha_c_Ci_query;
    //G2_t *f2_h_t_i_query;
    G1_t *f2_g_t_i_query;
    G1_t *f1_beta_query;

    int size_f1_g_a_Ai_query, size_f1_g_b_Bi_query, size_f1_h_b_Bi_query, size_f1_g_c_Ci_query;
    int size_f1_g_a_alpha_a_Ai_query, size_f1_g_b_alpha_b_Bi_query, size_f1_g_c_alpha_c_Ci_query;
    //int size_f2_h_t_i_query;
    int size_f2_g_t_i_query;
    int size_f1_beta_query;
#if PROTOCOL == PINOCCHIO_ZK
    G1_t g_a_D, g_b_D /*, g_c_D */;
    G2_t h_b_D;

    G1_t g_a_alpha_a_D, g_b_alpha_b_D, g_c_alpha_c_D;
    G1_t g_a_beta_D, g_b_beta_D, g_c_beta_D;
#endif

    // VK
    G1_t g;
    G2_t h;
    G2_t h_alpha_a, h_alpha_b, h_alpha_c;
    G1_t g_alpha_b;
    G2_t h_gamma, h_beta_gamma;
    G1_t g_gamma, g_beta_gamma;
    G1_t g_c_D;
    G1_t g_a_A0;
    G2_t h_b_B0;
    G1_t g_c_C0;
    G1_t* g_a_Ai_io;
    mpz_t r_c_D;
    
    #if PUBLIC_VERIFIER == 0
    G1_t g_a_base;
    mpz_t *Ai_io;
    #endif
#endif
    // answers
    G1_t *f_ni_answers_G1;
    G2_t *f_ni_answers_G2;

    //pairing_t pairing;
#endif

    void init_state();
    void init_qap(const char *);
    void create_input();
#if NONINTERACTIVE == 1
    void create_noninteractive_query();
    void test_noninteractive_protocol(uint32_t beta);
    void prepare_noninteractive_answers(uint32_t beta);
    bool run_noninteractive_tests(uint32_t beta);
#if GGPR == 1
    void create_noninteractive_GGPR_query();
    bool run_noninteractive_GGPR_tests(uint32_t beta);
#endif
#else
    bool run_interactive_tests(uint32_t beta);
#endif
    void create_plain_queries();
    void populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta);
    bool run_correction_and_circuit_tests(uint32_t beta);
  public:
    ZComputationVerifier(int batch, int reps, int ip_size, int out_size,
                         int num_vars, int num_cons, int optimize_answers, char *prover_url,
                         const char *name_prover, int, int, int, const char *file_name_qap);
    ~ZComputationVerifier();
};
#endif  // CODE_PEPPER_APPS_POLYEVAL_D2_V_H_
