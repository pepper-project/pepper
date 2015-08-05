#ifndef CODE_PEPPER_LIBV_ZCOMPUTATION_P_H_
#define CODE_PEPPER_LIBV_ZCOMPUTATION_P_H_

#include <libv/computation_p.h>
#include <libv/zpcp.h>
#include <common/utility.h>


#ifdef USE_LIBSNARK
#include "relations/constraint_satisfaction_problems/r1cs/r1cs.hpp"
#include "common/default_types/ec_pp.hpp"
#include "zk_proof_systems/ppzksnark/r1cs_ppzksnark/r1cs_ppzksnark.hpp"
#include "algebra/curves/public_params.hpp"
#include "algebra/curves/bn128/bn128_pp.hpp"
#include "algebra/curves/public_params.hpp"
#include "algebra/fields/fp.hpp"
#endif


#if NONINTERACTIVE == 1
#include <common/pairing_util.h>
#endif
static const unsigned char bit_reversal_table[] = {
  0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30, 0xB0, 0x70, 0xF0, 
  0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98, 0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 
  0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64, 0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 
  0x0C, 0x8C, 0x4C, 0xCC, 0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 
  0x02, 0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2, 0x72, 0xF2, 
  0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A, 0xDA, 0x3A, 0xBA, 0x7A, 0xFA,
  0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6, 0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 
  0x0E, 0x8E, 0x4E, 0xCE, 0x2E, 0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE,
  0x01, 0x81, 0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71, 0xF1,
  0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9, 0x39, 0xB9, 0x79, 0xF9, 
  0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15, 0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5,
  0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD, 0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD,
  0x03, 0x83, 0x43, 0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3, 
  0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B, 0xBB, 0x7B, 0xFB,
  0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97, 0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 
  0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F, 0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};

class ZComputationProver : public ComputationProver {
  protected:
    int n, n_prime, chi;

    poly_compressed *poly_A, *poly_B, *poly_C;
    int num_aij, num_bij, num_cij;
    int num_levels;

    mpz_t *poly_A_pv, *poly_B_pv, *poly_C_pv;
    mpz_t *eval_poly_A, *eval_poly_B, *eval_poly_C;
    mpz_t *powers_of_omega, omega_inv;
    mpz_t *set_v_z;
    vec_ZZ_p qap_roots, qap_roots2, single_root, set_v, v_prime;
    vec_ZZ_p z_poly_A_pv, z_poly_B_pv, z_poly_C_pv;
    ZZ_pX z_poly_A_c, z_poly_B_c, z_poly_C_c;
    ZZ_pX z_poly_P_c;
    ZZ_pX z_poly_P_c2;
    ZZ_pX z_poly_D_c, z_poly_H_c;
    ZZ_pX *poly_tree, *interpolation_tree;

    mpz_t omega;

    #if NONINTERACTIVE == 1
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
    G2_t h_alpha, h_gamma, h_beta_a_gamma, h_beta_b_gamma, h_beta_c_gamma;
    G1_t g_D;
    G1_t g_A0;
    G2_t h_B0;
    G1_t g_C0;
    G1_t* g_Ai_io;
#else
    // EK
    G1_t *f1_g_a_Ai_query, *f1_g_b_Bi_query, *f1_g_c_Ci_query;
    G2_t *f1_h_b_Bi_query;
    G1_t *f1_g_a_alpha_a_Ai_query, *f1_g_b_alpha_b_Bi_query, *f1_g_c_alpha_c_Ci_query;
//    G2_t *f2_h_t_i_query;
    G1_t *f2_g_t_i_query;
    G1_t *f1_beta_query;

#if PROTOCOL == PINOCCHIO_ZK
    //Additional terms given by the verifier to the prover for the ZK
    //modification
    G1_t g_a_D, g_b_D, g_c_D;
    G2_t h_b_D;
    G1_t g_a_alpha_a_D, g_b_alpha_b_D, g_c_alpha_c_D;
    G1_t g_a_beta_D, g_b_beta_D, g_c_beta_D;

    //delta_abc, the three random field elements chosen secretly by
    //the prover
    mpz_t *delta_abc;
#endif

    int size_f1_g_a_Ai_query, size_f1_g_b_Bi_query, size_f1_h_b_Bi_query, size_f1_g_c_Ci_query;
    int size_f1_g_a_alpha_a_Ai_query, size_f1_g_b_alpha_b_Bi_query, size_f1_g_c_alpha_c_Ci_query;
//    int size_f2_h_t_i_query;
    int size_f2_g_t_i_query;
    int size_f1_beta_query;
#endif
    // answers
    G1_t *f_ni_answers_G1;
    G2_t *f_ni_answers_G2;

    //pairing_t pairing;
#endif

  private:
    void build_poly_tree(int level, int j, int index);
    void zcomp_interpolate(int level, int j, int index, vec_ZZ_p *);
    mpz_t* zcomp_fast_interpolate(int k, mpz_t *A, mpz_t omega_k, mpz_t prime); 
    void init_state(const char *);
    //void init_noninteractive_state();
    void find_cur_qlengths();
    virtual void interpret_constraints() = 0;
    void compute_assignment_vectors();
    void init_qap(const char*);
    void prover_computation_commitment();
    void prover_do_computation();
#if NONINTERACTIVE == 1
    void prover_noninteractive();
#if GGPR == 1
    void prover_noninteractive_GGPR();
#endif
#else
    void prover_interactive();
#endif
    void deduce_queries();
    uint32_t zreverse (uint32_t v); 

  public:
    ZComputationProver(int ph, int b_size, int num_r, int size_input,
                       int size_output, int num_variables, int num_constraints,
                       const char *name_prover, int size_aij, int size_bij,
                       int size_cij, const char *file_name_qap, const char *file_name_f1_index);
    ZComputationProver(int ph, int b_size, int num_r, int size_input,
                       int size_output, int num_variables, int num_constraints,
                       const char *name_prover, int size_aij, int size_bij,
                       int size_cij, const char *file_name_qap, const char *file_name_f1_index, const char *_shared_bstore_file_name);
    ~ZComputationProver();
};


#endif  // CODE_PEPPER_LIBV_ZCOMPUTATION_P_H_
