#ifndef CODE_PEPPER_APPS_HDIST_V_H_
#define CODE_PEPPER_APPS_HDIST_V_H_

#include <libv/verifier.h>
#include <apps_tailored/hdist_pcp.h>
#define NAME_PROVER "hdist"

#ifdef INTERFACE_MPI
#include <apps_tailored/hdist_p.h>
#endif

class HDistVerifier : public Verifier {
  private:
    mpz_t *A, *B;
    mpz_t *f1_q1, *f1_q2, *f1_q3, *f1_q4, *f1_commitment, *f1_consistency;
    mpz_t *f2_q1, *f2_q2, *f2_q3, *f2_q4, *f2_commitment, *f2_consistency;
    mpz_t *f3_q1, *f3_q2, *f3_q3, *f3_q4, *f3_commitment, *f3_consistency;
    mpz_t *f4_q1, *f4_q2, *f4_q3, *f4_q4, *f4_commitment, *f4_consistency;
    mpz_t *f5_q1, *f5_q2, *f5_q3, *f5_q4, *f5_commitment, *f5_consistency;
    mpz_t *ckt_answers;
    mpz_t *alpha;
    mpz_t neg, neg_i;
    mpz_t temp, *output;
    mpz_t a1, a2, a3, a4, a5, f1_s, f2_s, f3_s, f4_s, f5_s;
    mpz_t *temp_arr, *temp_arr2, *temp_arr3, *temp_arr4, *temp_arr5;
    mpz_t c_init_val;
    int m;
    int size_f1_vec, size_f2_vec, size_f3_vec, size_f4_vec, size_f5_vec;

  public:
    HDistVerifier(int batch, int reps, int ip_size, int optimize_answers, char *prover_url);
    void init_state();
    void create_input();
    void create_plain_queries();
    bool run_correction_and_circuit_tests(uint32_t beta);
    void populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta);
};
#endif  // CODE_PEPPER_APPS_HDIST_V_H_
