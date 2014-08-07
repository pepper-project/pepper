#ifndef CODE_PEPPER_APPS_POLYEVAL_D3_V_H_
#define CODE_PEPPER_APPS_POLYEVAL_D3_V_H_

#include <libv/verifier.h>
#include <apps_tailored/polyeval_d3_pcp.h>

#define NAME_PROVER "polyeval_d3"

#ifdef INTERFACE_MPI
#include <apps_tailored/polyeval_d3_p.h>
#endif

class PolyEvalD3Verifier : public Verifier {
  private:
    int num_coefficients; 
    int size_f1_vec, size_f2_vec, size_f3_vec;
    mpz_t *f1_q1, *f1_q2, *f1_q3, *f1_q4, *f1_q5, *f1_commitment, *f1_consistency;
    mpz_t *f2_q1, *f2_q2, *f2_q3, *f2_q4, *f2_commitment, *f2_consistency;
    mpz_t *f3_q1, *f3_q2, *f3_q3, *f3_q4, *f3_commitment, *f3_consistency;
    mpz_t *coefficients, *input, *alpha;
    mpz_t neg, neg_i;
    mpz_t temp, output;
    mpz_t *ckt_answers, *temp_arr, *temp_arr2, *temp_arr3;
    int f_con_filled;

  public:
    PolyEvalD3Verifier(int batch, int reps, int ip_size, int optimize_answers, char *prover_url);
    void init_state();
    void create_input();
    void create_plain_queries();
    bool run_correction_and_circuit_tests(uint32_t beta);
    void populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta);
};
#endif  // CODE_PEPPER_APPS_POLYEVAL_D3_V_H_
