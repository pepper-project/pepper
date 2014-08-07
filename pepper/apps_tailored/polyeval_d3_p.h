#ifndef CODE_PEPPER_APPS_POLYEVAL_D3_P_H_
#define CODE_PEPPER_APPS_POLYEVAL_D3_P_H_

#include <libv/prover.h>
#include <apps_tailored/polyeval_d3_pcp.h>

#define PROVER_NAME "polyeval_d3"

class PolyEvalD3Prover : public Prover {
  private:
    mpz_t *variables, *coefficients, *output, *F1, *F2, *F3;
    mpz_t *f1_commitment, *f2_commitment, *f3_commitment;
    mpz_t *f1_consistency, *f2_consistency, *f3_consistency;
    mpz_t *f1_q1, *f2_q1, *f3_q1;
    mpz_t *f1_q2, *f2_q2, *f3_q2;
    mpz_t *f1_q3, *f2_q3, *f3_q3;
    mpz_t *f1_q4, *f2_q4, *f3_q4;
    mpz_t *f1_q5;
    mpz_t *alpha, neg;
    mpz_t temp, temp2;
    int num_coefficients;
    int size_f1_vec, size_f2_vec, size_f3_vec;

  public:
    PolyEvalD3Prover(int, int, int, int);
    void init_state();
    void find_cur_qlengths();
    void prover_computation_commitment();
    void computation_polyeval(mpz_t);
    void deduce_queries();
    void computation_assignment(mpz_t);
};

#endif  // CODE_PEPPER_APPS_POLYEVAL_D3_P_H_
