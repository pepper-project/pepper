#ifndef CODE_PEPPER_APPS_HDIST_P_H_
#define CODE_PEPPER_APPS_HDIST_P_H_

#include <libv/prover.h>
#include <apps_tailored/hdist_pcp.h>

#define PROVER_NAME "hdist"

class HDistProver : public Prover {
  private:
    mpz_t neg, *alpha;
    mpz_t *A, *B, *Y, *S, *M, *AM, *AS; 
    mpz_t *F1, *F2, *F3, *F4, *F5;
    mpz_t *f1_commitment, *f2_commitment, *f3_commitment, *f4_commitment, *f5_commitment;
    mpz_t *f1_consistency, *f2_consistency, *f3_consistency, *f4_consistency, *f5_consistency;
    mpz_t *f1_q1, *f2_q1, *f1_q2, *f2_q2, *f1_q3, *f1_q4, *f2_q3, *f2_q4;
    mpz_t *f3_q1, *f4_q1, *f3_q2, *f4_q2, *f3_q3, *f3_q4, *f4_q3, *f4_q4;
    mpz_t *f5_q1, *f5_q2, *f5_q3, *f5_q4;
    mpz_t *dotp;
    int size_f1_vec, size_f2_vec, size_f3_vec, size_f4_vec, size_f5_vec;

  public:
    HDistProver(int, int, int, int);
    void init_state();
    void find_cur_qlengths();
    void prover_computation_commitment();
    void computation();
    void deduce_queries();
};

#endif  // CODE_PEPPER_APPS_HDIST_P_H_
