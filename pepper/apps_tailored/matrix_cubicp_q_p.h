#ifndef CODE_PEPPER_APPS_MATRIX_CUBICP_Q_P_H_
#define CODE_PEPPER_APPS_MATRIX_CUBICP_Q_P_H_
#include <libv/prover.h>
#include <apps_tailored/matrix_cubicp_pcp.h>

#define PROVER_NAME "/matrix_cubicp_p"

class MatrixCubicQProver : public Prover {
  
  private:
    mpz_t *A, *B, *C, *F1, *output, *f1_commitment, *f1_q1, *f1_q2, *f1_q3, *f1_q4, *f2_q1, *f2_q2;
    mpz_t *gamma;
    mpq_t *A_q, *B_q, *C_q;
    int size_f1_vec;

  public:
    MatrixCubicQProver(int, int, int, int);
    void init_state();
    void find_cur_qlengths();
    void prover_computation_commitment();
    void computation_matrixmult();
    void computation_assignment();
    void deduce_queries();
};
#endif  // CODE_PEPPER_APPS_MATRIX_CUBICP_Q_P_H_
