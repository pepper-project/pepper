#ifndef CODE_PEPPER_APPS_MATRIX_CUBICP_Q_V_H_
#define CODE_PEPPER_APPS_MATRIX_CUBICP_Q_V_H_
#include <libv/verifier.h>
#include <apps_tailored/matrix_cubicp_pcp.h>

#define NAME_PROVER "matrix_cubicp_q"
#ifdef INTERFACE_MPI
#include <apps_tailored/matrix_cubicp_q_p.h>
#endif

class MatrixCubicQVerifier : public Verifier {
  private:
    mpz_t *f1_commitment, *f1_consistency;
    mpz_t *A, *B, *C;
    mpq_t *A_q, *C_q;
    mpq_t *q_input, *q_output;
    mpz_t *f1_q1, *f1_q2, *f1_q3, *f1_q4;
    mpz_t *f2_q1, *f2_q2;
    mpz_t *f1_con_coins;
    mpz_t *gamma;
    mpz_t temp, temp2, f1_s, a1, a2, a3;
    mpz_t *f1_answers, *f2_answers, *ckt_answers;
    mpz_t *temp_arr;

    int hadamard_code_size;
  
  public:
    MatrixCubicQVerifier(int batch, int reps, int ip_size, int optimize_answers, char *prover_url);
    void init_state();
    void create_input();
    void create_plain_queries();
    bool run_correction_and_circuit_tests(uint32_t beta);
    void populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta);
};
#endif  //CODE_PEPPER_APPS_MATRIX_CUBICP_Q_V_H_
