#ifndef CODE_PEPPER_LIBV_computation_V_H_
#define CODE_PEPPER_LIBV_computation_V_H_

#include <libv/verifier.h>
#include <libv/input_creator.h>
#include <libv/gpcp_compiled.h>

class GComputationVerifier : public Verifier {
  protected:
    mpz_t *ckt_answers, *alpha;
    mpz_t neg, neg_i;
    mpz_t temp;
    uint32_t *F1_index;
    mpz_t *input_output;

    mpz_t *f1_commitment, *f2_commitment;
    mpz_t *f1_consistency, *f2_consistency;
    mpz_t *temp_arr, *temp_arr2;
    mpz_t *f1_q1, *f1_q2, *f1_q3, *f1_q4, *f1_q5;
    mpz_t *f2_q1, *f2_q2, *f2_q3, *f2_q4;

    int num_cons;
    int f_con_filled;
    int query_id; //TODO: use f_con_filled in place of query_id
    InputCreator* input_creator;

  public:
    GComputationVerifier(int batch, int reps, int ip_size, int op_size,
                         int optimize_answers, int num_v, int num_c,
                         char *prover_url, char *prover_name, const char *file_name_f1_index);
    ~GComputationVerifier();
    void init_state(const char *file_name_f1_index);
    bool run_noninteractive_tests(int beta) {return false; /* ginger doesn't support noninteractive version*/}
    void create_plain_queries();
    void create_qct_queries(int);
    void create_ckt_queries(int);
    void create_lin_queries(int);
    bool run_correction_and_circuit_tests(uint32_t beta);
    void populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta);

    virtual void create_input();
    virtual void create_gamma12() = 0;
    virtual void create_gamma0(int) = 0;
    void parse_gamma12(const char *file_name);
    void parse_gamma0(const char *file_name, int rho_i);
};
#endif  // CODE_PEPPER_LIBV_computation_V_H_
