#ifndef CODE_PEPPER_LIBV_computation_P_H_
#define CODE_PEPPER_LIBV_computation_P_H_

#include <libv/computation_p.h>
#include <libv/gpcp_compiled.h>

class GComputationProver : public ComputationProver {
  protected:
    mpz_t *dotp, *alpha, neg;
    int query_id;

  public:
    GComputationProver(int, int, int, int, int, int, int, const char *, const char *);
    ~GComputationProver();
    void init_state(const char *file_name_f1_index);
    void find_cur_qlengths();
    void prover_computation_commitment();
    void compute_assignment_vectors();

    virtual void interpret_constraints() = 0;
    virtual void create_gamma12() = 0;
    
    void deduce_queries();
    void create_lin_queries(int);
    void create_qct_queries(int);
    void create_ckt_queries(int);
    void parse_gamma12(const char *);
};
#endif  // CODE_PEPPER_LIBV_computation_P_H_
