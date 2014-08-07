#ifndef APPS_SFDL_GEN_MR_VECADD_RED_P_EXO_H_
#define APPS_SFDL_GEN_MR_VECADD_RED_P_EXO_H_

#include <apps_sfdl_gen/mr_vecadd_red_p.h>

//Comment out line to disable exogenous checking for this computation.
#define ENABLE_EXOGENOUS_CHECKING

/*
* Overrides the exogenous check method of ComputationProver.
*/
class mr_vecadd_redProverExo : public ExogenousChecker {
  public:
    mr_vecadd_redProverExo();

    bool exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime);
    
    void baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs);
};
#endif  // APPS_SFDL_GEN_MR_VECADD_RED_P_EXO_H_
