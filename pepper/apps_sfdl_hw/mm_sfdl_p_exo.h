#ifndef APPS_SFDL_GEN_MM_SFDL_P_EXO_H_
#define APPS_SFDL_GEN_MM_SFDL_P_EXO_H_

#include <apps_sfdl_gen/mm_sfdl_p.h>

//Comment out line to disable exogenous checking for this computation.
#define ENABLE_EXOGENOUS_CHECKING = 1

/*
* Overrides the exogenous check method of ComputationProver.
*/
class mm_sfdlProverExo : public ExogenousChecker {
  public:
    mm_sfdlProverExo();
    
    void baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs); 

    bool exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime);
};
#endif  // APPS_SFDL_GEN_MM_SFDL_P_EXO_H_
