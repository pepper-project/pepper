#ifndef APPS_SFDL_GEN_DNA_ALIGN_P_EXO_H_
#define APPS_SFDL_GEN_DNA_ALIGN_P_EXO_H_

#include <apps_sfdl_gen/dna_align_p.h>

//Comment out line to disable exogenous checking for this computation.
#define ENABLE_EXOGENOUS_CHECKING

/*
* Overrides the exogenous check method of ComputationProver.
*/
class dna_alignProverExo : public ExogenousChecker {
  public:
    dna_alignProverExo();
    
    void baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs); 
 
    bool exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime);
};
#endif  // APPS_SFDL_GEN_DNA_ALIGN_P_EXO_H_
