#ifndef APPS_SFDL_GEN_SHARED_STORE_P_EXO_H_
#define APPS_SFDL_GEN_SHARED_STORE_P_EXO_H_

#include <apps_sfdl_gen/shared_store_p.h>

//Comment out line to disable exogenous checking for this computation.
#define ENABLE_EXOGENOUS_CHECKING

/*
* Overrides the exogenous check method of ComputationProver.
*/
class shared_storeProverExo : public ExogenousChecker {
  public:
    shared_storeProverExo();

    bool exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime);
    
    void baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs);
    void init_exo_inputs(const mpq_t*, int, char*, HashBlockStore*);
    void export_exo_inputs(const mpq_t*, int, char*, HashBlockStore*);
    void run_shuffle_phase(char *folder_path); 
};
#endif  // APPS_SFDL_GEN_SHARED_STORE_P_EXO_H_
