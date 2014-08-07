#ifndef APPS_SFDL_GEN_BASE_2_LOG_P_EXO_H_
#define APPS_SFDL_GEN_BASE_2_LOG_P_EXO_H_

#include <apps_sfdl_gen/base_2_log_p.h>
#include <apps_sfdl_gen/base_2_log_v_inp_gen.h>

//Comment out line to disable exogenous checking for this computation.
#define ENABLE_EXOGENOUS_CHECKING

/*
* Overrides the exogenous check method of ComputationProver.
*/
class base_2_logProverExo : public ExogenousChecker {
  public:
    base_2_logProverExo();

    bool exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime);
    
    void baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs);
    void init_exo_inputs(const mpq_t*, int, char*, HashBlockStore*);
    void export_exo_inputs(const mpq_t*, int, char*, HashBlockStore*);
    void run_shuffle_phase(char *folder_path); 
};
#endif  // APPS_SFDL_GEN_BASE_2_LOG_P_EXO_H_
