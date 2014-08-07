#ifndef APPS_SFDL_GEN_CQL_RW_UPDATE_P_EXO_H_
#define APPS_SFDL_GEN_CQL_RW_UPDATE_P_EXO_H_

#include <apps_sfdl_gen/cql_rw_update_p.h>
#include <apps_sfdl_gen/cql_rw_update_v_inp_gen.h>

//Comment out line to disable exogenous checking for this computation.
#define ENABLE_EXOGENOUS_CHECKING

/*
* Overrides the exogenous check method of ComputationProver.
*/
class cql_rw_updateProverExo : public ExogenousChecker {
  public:
    cql_rw_updateProverExo();

    bool exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime);
    
    void baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs);
    void init_exo_inputs(const mpq_t*, int, char*, HashBlockStore*);
    void export_exo_inputs(const mpq_t*, int, char*, HashBlockStore*);
    void run_shuffle_phase(char *folder_path); 
};
#endif  // APPS_SFDL_GEN_CQL_RW_UPDATE_P_EXO_H_
