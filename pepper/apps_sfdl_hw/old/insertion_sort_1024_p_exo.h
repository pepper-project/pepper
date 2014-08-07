#ifndef APPS_SFDL_GEN_INSERTION_SORT_P_EXO_H_
#define APPS_SFDL_GEN_INSERTION_SORT_P_EXO_H_

#include <apps_sfdl_gen/insertion_sort_1024_p.h>

//Comment out line to disable exogenous checking for this computation.
#define ENABLE_EXOGENOUS_CHECKING = 1

/*
* Overrides the exogenous check method of ComputationProver.
*/
class insertion_sort_1024ProverExo : public ExogenousChecker {
  public:
    insertion_sort_1024ProverExo();

    bool exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs);
};
#endif  // APPS_SFDL_GEN_INSERTION_SORT_P_EXO_H_
