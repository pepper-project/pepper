#ifndef APPS_SFDL_GEN_BINARY_TREE_TEST_P_EXO_H_
#define APPS_SFDL_GEN_BINARY_TREE_TEST_P_EXO_H_

#include <apps_sfdl_gen/binary_tree_test_p.h>

//Comment out line to disable exogenous checking for this computation.
#define ENABLE_EXOGENOUS_CHECKING

/*
* Overrides the exogenous check method of ComputationProver.
*/
class binary_tree_testProverExo : public ExogenousChecker {
  public:
    binary_tree_testProverExo();

    bool exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime);
    
    void baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs);
};
#endif  // APPS_SFDL_GEN_BINARY_TREE_TEST_P_EXO_H_
