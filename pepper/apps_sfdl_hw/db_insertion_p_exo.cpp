#include <apps_sfdl_hw/db_insertion_p_exo.h>
#include <apps_sfdl_gen/db_insertion_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

db_insertionProverExo::db_insertionProverExo() { }

void db_insertionProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
}

//Refer to apps_sfdl_gen/db_insertion_cons.h for constants to use in this exogenous
//check.
bool db_insertionProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  //Fill me out!
  gmp_printf("<Exogenous check not yet implemented>\n");
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

