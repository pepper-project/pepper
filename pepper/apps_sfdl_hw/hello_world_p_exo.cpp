#include <apps_sfdl_hw/hello_world_p_exo.h>
#include <apps_sfdl_gen/hello_world_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

hello_worldProverExo::hello_worldProverExo() { }

void hello_worldProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
}

//Refer to apps_sfdl_gen/hello_world_cons.h for constants to use in this exogenous
//check.
bool hello_worldProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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

