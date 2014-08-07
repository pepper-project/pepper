#include <apps_sfdl_hw/nested_ifs_p_exo.h>
#include <apps_sfdl_gen/nested_ifs_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

nested_ifsProverExo::nested_ifsProverExo()
{
}

//Refer to apps_sfdl_gen/nested_ifs_cons.h for constants to use in this exogenous
//check.
bool nested_ifsProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs){

#ifdef ENABLE_EXOGENOUS_CHECKING
  //Fill me out!

  gmp_printf("<Exogenous check not yet implemented>\n");
#endif

  return true;
};

