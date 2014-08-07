#include <apps_sfdl_hw/mr_vecadd_red_p_exo.h>
#include <apps_sfdl_gen/mr_vecadd_red_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_vecadd_redProverExo::mr_vecadd_redProverExo() { }

void mr_vecadd_redProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  mpz_set_ui(mpq_numref(output_recomputed[0]), 0);
  for (int i=0; i<num_inputs; i++) {
    mpz_add(mpq_numref(output_recomputed[0]), mpq_numref(output_recomputed[0]), mpq_numref(input_q[i]));
  }
}

//Refer to apps_sfdl_gen/mr_vecadd_red_cons.h for constants to use in this exogenous
//check.
bool mr_vecadd_redProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  int size_input = mr_vecadd_red_cons::SIZE_INPUT;
  int size_output = mr_vecadd_red_cons::SIZE_OUTPUT;

  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, size_output);
  
  baseline(input_q, size_input, output_recomputed, size_output);

  for(int i = 1; i < size_output; i++) {
    if (mpz_cmp(mpq_numref(output_q[i]), mpq_numref(output_recomputed[i-1])) != 0) {
      passed_test = false;
      break;
    }
  }
  
  // finally check that return code is zero
  if (mpz_cmp_ui(mpq_numref(output_q[0]), 0) != 0)
    passed_test = false;

  clear_vec(size_output, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};
