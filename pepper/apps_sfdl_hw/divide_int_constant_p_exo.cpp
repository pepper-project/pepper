#include <apps_sfdl_hw/divide_int_constant_p_exo.h>
#include <apps_sfdl_gen/divide_int_constant_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

divide_int_constantProverExo::divide_int_constantProverExo() { }

void divide_int_constantProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

  uint32_t DIVISOR = divide_int_constant_cons::DIVISOR;

  uint32_t a;
  uint32_t b;
  a = mpz_get_ui(mpq_numref(input_q[0]));

  //void compute(struct In *input, struct Out *output){
  b = a % DIVISOR;
  //}

  //Return value
  mpq_set_si(output_recomputed[0], 0, 1); 
  //Output struct
  mpq_set_ui(output_recomputed[1], b, 1);

}

//Refer to apps_sfdl_gen/divide_int_constant_cons.h for constants to use in this exogenous
//check.
bool divide_int_constantProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      passed_test = false;
      break;
    }
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;

};

