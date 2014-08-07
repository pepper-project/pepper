#include <apps_sfdl_hw/bounded_loop_p_exo.h>
#include <apps_sfdl_gen/bounded_loop_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

bounded_loopProverExo::bounded_loopProverExo() { }

void bounded_loopProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

  int32_t SIZE = bounded_loop_cons::SIZE;

  int32_t a[SIZE];
  int64_t subSum = 0;
  for(int i = 0; i < SIZE; i++){
    a[i] = mpz_get_si(mpq_numref(input_q[i]));
    if (a[i] < 0){
      break;
    }
    subSum += a[i];
  }

  //void compute(struct In *input, struct Out *output){
  //}

  //Return value
  mpq_set_si(output_recomputed[0], 0, 1); 
  //Output struct
  mpq_set_si(output_recomputed[1], subSum, 1);

}

//Refer to apps_sfdl_gen/bounded_loop_cons.h for constants to use in this exogenous
//check.
bool bounded_loopProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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

