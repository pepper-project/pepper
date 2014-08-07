#include <apps_sfdl_hw/divide_int_int_p_exo.h>
#include <apps_sfdl_gen/divide_int_int_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

divide_int_intProverExo::divide_int_intProverExo() { }

using namespace divide_int_int_cons;

void divide_int_intProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void divide_int_intProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void divide_int_intProverExo::run_shuffle_phase(char *folder_path) {

}

void divide_int_intProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  //struct In input;
  //struct Out output;
  // Fill code here to prepare input from input_q.
 
  int32_t a = mpz_get_si(mpq_numref(input_q[0])); 
  int32_t b = mpz_get_si(mpq_numref(input_q[1])); 
  // Do the computation
  int32_t c = a % b;

  // Fill code here to dump output to output_recomputed.
  mpq_set_si(output_recomputed[0], 0, 1);
  mpq_set_si(output_recomputed[1], c, 1);
}

//Refer to apps_sfdl_gen/divide_int_int_cons.h for constants to use in this exogenous
//check.
bool divide_int_intProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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

#pragma pack(pop)
