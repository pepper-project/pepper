#include <apps_sfdl_hw/shared_store_p_exo.h>
#include <apps_sfdl_gen/shared_store_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

shared_storeProverExo::shared_storeProverExo() { }

using namespace shared_store_cons;

void shared_storeProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
}

void shared_storeProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void shared_storeProverExo::run_shuffle_phase(char *folder_path) {

}

void shared_storeProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  //struct In input;
  //struct Out output;
  // Fill code here to prepare input from input_q.
  
  // Do the computation
  mpq_set_ui(output_recomputed[0], 0, 1);
  mpq_set_ui(output_recomputed[1], 15, 1);
  mpq_set_ui(output_recomputed[2], 20329, 1);
  // Fill code here to dump output to output_recomputed.
}

//Refer to apps_sfdl_gen/shared_store_cons.h for constants to use in this exogenous
//check.
bool shared_storeProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  //gmp_printf("<Exogenous check not implemented>");
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      passed_test = false;
      gmp_printf("Failure: %Qx %Qx %d\n", output_recomputed[i], output_q[i], i);
      //break;
    }
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
