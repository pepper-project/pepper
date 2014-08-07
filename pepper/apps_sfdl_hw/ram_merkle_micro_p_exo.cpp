#include <apps_sfdl_hw/ram_merkle_micro_p_exo.h>
#include <apps_sfdl_gen/ram_merkle_micro_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

ram_merkle_microProverExo::ram_merkle_microProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace ram_merkle_micro_cons;

void ram_merkle_microProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void ram_merkle_microProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void ram_merkle_microProverExo::run_shuffle_phase(char *folder_path) {

}

void ram_merkle_microProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

void ram_merkle_microProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  //struct In input;
  //struct Out output;
  // Fill code here to prepare input from input_q.
  
  // Call baseline_minimal to run the computation

  // Fill code here to dump output to output_recomputed.
}

//Refer to apps_sfdl_gen/ram_merkle_micro_cons.h for constants to use in this exogenous
//check.
bool ram_merkle_microProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  // for (int i=0; i<num_outputs; i++) {
  //  gmp_printf("Output is %Zd", mpq_numref(output_q[i])); 
  // }
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
