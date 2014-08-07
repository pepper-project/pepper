#include <apps_sfdl_hw/merge_sort_p_exo.h>
#include <apps_sfdl_gen/merge_sort_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

merge_sortProverExo::merge_sortProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace merge_sort_cons;

void merge_sortProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  }

void merge_sortProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void merge_sortProverExo::run_shuffle_phase(char *folder_path) {

}

void merge_sortProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

void merge_sortProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  //struct In input;
  //struct Out output;
  // Fill code here to prepare input from input_q.
  
  // Call baseline_minimal to run the computation

  // Fill code here to dump output to output_recomputed.
}

//Refer to apps_sfdl_gen/merge_sort_cons.h for constants to use in this exogenous
//check.
bool merge_sortProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  for(int i = 1; i < num_outputs; i++){
    if (!(mpq_cmp(output_q[i - 1], output_q[i]) <= 0)) {
      passed_test = false;
      //break;
    }
    //gmp_printf("%Qd\n", output_q[i]);
  }
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
