#include <apps_sfdl_hw/tree_range_test_p_exo.h>
#include <apps_sfdl_gen/tree_range_test_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

static void print_outputs(const mpq_t* output_q, int num_outputs) {
	gmp_printf("\n");
	gmp_printf("tree_range_test outputs:\n");
	for (int i = 0; i < num_outputs; i++) {
	  gmp_printf("Output %d: %Zd\n", i, mpq_numref(output_q[i]));
	}
	gmp_printf("\n\n");
}

tree_range_testProverExo::tree_range_testProverExo() { }

using namespace tree_range_test_cons;

void tree_range_testProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void tree_range_testProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void tree_range_testProverExo::run_shuffle_phase(char *folder_path) {

}

void tree_range_testProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  //struct In input;
  //struct Out output;
  // Fill code here to prepare input from input_q.
  
  // Do the computation

  // Fill code here to dump output to output_recomputed.
}

//Refer to apps_sfdl_gen/tree_range_test_cons.h for constants to use in this exogenous
//check.
bool tree_range_testProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  gmp_printf("<Exogenous check not implemented>");
  /*mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      passed_test = false;
      break;
    }
  }
  clear_vec(num_outputs, output_recomputed);*/

  print_outputs(output_q, num_outputs);


#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
