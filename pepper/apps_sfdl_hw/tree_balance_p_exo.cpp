#include <apps_sfdl_hw/tree_balance_p_exo.h>
#include <apps_sfdl_gen/tree_balance_cons.h>
#include <common/sha1.h>
#include <include/avl_tree.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/tree_balance.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

tree_balanceProverExo::tree_balanceProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace tree_balance_cons;

void tree_balanceProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void tree_balanceProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void tree_balanceProverExo::run_shuffle_phase(char *folder_path) {

}

void tree_balanceProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

int compute(struct In *input, struct Out *output) {
  tree_t tree;
  tree.root = input->root;
  tree_balance(&tree, input->path_depth, input->tree_path);
  output->root = tree.root;
  return 0;
}

void tree_balanceProverExo::baseline(const mpq_t* input_q, int num_inputs,
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;
  // Fill code here to prepare input from input_q.
  uint64_t* input_ptr = (uint64_t*)&input.root;
  int number_of_hash_elements = sizeof(hash_t) / sizeof(uint64_t);
  for (int i = 0; i < number_of_hash_elements; i++) {
    input_ptr[i] = mpz_get_ui(mpq_numref(input_q[i]));
  }
  input.path_depth = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements]));
  input.tree_path = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements + 1]));

  // Call baseline_minimal to run the computation
  compute(&input, &output);

  // Fill code here to dump output to output_recomputed.
  mpq_set_si(output_recomputed[0], 0, 1);
  uint64_t* output_ptr = (uint64_t*)&output.root;
  for(int i = 0; i < number_of_hash_elements; i++) {
    mpq_set_ui(output_recomputed[1 + i], output_ptr[i], 1);
  }
}

//Refer to apps_sfdl_gen/tree_balance_cons.h for constants to use in this exogenous
//check.
bool tree_balanceProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
