#include <apps_sfdl_hw/shared_tree_p_exo.h>
#include <apps_sfdl_hw/shared_tree_v_inp_gen_hw.h>
#include <apps_sfdl_gen/shared_tree_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>
#include <include/binary_tree_int_hash_t.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

shared_treeProverExo::shared_treeProverExo() { }

using namespace shared_tree_cons;

void shared_treeProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
}

void shared_treeProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void shared_treeProverExo::run_shuffle_phase(char *folder_path) {

}

void compute(struct In *input, struct Out *output) {
  int tempInt, tempRowID;
  int nextRowID, numberOfRows, rowOffset, i;
  hash_t tempHash;
  uint32_t age;

  tree_t Age_index;
  tree_result_set_t result;
  memset(&Age_index, 0, sizeof(tree_t));
  memset(&result, 0, sizeof(tree_result_set_t));

  //tree_init(&Age_index);
  Age_index.root = input->root;
  /*tree_init(&Age_index);*/
  /*age = 15;*/
  /*hashput(&tempHash, &age);*/
  /*tree_insert(&Age_index, 15, tempHash);*/

  tree_find_lt(&(Age_index), 24, FALSE, &(result));
  output->rows = result.num_results;
  hashget(&(output->values), &(result.results[0].value));
}

void shared_treeProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;
  // Fill code here to prepare input from input_q.
  for (int i = 0; i < num_inputs; i++) {
    input.root.bit[i] = mpz_get_ui(mpq_numref(input_q[i]));
  }
  // Do the computation
  compute(&input, &output);
  // Fill code here to dump output to output_recomputed.
  mpq_set_ui(output_recomputed[0], 0, 1);
  mpq_set_ui(output_recomputed[1], output.rows, 1);
  mpq_set_ui(output_recomputed[2], output.values, 1);
}

//Refer to apps_sfdl_gen/shared_tree_cons.h for constants to use in this exogenous
//check.
bool shared_treeProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0) {
      passed_test = false;
      gmp_printf("Failure: %Qd %Qd %d\n", output_recomputed[i], output_q[i], i);
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
