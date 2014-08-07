#include <apps_sfdl_hw/cql_rw_update_p_exo.h>
#include <apps_sfdl_gen/cql_rw_update_cons.h>
#include <common/sha1.h>
#include <include/avl_tree.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/cql_rw_update.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

cql_rw_updateProverExo::cql_rw_updateProverExo() { }

using namespace cql_rw_update_cons;

void cql_rw_updateProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
}

void cql_rw_updateProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {
  //dump_vector(num_inputs, input_q, "db_handle", FOLDER_PERSIST_STATE);
}

void cql_rw_updateProverExo::run_shuffle_phase(char *folder_path) {

}

int compute(struct In *input, struct Out *output) {
  Student_handle_t handle;
  tree_t KEY_index;
  tree_t Average_index;
  struct {Student_result_t student[1];} results;
  int num_results;

  hashget(&handle, &(input->db_handle));
  KEY_index.root = handle.KEY_index;
  Average_index.root = handle.Average_index;

  uint32_t path_depth = 0, tree_path = 0;
  {
    tree_result_set_t tempResult;
    Student_t tempStudent;
    memset(&tempStudent, 0, sizeof(Student_t));
    hash_t tempHash;
    hash_t oldHash;
    tree_find_eq(&(KEY_index), (90), &(tempResult));
    if (tempResult.num_results == 0) {
      tempStudent.KEY = 90;
      oldHash = *NULL_HASH;
    } else {
      hashget(&(tempStudent), &(tempResult.results[0].value));
      oldHash = tempResult.results[0].value;
    }
    tempStudent.Honored = 1;
    hashput(&(tempHash), &(tempStudent));
    tree_update_no_balance(&(KEY_index), (tempStudent.KEY), (oldHash), (tempHash), &path_depth, &tree_path);
  }

  handle.KEY_index = KEY_index.root;
  handle.Average_index = Average_index.root;
  hashput(&(output->db_handle), &handle);

  output->path_depth = path_depth;
  output->tree_path = tree_path;

  return 0;
}

void cql_rw_updateProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;

  // Fill code here to prepare input from input_q.
  uint64_t* input_ptr = (uint64_t*)&input;
  int number_of_hash_elements = sizeof(hash_t) / sizeof(uint64_t);
  for(int i = 0; i < number_of_hash_elements; i++) {
    input_ptr[i] = mpz_get_ui(mpq_numref(input_q[i]));
  }

  // Do the computation
  compute(&input, &output);

  // Fill code here to dump output to output_recomputed.
  mpq_set_si(output_recomputed[0], 0, 1);

  uint64_t* output_ptr = (uint64_t*)&output.db_handle;
  number_of_hash_elements = sizeof(hash_t) / sizeof(uint64_t);
  for(int i = 0; i < number_of_hash_elements; i++) {
    mpq_set_ui(output_recomputed[1 + i], output_ptr[i], 1);
  }
  mpq_set_ui(output_recomputed[1 + number_of_hash_elements], output.path_depth, 1);
  mpq_set_ui(output_recomputed[1 + number_of_hash_elements + 1], output.tree_path, 1);
}

//Refer to apps_sfdl_gen/cql_rw_update_cons.h for constants to use in this exogenous
//check.
bool cql_rw_updateProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      passed_test = false;
      gmp_printf("Failure: %Qd %Qd %d\n", output_recomputed[i], output_q[i], i);
      //break;
    } else {
      gmp_printf("Output %d: %Qd\n", i, output_q[i]);
    }
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
