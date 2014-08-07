#include <apps_sfdl_hw/cql_ro_range_p_exo.h>
#include <apps_sfdl_gen/cql_ro_range_cons.h>
#include <common/sha1.h>
#include <include/avl_tree.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/cql_ro_range.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

cql_ro_rangeProverExo::cql_ro_rangeProverExo() { }

using namespace cql_ro_range_cons;

void cql_ro_rangeProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
}

void cql_ro_rangeProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void cql_ro_rangeProverExo::run_shuffle_phase(char *folder_path) {

}

int compute(struct In *input, struct Out *output) {
  Student_handle_t handle;
  tree_t Age_index;

  hashget(&handle, &(input->db_handle));
  Age_index.root = handle.Age_index;

  //CQL("SELECT KEY, FName, LName, Age, Major, State, PhoneNum, Class, Credits, Average, Honored FROM Student WHERE Age > 20 AND Age < 27 LIMIT 5", output->result);
  int i;
  Student_t tempStudent;
  tree_result_set_t tempResult;
  tree_find_range(&(Age_index), (20), (FALSE), (27), (FALSE), &(tempResult));
  for (i = 0; i < 5; i++) {
    if (i < tempResult.num_results) {
      hashget(&(tempStudent), &(tempResult.results[i].value));
      output->result[i].KEY = tempStudent.KEY;
      output->result[i].Major = tempStudent.Major;
      output->result[i].LName = tempStudent.LName;
      output->result[i].State = tempStudent.State;
      output->result[i].Age = tempStudent.Age;
      output->result[i].Class = tempStudent.Class;
      output->result[i].FName = tempStudent.FName;
      output->result[i].Credits = tempStudent.Credits;
      output->result[i].Average = tempStudent.Average;
      output->result[i].PhoneNum = tempStudent.PhoneNum;
      output->result[i].Honored = tempStudent.Honored;
    } else {
      output->result[i].KEY = 0;
      output->result[i].Major = 0;
      output->result[i].LName = 0;
      output->result[i].State = 0;
      output->result[i].Age = 0;
      output->result[i].Class = 0;
      output->result[i].FName = 0;
      output->result[i].Credits = 0;
      output->result[i].Average = 0;
      output->result[i].PhoneNum = 0;
      output->result[i].Honored = 0;
    }
  }
  return 0;
}

void cql_ro_rangeProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;

  // Fill code here to prepare input from input_q.
  uint64_t* input_ptr = (uint64_t*)&input.db_handle;
  for(int i = 0; i < num_inputs; i++) {
    input_ptr[i] = mpz_get_ui(mpq_numref(input_q[i]));
  }

  // Do the computation
  compute(&input, &output);

  // Fill code here to dump output to output_recomputed.
  int number_of_fields = 11;
  mpq_set_si(output_recomputed[0], 0, 1);
  for (int i = 0; i < 5; i++) {
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 0], output.result[i].KEY, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 1], output.result[i].FName, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 2], output.result[i].LName, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 3], output.result[i].Age, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 4], output.result[i].Major, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 5], output.result[i].State, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 6], output.result[i].PhoneNum, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 7], output.result[i].Class, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 8], output.result[i].Credits, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 9], output.result[i].Average, 1);
    mpq_set_ui(output_recomputed[1 + i * number_of_fields + 10], output.result[i].Honored, 1);
  }
}

//Refer to apps_sfdl_gen/cql_ro_range_cons.h for constants to use in this exogenous
//check.
bool cql_ro_rangeProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
