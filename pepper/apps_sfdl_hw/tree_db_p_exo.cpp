#include <apps_sfdl_hw/tree_db_p_exo.h>
#include <apps_sfdl_hw/tree_db_v_inp_gen_hw.h>
#include <apps_sfdl_gen/tree_db_cons.h>
#include <include/binary_tree_int_hash_t.h>
#include <storage/exo.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

tree_dbProverExo::tree_dbProverExo() { }

void tree_dbProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
}

void tree_dbProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void tree_dbProverExo::run_shuffle_phase(char *folder_path) {

}

void compute(struct In *input, struct Out *output) {
  uint32_t tempInt, tempRowID;
  uint32_t nextRowID, numberOfRows, rowOffset, i;

  Student_t tempStudent, tempOldStudent;
  tree_t Age_index;
  Age_t tempAge;
  tree_result_set_t result;
  memset(&Age_index, 0, sizeof(tree_t));
  memset(&result, 0, sizeof(tree_result_set_t));

  /*SELECT KEY, Age FROM Student WHERE Age < 24*/
  Age_index.root = input->hash;
  tempAge = 24;
  tree_find_lt(&(Age_index), (tempAge), FALSE, &(result));
  output->rows = result.num_results;
  for (i = 0; i < 3; i++) {
    if (i < result.num_results) {
      hashget(&(tempStudent), &(result.results[i].value));
      output->result[i].KEY = tempStudent.KEY;
      output->result[i].FName = tempStudent.FName;
      output->result[i].LName = tempStudent.LName;
      output->result[i].Age = tempStudent.Age;
      output->result[i].Major = tempStudent.Major;
      output->result[i].State = tempStudent.State;
      output->result[i].PhoneNum = tempStudent.PhoneNum;
      output->result[i].Class = tempStudent.Class;
      output->result[i].Credits = tempStudent.Credits;
      output->result[i].Average = tempStudent.Average;
    } else {
      output->result[i].KEY = 0;
      output->result[i].FName = 0;
      output->result[i].LName = 0;
      output->result[i].Age = 0;
      output->result[i].Major = 0;
      output->result[i].State = 0;
      output->result[i].PhoneNum = 0;
      output->result[i].Class = 0;
      output->result[i].Credits = 0;
      output->result[i].Average = 0;
    }
  }
}

void tree_dbProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;

  for(int i = 0; i < num_inputs; i++) {
    input.hash.bit[i] = mpz_get_ui(mpq_numref(input_q[i]));
  }
  compute(&input, &output);

  mpq_set_si(output_recomputed[0], 0, 1);
  mpq_set_si(output_recomputed[1], output.rows, 1);
  for (int i = 0; i < 3; i++) {
    mpq_set_ui(output_recomputed[2 + i * 10 + 0], output.result[i].KEY, 1);
    mpq_set_ui(output_recomputed[2 + i * 10 + 1], output.result[i].FName, 1);
    mpq_set_ui(output_recomputed[2 + i * 10 + 2], output.result[i].LName, 1);
    mpq_set_ui(output_recomputed[2 + i * 10 + 3], output.result[i].Age, 1);
    mpq_set_ui(output_recomputed[2 + i * 10 + 4], output.result[i].Major, 1);
    mpq_set_ui(output_recomputed[2 + i * 10 + 5], output.result[i].State, 1);
    mpq_set_ui(output_recomputed[2 + i * 10 + 6], output.result[i].PhoneNum, 1);
    mpq_set_ui(output_recomputed[2 + i * 10 + 7], output.result[i].Class, 1);
    mpq_set_ui(output_recomputed[2 + i * 10 + 8], output.result[i].Credits, 1);
    mpq_set_ui(output_recomputed[2 + i * 10 + 9], output.result[i].Average, 1);
  }

}

//Refer to apps_sfdl_gen/tree_db_cons.h for constants to use in this exogenous
//check.
bool tree_dbProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
    }
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};
