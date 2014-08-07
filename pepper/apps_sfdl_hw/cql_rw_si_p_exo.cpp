#include <apps_sfdl_hw/cql_rw_si_p_exo.h>
#include <apps_sfdl_gen/cql_rw_si_cons.h>
#include <common/sha1.h>
#include <include/avl_tree.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/cql_rw_si.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

cql_rw_siProverExo::cql_rw_siProverExo() { }

using namespace cql_rw_si_cons;

void cql_rw_siProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
}

void cql_rw_siProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void cql_rw_siProverExo::run_shuffle_phase(char *folder_path) {

}

int compute(struct In *input, struct Out *output) {
  memset(output, 0, sizeof(struct Out));
  Student_handle_t handle;
  avl_tree_t_int_hash_t KEY_index;
  Student_t student;

  hashget(&handle, &(input->db_handle));
  KEY_index.root = handle.KEY_index;
  student = input->student;

  int i;
  uint32_t path_depth, encoded_tree_path;
  KEY_t tempKEY = student.KEY;
  FName_t tempFName = student.FName;
  LName_t tempLName = student.LName;
  Age_t tempAge = student.Age;
  Major_t tempMajor = student.Major;
  State_t tempState = student.State;
  PhoneNum_t tempPhoneNum = student.PhoneNum;
  Class_t tempClass = student.Class;
  Credits_t tempCredits = student.Credits;
  Average_t tempAverage = student.Average;
  Honored_t tempHonored = student.Honored;
  Address_t tempAddress = student.Address;
  {
    tree_result_set_t tempResult;
    Student_t tempStudent;
    memset(&tempStudent, 0, sizeof(tempStudent));
    hash_t tempHash;
    hash_t oldHash;
    tree_find_eq(&(KEY_index), (tempKEY), &(tempResult));
    if (tempResult.num_results == 0) {
      tempStudent.KEY = tempKEY;
      oldHash = *NULL_HASH;
    } else {
      hashget(&(tempStudent), &(tempResult.results[0].value));
      oldHash = tempResult.results[0].value;
    }
    tempStudent.FName = tempFName;
    tempStudent.Average = tempAverage;
    tempStudent.Credits = tempCredits;
    tempStudent.Address = tempAddress;
    tempStudent.Age = tempAge;
    tempStudent.PhoneNum = tempPhoneNum;
    tempStudent.Honored = tempHonored;
    tempStudent.Class = tempClass;
    tempStudent.Major = tempMajor;
    tempStudent.LName = tempLName;
    tempStudent.State = tempState;
    hashput(&(tempHash), &(tempStudent));
    tree_update_no_balance(&(KEY_index), (tempStudent.KEY), (oldHash), (tempHash), &(path_depth), &(encoded_tree_path));
  }

  handle.KEY_index = KEY_index.root;
  hashput(&(output->db_handle), &handle);
  output->path_depth = path_depth;
  output->tree_path= encoded_tree_path;
  return 0;
}

void cql_rw_siProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;

  // Fill code here to prepare input from input_q.
  uint64_t* input_ptr = (uint64_t*)&input;
  int number_of_hash_elements = sizeof(hash_t) / sizeof(uint64_t);
  for(int i = 0; i < number_of_hash_elements; i++) {
    input_ptr[i] = mpz_get_ui(mpq_numref(input_q[i]));
  }
  input.student.KEY = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements]));
  input.student.FName = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+1]));
  input.student.LName = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+2]));
  input.student.Age = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+3]));
  input.student.Major = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+4]));
  input.student.State = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+5]));
  input.student.PhoneNum = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+6]));
  input.student.Class = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+7]));
  input.student.Credits = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+8]));
  input.student.Average = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+9]));
  input.student.Honored = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+10]));
  for (int i = 0; i < sizeof(Address_t)/sizeof(uint64_t); i++) {
    input.student.Address.address[i] = mpz_get_ui(mpq_numref(input_q[number_of_hash_elements+11+i]));
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

//Refer to apps_sfdl_gen/cql_rw_si_cons.h for constants to use in this exogenous
//check.
bool cql_rw_siProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
