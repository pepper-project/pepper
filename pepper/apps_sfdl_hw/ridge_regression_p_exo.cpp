#include <apps_sfdl_hw/ridge_regression_p_exo.h>
#include <apps_sfdl_gen/ridge_regression_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

#include <apps_sfdl/ridge_regression.c>

ridge_regressionProverExo::ridge_regressionProverExo() {

  baseline_minimal_input_size = sizeof(patientDB) + sizeof(fix_t);
  baseline_minimal_output_size = sizeof(struct Out);

}

void ridge_regressionProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {

}

void ridge_regressionProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void ridge_regressionProverExo::run_shuffle_phase(char *folder_path) {

}

void ridge_regressionProverExo::baseline_minimal(void* input, void*
output){

  ridge_regression(*((fix_t*)input),
    (patientDB*)(((char*)input) + sizeof(fix_t)),
    (struct Out*)output);

}

void ridge_regressionProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;
  // Fill code here to prepare input from input_q.
  int inp = 0;

  for(int j = 0; j < NUM_CK_BITS/8; j++){
    input.commitmentCK.bit[j] = mpz_get_ui(mpq_numref(input_q[inp++]));
  }
  //cout << "Read input: ";
  for(int j = 0; j < NUM_COMMITMENT_CHUNKS; j++){
    input.commitment.bit[j] = mpz_get_ui(mpq_numref(input_q[inp++]));
    //cout << (int)input.commitment.bit[j] << " ";
  }
  //cout << endl;
  input.k = mpz_get_si(mpq_numref(input_q[inp++]));

  // Do the computation
  compute(&input, &output);

  // Fill code here to dump output to output_recomputed.
  mpq_set_ui(output_recomputed[0], 0, 1);
  for(int i = 1; i < num_outputs; i++){
    mpq_set_si(output_recomputed[i], output.beta[i-1], 1);
  }
}

//Refer to apps_sfdl_gen/ridge_regression_cons.h for constants to use in this exogenous
//check.
bool ridge_regressionProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
     gmp_printf("Index %d : Exogenous %Qd Pantry %Qd \n", i,
     output_recomputed[i], output_q[i]);

    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
     passed_test = false;
      break;
    }

  }

#if COMPUTE_SOLUTION_ERROR == 1
  //Check that the last NUM_FEATURES output is all 0s
  for(int i = num_outputs - NUM_FEATURES; i < num_outputs; i++){
    if (mpq_sgn(output_recomputed[i]) != 0){
      gmp_printf("Nonzero in exogenous index %d\n", i);
    }
    if (mpq_sgn(output_q[i]) != 0){
      gmp_printf("Nonzero in pantry index %d\n", i);
    }
  }
#endif
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
