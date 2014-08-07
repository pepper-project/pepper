#include <apps_sfdl_hw/fix16_test_p_exo.h>
#include <apps_sfdl_gen/fix16_test_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

fix16_testProverExo::fix16_testProverExo() { }

using namespace fix16_test_cons;

void fix16_testProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
}

void fix16_testProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void fix16_testProverExo::run_shuffle_phase(char *folder_path) {

}

#include <apps_sfdl/fix16_test.c>

void fix16_testProverExo::baseline(const mpq_t* input_q, int num_inputs,
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;
  // Fill code here to prepare input from input_q.
  for(int i = 0; i < num_inputs; i++){
    input.data[i] = mpz_get_si(mpq_numref(input_q[i]));
  }

  // Do the computation
  compute(&input, &output);

  // Fill code here to dump output to output_recomputed.
  mpq_set_si(output_recomputed[0], 0, 1);
  mpq_set_si(output_recomputed[1], output.result, 1);
}

//Refer to apps_sfdl_gen/fix16_test_cons.h for constants to use in this exogenous
//check.
bool fix16_testProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    gmp_printf("Index %d : Exogenous %Qd Pantry %Qd\n", i, output_recomputed[i], output_q[i]);

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
