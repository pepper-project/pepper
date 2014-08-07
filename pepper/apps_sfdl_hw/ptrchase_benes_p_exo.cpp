#include <apps_sfdl_hw/ptrchase_benes_p_exo.h>
#include <apps_sfdl_gen/ptrchase_benes_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

ptrchase_benesProverExo::ptrchase_benesProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace ptrchase_benes_cons;

void ptrchase_benesProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void ptrchase_benesProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void ptrchase_benesProverExo::run_shuffle_phase(char *folder_path) {

}

void ptrchase_benesProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

using ptrchase_benes_cons::NELMS;
using ptrchase_benes_cons::NDEEP;
void ptrchase_benesProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

    uint32_t input[NELMS];

    // read in data
    for(int i=0; i < NELMS; i++) {
        input[i] = mpz_get_ui(mpq_numref(input_q[i]));
    }

    // chase pointers
    uint32_t current = input[0];
    for(int i=0; i < NDEEP; i++) {
        current = input[current];
    }

    // set output
    mpq_set_ui(output_recomputed[0], 0, 1);
    mpq_set_ui(output_recomputed[1], current, 1);
}

//Refer to apps_sfdl_gen/ptrchase_benes_cons.h for constants to use in this exogenous
//check.
bool ptrchase_benesProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  //gmp_printf("<Exogenous check not implemented>");
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
