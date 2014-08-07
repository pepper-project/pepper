#include <apps_sfdl_hw/boyer_occur_merkle_p_exo.h>
#include <apps_sfdl_gen/boyer_occur_merkle_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

boyer_occur_merkleProverExo::boyer_occur_merkleProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace boyer_occur_merkle_cons;

void boyer_occur_merkleProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void boyer_occur_merkleProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void boyer_occur_merkleProverExo::run_shuffle_phase(char *folder_path) {

}

void boyer_occur_merkleProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

using boyer_occur_merkle_cons::ALPHABET_LENGTH;
using boyer_occur_merkle_cons::PATTERN_LENGTH;
void boyer_occur_merkleProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

    uint8_t input[PATTERN_LENGTH];
    uint8_t output[ALPHABET_LENGTH];

    for(int i=0; i < PATTERN_LENGTH; i++) {
        input[i] = mpz_get_ui(mpq_numref(input_q[num_inputs-PATTERN_LENGTH+i]));
    }

    for(int i=0; i < ALPHABET_LENGTH; i++) {
        output[i] = PATTERN_LENGTH;
    }

    for(int i=0; i < PATTERN_LENGTH - 1; i++) {
        uint8_t addr = input[i];
        output[addr] = PATTERN_LENGTH - 1 - i;
    }

    mpq_set_ui(output_recomputed[0], 0, 1);
    for(int i=0; i < ALPHABET_LENGTH; i++) {
        mpq_set_ui(output_recomputed[i+1], output[i], 1);
    }
}

//Refer to apps_sfdl_gen/boyer_occur_merkle_cons.h for constants to use in this exogenous
//check.
bool boyer_occur_merkleProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
