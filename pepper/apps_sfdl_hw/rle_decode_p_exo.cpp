#include <apps_sfdl_hw/rle_decode_p_exo.h>
#include <apps_sfdl_gen/rle_decode_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

rle_decodeProverExo::rle_decodeProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace rle_decode_cons;

void rle_decodeProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {

}

void rle_decodeProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void rle_decodeProverExo::run_shuffle_phase(char *folder_path) {

}

void rle_decodeProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

using rle_decode_cons::OUTPUT_SIZE;
void rle_decodeProverExo::baseline(const mpq_t* input_q, int num_inputs,
      mpq_t* output_recomputed, int num_outputs) {
    int input[2*OUTPUT_SIZE];
    int output[OUTPUT_SIZE];
    int outp = 0;

    for(int i=0; i < 2*OUTPUT_SIZE; i++) {
        input[i] = mpz_get_ui(mpq_numref(input_q[i]));
    }

    for(int i = 0; outp < OUTPUT_SIZE; i += 2) {
        int data = input[i];
        int len = input[i+1];

        for(int j = 0; j <= len; j++) {
            output[outp++] = data;
            if (outp >= OUTPUT_SIZE) { break; }
        }
    }

    mpq_set_ui(output_recomputed[0], 0, 1);
    for(int i=0; i < OUTPUT_SIZE; i++) {
        mpq_set_ui(output_recomputed[i+1], output[i], 1);
    }
}

//Refer to apps_sfdl_gen/rle_decode_cons.h for constants to use in this exogenous
//check.
bool rle_decodeProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
