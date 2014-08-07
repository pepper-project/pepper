#include <apps_sfdl_hw/rle_codec_flat_p_exo.h>
#include <apps_sfdl_gen/rle_codec_flat_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

rle_codec_flatProverExo::rle_codec_flatProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace rle_codec_flat_cons;

void rle_codec_flatProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void rle_codec_flatProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void rle_codec_flatProverExo::run_shuffle_phase(char *folder_path) {

}

void rle_codec_flatProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

using rle_codec_flat_cons::SIZE;
void rle_codec_flatProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

    int input[SIZE];

    int mid[2*SIZE] = {0,};
    int midp = 0;

    int output[SIZE];
    int outp = 0;

    for(int i=0; i < SIZE; i++) {
        input[i] = mpz_get_si(mpq_numref(input_q[i]));
    }

    int data = input[0];
    int dcount = 0;

    for (int i = 1; i < SIZE; i++) {
        if (input[i] == data) {
            dcount++;
        } else {
            mid[midp] = data;
            mid[midp + 1] = dcount;
            midp += 2;
            data = input[i];
            dcount = 0;
        }
    }
    mid[midp] = data;
    mid[midp + 1] = dcount;

    midp = 0;
    while (outp < SIZE) {
        data = mid[midp];
        dcount = mid[midp+1];
        midp += 2;

        do {
            output[outp] = data;
            outp++;
        } while (dcount-- > 0);
    }

    mpq_set_ui(output_recomputed[0], 0, 1);
    for(int i = 0; i < SIZE; i++) {
        mpq_set_ui(output_recomputed[i+1], output[i], 1);
    }
}

//Refer to apps_sfdl_gen/rle_codec_flat_cons.h for constants to use in this exogenous
//check.
bool rle_codec_flatProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
