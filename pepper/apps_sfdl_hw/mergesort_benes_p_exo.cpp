#include <apps_sfdl_hw/mergesort_benes_p_exo.h>
#include <apps_sfdl_gen/mergesort_benes_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mergesort_benesProverExo::mergesort_benesProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace mergesort_benes_cons;

void mergesort_benesProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void mergesort_benesProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void mergesort_benesProverExo::run_shuffle_phase(char *folder_path) {

}

void mergesort_benesProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

using mergesort_benes_cons::MAX_SIZE;
void mergesort_benesProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

    uint32_t input[MAX_SIZE];
    uint32_t output[MAX_SIZE];

    // read in data
    for(int i=0; i < MAX_SIZE; i++) {
        input[i] = mpz_get_ui(mpq_numref(input_q[i]));
    }

    int bPtr, ePtr, mPtr, lPtr, rPtr;
    bool out2in = false;
    uint32_t *dst, *src;

    // merge sort
    for(int span = 1; span < MAX_SIZE; span *= 2) {
        if (out2in) {
            src = output;
            dst = input;
        } else {
            src = input;
            dst = output;
        }

        for (bPtr = 0; bPtr < MAX_SIZE; bPtr += 2*span) {
            lPtr = bPtr;
            mPtr = lPtr + span;
            rPtr = mPtr;
            ePtr = rPtr + span;

            for (int i = lPtr; i < ePtr; i++) {
                if ( (lPtr < mPtr) && ( (rPtr >= ePtr) || (src[lPtr] < src[rPtr]) ) ) {
                    dst[i] = src[lPtr++];
                } else {
                    dst[i] = src[rPtr++];
                }
            }
        }

        out2in = ! out2in;
    }

    // output data
    if (out2in) {
        src = output;
    } else {
        src = input;
    }

    mpq_set_ui(output_recomputed[0], 0, 1);
    for(int i=0; i < MAX_SIZE; i++) {
        mpq_set_ui(output_recomputed[1+i], src[i], 1);
    }
}

//Refer to apps_sfdl_gen/mergesort_benes_cons.h for constants to use in this exogenous
//check.
bool mergesort_benesProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
