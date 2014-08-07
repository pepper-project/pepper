#include <apps_sfdl_hw/horspool_p_exo.h>
#include <apps_sfdl_gen/horspool_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

horspoolProverExo::horspoolProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace horspool_cons;

void horspoolProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void horspoolProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void horspoolProverExo::run_shuffle_phase(char *folder_path) {

}

void horspoolProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

using horspool_cons::ALPHABET_LENGTH;
using horspool_cons::PATTERN_LENGTH;
using horspool_cons::HAYSTACK_LENGTH;
void horspoolProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

    uint8_t needle[PATTERN_LENGTH];
    uint8_t haystack[HAYSTACK_LENGTH];
    uint8_t table[ALPHABET_LENGTH];

    for(int i=0; i < PATTERN_LENGTH; i++) {
        needle[i] = mpz_get_ui(mpq_numref(input_q[i]));
    }

    for(int i=0; i < HAYSTACK_LENGTH; i++) {
        haystack[i] = mpz_get_ui(mpq_numref(input_q[PATTERN_LENGTH + i]));
    }

    for(int i=0; i < ALPHABET_LENGTH; i++) {
        table[i] = PATTERN_LENGTH;
    }

    int last = PATTERN_LENGTH - 1;
    for(int i=0; i < last; i++) {
        uint8_t addr = needle[i];
        table[addr] = last - i;
    }

    int result = HAYSTACK_LENGTH;
    int hlen = HAYSTACK_LENGTH;
    uint8_t *hptr = haystack;

    while (hlen >= PATTERN_LENGTH) {
        for (int i=last; hptr[i] == needle[i]; i--) {
            if (i == 0) {
                result = (int) (hptr - haystack);
                goto search_done;
            }
        }

        // mismatch occurred
        uint8_t skip = table[hptr[last]];
        hlen -= skip;
        hptr += skip;
    }
search_done:
    mpq_set_ui(output_recomputed[0], 0, 1);
    mpq_set_ui(output_recomputed[1], result, 1);
}

//Refer to apps_sfdl_gen/horspool_cons.h for constants to use in this exogenous
//check.
bool horspoolProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
