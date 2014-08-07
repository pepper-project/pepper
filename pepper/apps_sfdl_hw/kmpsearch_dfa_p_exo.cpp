#include <apps_sfdl_hw/kmpsearch_dfa_p_exo.h>
#include <apps_sfdl_gen/kmpsearch_dfa_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

kmpsearch_dfaProverExo::kmpsearch_dfaProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace kmpsearch_dfa_cons;

void kmpsearch_dfaProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void kmpsearch_dfaProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void kmpsearch_dfaProverExo::run_shuffle_phase(char *folder_path) {

}

void kmpsearch_dfaProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

using kmpsearch_dfa_cons::NEEDLE;
using kmpsearch_dfa_cons::HAYSTACK;
void kmpsearch_dfaProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

    int needle[NEEDLE];
    int haystack[HAYSTACK];
    for(int i=0; i < NEEDLE; i++) {
        needle[i] = mpz_get_ui(mpq_numref(input_q[i]));
    }

    for(int i=0; i < HAYSTACK; i++) {
        haystack[i] = mpz_get_ui(mpq_numref(input_q[NEEDLE + i]));
    }

    int fail[NEEDLE];
    fail[0] = -1;
    fail[1] = 0;
    int tpos = 2;
    int cand = 0;

    while (tpos < NEEDLE) {
        if (needle[tpos - 1] == needle[cand]) {
            cand++;
            fail[tpos++] = cand;
        } else if (cand > 0) {
            cand = fail[cand];
        } else {
            fail[tpos++] = 0;
        }
    }

    int found = HAYSTACK;
    int m = 0;
    int i = 0;
    const int last = NEEDLE - 1;
    const int end = HAYSTACK - last;

    while (m < end) {
        if (needle[i] == haystack[m + i]) {
            if (last == i) {
                found = m;
                break;
            }
            i++;
        } else if (fail[i] > 0) {
            i = fail[i];
            m = m + i - fail[i];
        } else {
            i = 0;
            m++;
        }
    }
    mpq_set_ui(output_recomputed[0], 0, 1);
    mpq_set_ui(output_recomputed[1], found, 1);
}

//Refer to apps_sfdl_gen/kmpsearch_dfa_cons.h for constants to use in this exogenous
//check.
bool kmpsearch_dfaProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
