#include <apps_sfdl_hw/sparse_matvec_p_exo.h>
#include <apps_sfdl_gen/sparse_matvec_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

sparse_matvecProverExo::sparse_matvecProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace sparse_matvec_cons;

void sparse_matvecProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void sparse_matvecProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void sparse_matvecProverExo::run_shuffle_phase(char *folder_path) {

}

void sparse_matvecProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

using sparse_matvec_cons::N;
using sparse_matvec_cons::K;
void sparse_matvecProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

    int values[N+N+K+K+1];
    int *vector = values;
    int *elms = &(values[N]);
    int *inds = &(values[N+K]);
    int *ptrs = &(values[N+K+K]);
    for(int i=0; i<N+N+K+K+1; i++) {
        values[i] = mpz_get_ui(mpq_numref(input_q[i]));
    }

    int out[N] = {0,};
    for(int i=0; i < N; i++) {
        for(int j=ptrs[i]; j < ptrs[i+1]; j++) {
            out[i] += elms[j] * vector[inds[j]];
        }
    }

    mpq_set_ui(output_recomputed[0], 0, 1);
    for(int i=0; i < N; i++) {
        mpq_set_ui(output_recomputed[i+1], out[i], 1);
    }
}

//Refer to apps_sfdl_gen/sparse_matvec_cons.h for constants to use in this exogenous
//check.
bool sparse_matvecProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
