#include <apps_sfdl_hw/mergesort_merkle_p_exo.h>
#include <apps_sfdl_gen/mergesort_merkle_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mergesort_merkleProverExo::mergesort_merkleProverExo() { }

using namespace mergesort_merkle_cons;

void mergesort_merkleProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void mergesort_merkleProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void mergesort_merkleProverExo::run_shuffle_phase(char *folder_path) {

}

void mergesort_merkleProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

void mergesort_merkleProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) { }

//Refer to apps_sfdl_gen/mergesort_merkle_cons.h for constants to use in this exogenous
//check.
bool mergesort_merkleProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  gmp_printf("\n\n");
  for (int i=1; i < num_outputs; i++) {
      gmp_printf("%Qd ",input_q[i+num_inputs-num_outputs]);
  }
  gmp_printf("\n\noutput[0] = %Qd\n",output_q[0]);
  for (int i=1; i < num_outputs; i++) {
      gmp_printf("%Qd ",output_q[i]);
  }
  gmp_printf("\n\n");
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
