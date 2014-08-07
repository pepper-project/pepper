#include <apps_sfdl_hw/memcpy_merkle_p_exo.h>
#include <apps_sfdl_gen/memcpy_merkle_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

memcpy_merkleProverExo::memcpy_merkleProverExo() { }

using namespace memcpy_merkle_cons;

void memcpy_merkleProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void memcpy_merkleProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void memcpy_merkleProverExo::run_shuffle_phase(char *folder_path) {

}

void memcpy_merkleProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}

void memcpy_merkleProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

  for (int i=0; i < num_outputs; i++) {
      mpq_set(output_recomputed[i],input_q[i+num_inputs-num_outputs]);
  }

}

//Refer to apps_sfdl_gen/memcpy_merkle_cons.h for constants to use in this exogenous
//check.
bool memcpy_merkleProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs - 1);
  baseline(input_q, num_inputs, output_recomputed, num_outputs - 1);

  gmp_printf("\noutput[0] = %Qd\n",output_q[0]);
  for(int i = 0; i < 4; i++){
    gmp_printf("%Qd (%Qd) ",output_q[i+1],output_recomputed[i]);
    if (mpq_equal(output_recomputed[i], output_q[i+1]) == 0){
      passed_test = false;
    }
  }
  gmp_printf("\n\n");
  clear_vec(4, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
