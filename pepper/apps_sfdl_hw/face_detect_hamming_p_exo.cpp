#include <apps_sfdl_hw/face_detect_hamming_p_exo.h>
#include <apps_sfdl_gen/face_detect_hamming_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/face_detect_hamming.c>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

face_detect_hammingProverExo::face_detect_hammingProverExo() {

  baseline_minimal_input_size = sizeof(struct FaceDB) +
  sizeof(VerifierSideIn);
  baseline_minimal_output_size = sizeof(struct Out);
}

//using namespace face_detect_hamming_cons;

void face_detect_hammingProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
}

void face_detect_hammingProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void face_detect_hammingProverExo::run_shuffle_phase(char *folder_path) {

}

void face_detect_hammingProverExo::baseline_minimal(void* input, void*
output){
  face_detect(
    (struct FaceDB*)input,
    (VerifierSideIn*)(((char*)input) + sizeof(struct FaceDB)),
    (struct Out*)output
  );
}

void face_detect_hammingProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;
  // Fill code here to prepare input from input_q.
  int inp = 0;
  //for(int i = 0; i < NUM_HASH_CHUNKS; i++){
  for(int i = 0; i < NUM_CK_BITS/8; i++){
    input.commitmentCK.bit[i] = mpz_get_ui(mpq_numref(input_q[inp++]));
  }

  for(int i = 0; i < NUM_COMMITMENT_CHUNKS; i++){
    input.digest_of_db.bit[i] = mpz_get_ui(mpq_numref(input_q[inp++]));
  }
  for(int i = 0; inp < num_inputs; i++){
    input.verifier_in.target[i] = mpz_get_ui(mpq_numref(input_q[inp++]));
  }

  // Do the computation
  compute(&input, &output);

  // Fill code here to dump output to output_recomputed.
  mpq_set_ui(output_recomputed[0], 0, 1);
  mpq_set_si(output_recomputed[1], output.match_found, 1);
}

//Refer to apps_sfdl_gen/face_detect_hamming_cons.h for constants to use in this exogenous
//check.
bool face_detect_hammingProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
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
