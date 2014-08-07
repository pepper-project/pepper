#include <apps_sfdl_hw/gmp_test_p_exo.h>
#include <apps_sfdl_gen/gmp_test_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

gmp_testProverExo::gmp_testProverExo() { }

using namespace gmp_test_cons;

void gmp_testProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void gmp_testProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void gmp_testProverExo::run_shuffle_phase(char *folder_path) {

}

#include <apps_sfdl/gmp_test.c>

/**
 In constraints, mpz_t are represented as mpz_length many
 field elements.
**/
void mpz_to_constraint_rep(mpq_t* constraint_vars, mpz_t value, int mpz_length){
  //Each limb is unsigned 32 bits.
  for(int i = 0; i < mpz_length; i++){
    uint32_t limb = 0;
    for(int j = 0; j < 32; j++){
      //According to GMP documentation, tstbit returns bits
      //as if the mpz_t were expanded in infinitely long twos
      //complement.
      limb |= mpz_tstbit(value, i*32 + j) << j;
    }
    mpq_set_ui(constraint_vars[i], limb, 1);
  }
}

void gmp_testProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  struct In input;
  struct Out output;
  // Fill code here to prepare input from input_q.

  for(int i = 0; i < num_inputs; i++){
    input.data[i] = mpz_get_si(mpq_numref(input_q[i]));
  }
  alloc_init_scalar(output.b);

  // Do the computation
  compute(&input, &output);

  // Fill code here to dump output to output_recomputed.
  mpq_set_si(output_recomputed[0], 0, 1);

  //Convert output.b to match the representation used in the constraints
  mpz_to_constraint_rep(output_recomputed + 1, output.b, gmp_test_cons::MPZ_LENGTH);

  clear_scalar(output.b);
}

//Refer to apps_sfdl_gen/gmp_test_cons.h for constants to use in this exogenous
//check.
bool gmp_testProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_inputs; i++){
    gmp_printf("Input index %d = %Qd\n", i, input_q[i]);
  }

  for(int i = 0; i < num_outputs; i++){
    gmp_printf("Index %d : Exogenous %Qd Pantry %Qd\n", i,
      output_recomputed[i], output_q[i]);
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
