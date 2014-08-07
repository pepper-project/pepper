#include <apps_sfdl_hw/mm_pure_arith_p_exo.h>
#include <apps_sfdl_gen/mm_pure_arith_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mm_pure_arithProverExo::mm_pure_arithProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  //baseline_minimal_input_size = sizeof(something);
  //baseline_minimal_output_size = sizeof(something);

}

//using namespace mm_pure_arith_cons;

void mm_pure_arithProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  
}

void mm_pure_arithProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

}

void mm_pure_arithProverExo::run_shuffle_phase(char *folder_path) {

}

void mm_pure_arithProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
}


using mm_pure_arith_cons::SIZE;
void mm_pure_arithProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  //struct In input;
  //struct Out output;
  int16_t A[SIZE][SIZE];
  int16_t B[SIZE][SIZE];
  int64_t C[SIZE][SIZE];
  const mpq_t* mpqA = input_q;
  const mpq_t* mpqB = input_q + SIZE*SIZE;
  for(int i = 0; i < SIZE; i++){
    for(int j = 0; j < SIZE; j++){
      A[i][j] = mpz_get_si(mpq_numref(mpqA[j+i*SIZE]));
      B[i][j] = mpz_get_si(mpq_numref(mpqB[j+i*SIZE]));
    }
  }

  // Do the computation
  for(int i = 0; i < SIZE; i++){
    for(int j = 0; j < SIZE; j++){
      C[i][j] = 0;
      for(int k = 0; k < SIZE; k++){
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }

  // Fill code here to dump output to output_recomputed.
  mpq_set_si(output_recomputed[0],0,1);
  //C
  mpq_t* mpqC = output_recomputed + 1;
  for(int i = 0; i < SIZE; i++){
    for(int j = 0; j < SIZE; j++){
      mpq_set_si(mpqC[j+i*SIZE], C[i][j], 1);
    }
  }
}

//Refer to apps_sfdl_gen/mm_pure_arith_cons.h for constants to use in this exogenous
//check.
bool mm_pure_arithProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
    //gmp_printf("Output actual: %Qd expected: %Qd\n", output_q[i], output_recomputed[i]);
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
