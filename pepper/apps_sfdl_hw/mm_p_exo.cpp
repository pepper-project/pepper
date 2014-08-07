#include <apps_sfdl_hw/mm_p_exo.h>
#include <apps_sfdl_gen/mm_cons.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mmProverExo::mmProverExo() { }

using namespace mm_cons;
typedef int32_t num_t;

void mmProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  // Fill code here to prepare input from input_q.
  num_t A[SIZE][SIZE];
  num_t B[SIZE][SIZE];
  num_t C[SIZE][SIZE];
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

//Refer to apps_sfdl_gen/mm_cons.h for constants to use in this exogenous
//check.
bool mmProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
