#include <apps_sfdl_hw/hello_world_bitwise_p_exo.h>
#include <apps_sfdl_gen/hello_world_bitwise_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

hello_world_bitwiseProverExo::hello_world_bitwiseProverExo() { }

void hello_world_bitwiseProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  const int32_t SIZE = hello_world_bitwise_cons::SIZE;

  struct In { int32_t A[SIZE][SIZE]; int32_t B[SIZE][SIZE]; };
  struct Out { int32_t C[SIZE][SIZE]; };
  struct In input2;
  struct Out output2;
  struct In *input = &input2;
  struct Out *output = &output2;
  const mpq_t* input_qA = input_q;
  const mpq_t* input_qB = input_q + SIZE*SIZE;
  {
    for(int i = 0; i < SIZE; i++){
      for(int j = 0; j < SIZE; j++){
	input->A[i][j] = mpz_get_si(mpq_numref(input_qA[i*SIZE+j]));
	input->B[i][j] = mpz_get_si(mpq_numref(input_qB[i*SIZE+j]));
      }
    }
  }

  //void compute(struct In *input, struct Out *output){
    int i, j, k;
    for (i=0; i<SIZE; i++) {
      for (j=0; j < SIZE; j++){
	int32_t t = 0;
	for(k = 0; k < SIZE; k++){
	  t |= input->A[i][k] & input->B[k][j];
	}
	output->C[i][j] = t;
      }
    }
  //}

  {
    //Return value
    mpq_set_ui(output_recomputed[0], 0, 1);
    mpq_t* output_C = output_recomputed + 1;
    for(int i = 0; i < SIZE; i++){
      for(int j = 0; j < SIZE; j++){
	mpq_set_si(output_C[i*SIZE+j], output->C[i][j], 1);
      }
    }
  }
}

//Refer to apps_sfdl_gen/hello_world_bitwise_cons.h for constants to use in this exogenous
//check.
bool hello_world_bitwiseProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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

