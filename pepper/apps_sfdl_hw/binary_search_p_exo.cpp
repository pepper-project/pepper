#include <apps_sfdl_hw/binary_search_p_exo.h>
#include <apps_sfdl_gen/binary_search_cons.h>
#include <storage/merkle_ram.h>
#include <storage/hasher.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

binary_searchProverExo::binary_searchProverExo() { }

void binary_searchProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

  int SIZE = binary_search_cons::SIZE;
  int logSIZE = binary_search_cons::logSIZE;

  //There are SIZE + 1 inputs, put at the end of the inputs
  //We ignore the merkle tree root hash, which comes at the start.
  int input_start = num_inputs - SIZE - 1;

  int present = 0;
  int min = 0;
  int max = SIZE-1;
  const mpq_t& search = input_q[input_start + SIZE];
  for(int i = 0; i < logSIZE; i++){
    int avg = (min + max) >> 1;
    const mpq_t& got = input_q[input_start + avg];
    
    int cmp = mpq_cmp(search, got);
    if (cmp == 0){
      min = avg;
      max = avg;
      present = 1;
    } else if (cmp < 0){
      max = avg;
    } else if (cmp > 0){
      min = avg;
    }
  }

  //Return value
  mpq_set_si(output_recomputed[0], 0, 1);
  //Present
  mpq_set_si(output_recomputed[1], present, 1);
  //Index
  mpq_set_si(output_recomputed[2], min, 1);
}

//Refer to apps_sfdl_gen/binary_search_cons.h for constants to use in this exogenous
//check.
bool binary_searchProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

 bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      passed_test = false;
      //gmp_printf("Failure: %Qd %Qd %d\n", output_recomputed[i], output_q[i], i);
      break;
    }
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;

};

