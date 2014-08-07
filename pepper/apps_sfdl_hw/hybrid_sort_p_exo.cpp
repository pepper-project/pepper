#include <apps_sfdl_hw/hybrid_sort_p_exo.h>
#include <apps_sfdl_gen/hybrid_sort_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

hybrid_sortProverExo::hybrid_sortProverExo() { }

void hybrid_sortProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

  std::vector<uint32_t> vec;
  int SIZE = hybrid_sort_cons::SIZE;
  int i;

  //There are SIZE inputs, put at the end of the inputs
  //We ignore the merkle tree root hash, which comes at the start.
  int input_start = num_inputs - SIZE;

  for(i = 0; i < SIZE; i++){
    vec.push_back(mpz_get_ui(mpq_numref(input_q[i+input_start])));
  }
  
  std::sort(vec.begin(), vec.end());

  //Return value
  mpq_set_si(output_recomputed[0], 0, 1);
  //sha1
  mpq_t* outvec = output_recomputed + 1;
  for(i = 0; i < SIZE; i++) {
    mpq_set_ui(outvec[i], vec[i], 1);
  }
}

//Refer to apps_sfdl_gen/hybrid_sort_cons.h for constants to use in this exogenous
//check.
bool hybrid_sortProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {
 bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    gmp_printf("%Qd %Qd\n",output_recomputed[i], output_q[i]);
  }
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
