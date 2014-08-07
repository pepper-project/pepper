#include <apps_sfdl_hw/hash_blocks_p_exo.h>
#include <apps_sfdl_gen/hash_blocks_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

hash_blocksProverExo::hash_blocksProverExo() { }

void hash_blocksProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  
}

//Refer to apps_sfdl_gen/hash_blocks_cons.h for constants to use in this exogenous
//check.
bool hash_blocksProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  //baseline(input_q, num_inputs, output_recomputed, num_outputs);
  cout<<"Exogeneous check not implemented, but the hashes are as follows: "<<endl;
  for (int i=0; i<num_outputs; i++) {
    gmp_printf("%Zx ", mpq_numref(output_q[i]));
  }
  cout<<endl;
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};
