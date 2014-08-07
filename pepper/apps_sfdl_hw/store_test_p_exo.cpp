#include <apps_sfdl_hw/store_test_p_exo.h>
#include <apps_sfdl_gen/store_test_cons.h>

#pragma pack(push)
//#pragma pack(4)
#pragma pack(1)

#include <storage/exo.h>
#include <include/db.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

store_testProverExo::store_testProverExo() { }

using namespace store_test_cons;

struct In {int32_t data[SIZE];};

void store_testProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  // Fill code here to prepare input from input_q.

  struct In in;
  for(int i = 0; i < num_inputs; i++){
    in.data[i] = mpz_get_ui(mpq_numref(input_q[i]));
  }  

  hash_t digest;
  __hashput(&digest, &in, sizeof(In));

  // Fill code here to dump output to output_recomputed.
  //IMPORTANT: 0th element in output_q is return status!! 
  mpq_set_ui(output_recomputed[0], 0, 1);

  int k=1;
  for (int j=0; j<NUM_HASH_CHUNKS; j++) {
    mpq_set_ui(output_recomputed[k], digest.bit[j], 1);
    k++;
  }
}

//Refer to apps_sfdl_gen/store_test_cons.h for constants to use in this exogenous
//check.
bool store_testProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  //gmp_printf("Printing input in exo check: \n");
  //for(int i = 0; i < num_inputs; i++){
  //  gmp_printf("%d %Qd\n", i, input_q[i]);
  //}
  
  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      passed_test = false;
      gmp_printf("Failure: %Qx %Qx %d\n", output_recomputed[i], output_q[i], i);
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
