#include <apps_sfdl_gen/store_test_v_inp_gen.h>
#include <apps_sfdl_hw/store_test_v_inp_gen_hw.h>
#include <apps_sfdl_gen/store_test_cons.h>
#include <storage/ram_impl.h>
#include <storage/hasher.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

store_testVerifierInpGenHw::store_testVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/store_test_cons.h for constants to use when generating input.
void store_testVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  //gmp_printf("Printing input: \n");

  //for(int i = 0; i < num_inputs; i++){
  //  gmp_printf("%d %Qd\n",i,input_q[i]);
  //}
}

#pragma pack(pop)
