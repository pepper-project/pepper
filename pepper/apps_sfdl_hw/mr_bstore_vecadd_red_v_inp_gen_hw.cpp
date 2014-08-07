#include <apps_sfdl_gen/mr_bstore_vecadd_red_v_inp_gen.h>
#include <apps_sfdl_hw/mr_bstore_vecadd_red_v_inp_gen_hw.h>
#include <apps_sfdl_gen/mr_bstore_vecadd_red_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_bstore_vecadd_redVerifierInpGenHw::mr_bstore_vecadd_redVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/mr_bstore_vecadd_red_cons.h for constants to use when generating input.
void mr_bstore_vecadd_redVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  //Default implementation is provided by compiler
  #if IS_REDUCER == 0
  compiler_implementation.create_input(input_q, num_inputs);
  #endif
}
