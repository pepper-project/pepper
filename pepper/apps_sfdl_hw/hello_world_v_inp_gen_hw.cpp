#include <apps_sfdl_gen/hello_world_v_inp_gen.h>
#include <apps_sfdl_hw/hello_world_v_inp_gen_hw.h>
#include <apps_sfdl_gen/hello_world_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

hello_worldVerifierInpGenHw::hello_worldVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/hello_world_cons.h for constants to use when generating input.
void hello_worldVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
}
