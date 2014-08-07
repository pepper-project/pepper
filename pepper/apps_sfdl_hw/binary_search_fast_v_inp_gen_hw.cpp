#include <apps_sfdl_gen/binary_search_fast_v_inp_gen.h>
#include <apps_sfdl_hw/binary_search_fast_v_inp_gen_hw.h>
#include <apps_sfdl_gen/binary_search_fast_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

binary_search_fastVerifierInpGenHw::binary_search_fastVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/binary_search_fast_cons.h for constants to use when generating input.
void binary_search_fastVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }
}
