#include <apps_sfdl_gen/uintsqrt_v_inp_gen.h>
#include <apps_sfdl_hw/uintsqrt_v_inp_gen_hw.h>
#include <apps_sfdl_gen/uintsqrt_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

uintsqrtVerifierInpGenHw::uintsqrtVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/uintsqrt_cons.h for constants to use when generating input.
void uintsqrtVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }
}
