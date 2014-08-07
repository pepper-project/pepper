#include <apps_sfdl_gen/gmp_test_v_inp_gen.h>
#include <apps_sfdl_hw/gmp_test_v_inp_gen_hw.h>
#include <apps_sfdl_gen/gmp_test_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

gmp_testVerifierInpGenHw::gmp_testVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

#include <apps_sfdl/gmp_test.h>

//Refer to apps_sfdl_gen/gmp_test_cons.h for constants to use when generating input.
void gmp_testVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  //There can be a race condition here...
  srand(time(0));
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }
}
