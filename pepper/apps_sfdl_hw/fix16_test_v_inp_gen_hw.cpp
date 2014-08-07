#include <apps_sfdl_gen/fix16_test_v_inp_gen.h>
#include <apps_sfdl_hw/fix16_test_v_inp_gen_hw.h>
#include <apps_sfdl_gen/fix16_test_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

fix16_testVerifierInpGenHw::fix16_testVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

#include <include/fix_t.h>

//Refer to apps_sfdl_gen/fix16_test_cons.h for constants to use when generating input.
void fix16_testVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  //compiler_implementation.create_input(input_q, num_inputs);
  #endif

  for(int i = 0; i < num_inputs; i++){
    fix_t x = -1926;
    if (i % 2 == 1){
      x = 66493;
    }
    mpq_set_si(input_q[i], x, 1);
  }

  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }
}
