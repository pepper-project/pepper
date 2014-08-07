#include <apps_sfdl_gen/horspool_flat_v_inp_gen.h>
#include <apps_sfdl_hw/horspool_flat_v_inp_gen_hw.h>
#include <apps_sfdl_gen/horspool_flat_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

horspool_flatVerifierInpGenHw::horspool_flatVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/horspool_flat_cons.h for constants to use when generating input.
using horspool_flat_cons::ALPHABET_LENGTH;
using horspool_flat_cons::PATTERN_LENGTH;
using horspool_flat_cons::HAYSTACK_LENGTH;
void horspool_flatVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  srand(time(NULL));
  for(int j=0; j < PATTERN_LENGTH + HAYSTACK_LENGTH; j++) {
      mpq_set_ui(input_q[j], rand() % ALPHABET_LENGTH, 1);
  }
}
