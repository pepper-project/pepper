#include <apps_sfdl_gen/boyer_occur_benes_v_inp_gen.h>
#include <apps_sfdl_hw/boyer_occur_benes_v_inp_gen_hw.h>
#include <apps_sfdl_gen/boyer_occur_benes_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

boyer_occur_benesVerifierInpGenHw::boyer_occur_benesVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/boyer_occur_benes_cons.h for constants to use when generating input.
using boyer_occur_benes_cons::ALPHABET_LENGTH;
using boyer_occur_benes_cons::PATTERN_LENGTH;
void boyer_occur_benesVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  srand(time(NULL));
  for(int j = 0; j < PATTERN_LENGTH; j++) {
      mpq_set_ui(input_q[j], rand() % ALPHABET_LENGTH, 1);
  }
}
