#include <apps_sfdl_gen/rle_codec_flat_v_inp_gen.h>
#include <apps_sfdl_hw/rle_codec_flat_v_inp_gen_hw.h>
#include <apps_sfdl_gen/rle_codec_flat_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

rle_codec_flatVerifierInpGenHw::rle_codec_flatVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/rle_codec_flat_cons.h for constants to use when generating input.
using rle_codec_flat_cons::SIZE;
void rle_codec_flatVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  srand(time(NULL));
  for(int j=0; j < SIZE; j++) {
      mpq_set_ui(input_q[j], rand() % 256, 1);
  }
}
