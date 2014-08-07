#include <apps_sfdl_gen/rle_decode_v_inp_gen.h>
#include <apps_sfdl_hw/rle_decode_v_inp_gen_hw.h>
#include <apps_sfdl_gen/rle_decode_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

rle_decodeVerifierInpGenHw::rle_decodeVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/rle_decode_cons.h for constants to use when generating input.
using rle_decode_cons::OUTPUT_SIZE;
void rle_decodeVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  srand(time(NULL));
  for(int j=0; j < 2*OUTPUT_SIZE; j+=2) {
      mpq_set_ui(input_q[j], rand() % 256, 1);
      mpq_set_ui(input_q[j+1], 0, 1);
  }
}
