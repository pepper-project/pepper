#include <apps_sfdl_gen/ptrchase_benes_v_inp_gen.h>
#include <apps_sfdl_hw/ptrchase_benes_v_inp_gen_hw.h>
#include <apps_sfdl_gen/ptrchase_benes_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

ptrchase_benesVerifierInpGenHw::ptrchase_benesVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/ptrchase_benes_cons.h for constants to use when generating input.
using ptrchase_benes_cons::NELMS;
void ptrchase_benesVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  for(int i=0; i < NELMS-1; i++) {
      mpq_set_ui(input_q[i], i+1, 1);
  }
  mpq_set_ui(input_q[NELMS-1], 0, 1);
}
