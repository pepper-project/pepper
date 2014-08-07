#include <apps_sfdl_gen/f_w_apsp_v_inp_gen.h>
#include <apps_sfdl_hw/f_w_apsp_v_inp_gen_hw.h>
#include <apps_sfdl_gen/f_w_apsp_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

f_w_apspVerifierInpGenHw::f_w_apspVerifierInpGenHw(Venezia* v_) {
  v = v_;
  compiler_implementation.v = v_;
  srand(time(0));
}

//Refer to apps_sfdl_gen/f_w_apsp_cons.h for constants to use when generating input.
void f_w_apspVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs) {
  //Default implementation is provided by compiler
  //compiler_implementation.create_input(input_q, num_inputs);

  int32_t Infinity = f_w_apsp_cons::Infinity;

  for(int i = 0; i < num_inputs; i++) {
    if (rand() % 2 == 0) {
      mpq_set_si(input_q[i], Infinity, 1);
    } else {
      mpq_set_si(input_q[i], 1, 1);
    }
  }
}
