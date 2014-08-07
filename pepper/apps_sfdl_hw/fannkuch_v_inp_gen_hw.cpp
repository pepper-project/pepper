#include <apps_sfdl_gen/fannkuch_v_inp_gen.h>
#include <apps_sfdl_hw/fannkuch_v_inp_gen_hw.h>
#include <apps_sfdl_gen/fannkuch_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

fannkuchVerifierInpGenHw::fannkuchVerifierInpGenHw(Venezia* v_) {
  v = v_;
  compiler_implementation.v = v_;

  std::srand(time(0));
}

//Refer to apps_sfdl_gen/fannkuch_cons.h for constants to use when generating input.
void fannkuchVerifierInpGenHw::create_input(mpq_t* input_q, int input_size) {

  int m = fannkuch_cons::m;
  //Default implementation is provided by compiler
  //compiler_implementation.create_input(input_q, num_inputs);

  std::vector<int> permutation(m);
  for(int i = 0; i < m; i++) {
    permutation[i] = i+1;
  }
  std::random_shuffle(permutation.begin(), permutation.end());

  for(int i = 0; i < input_size; i++) {
    mpq_set_ui(input_q[i], permutation[i], 1);
  }

}
