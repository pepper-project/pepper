#include <apps_sfdl_gen/dna_align_v_inp_gen.h>
#include <apps_sfdl_hw/dna_align_v_inp_gen_hw.h>
#include <apps_sfdl_gen/dna_align_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

dna_alignVerifierInpGenHw::dna_alignVerifierInpGenHw(Venezia* v_) {
  v = v_;
  compiler_implementation.v = v_;
  srand(time(0));
}

//Refer to apps_sfdl_gen/dna_align_cons.h for constants to use when generating input.
void dna_alignVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs) {
  //Default implementation is provided by compiler
  //compiler_implementation.create_input(input_q, num_inputs);

  const char DNA [] = {'A', 'T', 'C', 'G'};

  //Generate random DNA sequences
  for(int i = 0; i < num_inputs; i++) {
    mpq_set_ui(input_q[i], DNA[((uint32_t)rand())%4], 1);
  }
}
