#include <apps_sfdl_gen/pam_clustering_v_inp_gen.h>
#include <apps_sfdl_hw/pam_clustering_v_inp_gen_hw.h>
#include <apps_sfdl_gen/pam_clustering_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

pam_clusteringVerifierInpGenHw::pam_clusteringVerifierInpGenHw(Venezia* v_) {
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/pam_clustering_cons.h for constants to use when generating input.
void pam_clusteringVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs) {
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
}
