#include <apps_sfdl_gen/mr_cov_map_v_inp_gen.h>
#include <apps_sfdl_hw/mr_cov_map_v_inp_gen_hw.h>
#include <apps_sfdl_gen/mr_cov_map_cons.h>
#include <apps_sfdl/mr_cov.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_cov_mapVerifierInpGenHw::mr_cov_mapVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/mr_cov_map_cons.h for constants to use when generating input.
void mr_cov_mapVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{  
  MapperIn mapper_in;
  for (int i=0; i<NUM_DATAPOINTS_PER_MAPPER; i++) {
    for(int j=0; j<NUM_VARS; j++){
      mapper_in.data[i][j] = rand();
    } 
  }

  hash_t digest;
  export_exo_inputs(&mapper_in, sizeof(MapperIn), &digest);

  digest_to_mpq_vec(input_q, &digest);
}
