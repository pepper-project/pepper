#include <apps_sfdl_gen/mr_matvec_map_v_inp_gen.h>
#include <apps_sfdl_hw/mr_matvec_map_v_inp_gen_hw.h>
#include <apps_sfdl_gen/mr_matvec_map_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_matvec_mapVerifierInpGenHw::mr_matvec_mapVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/mr_matvec_map_cons.h for constants to use when generating input.
void mr_matvec_mapVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{

  MapperIn mapper_in;
  int i,j,k=0;
  for (i=0; i<NUM_ROWS_PER_MAPPER; i++) {
    mapper_in.vector[i] = rand();
    for (j=0; j < NUM_VARS; j++){
      mapper_in.matrix[i][j] = rand();
    }
  }
  
  hash_t digest;
  export_exo_inputs(&mapper_in, sizeof(MapperIn), &digest);

  digest_to_mpq_vec(input_q, &digest);
}

