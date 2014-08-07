#include <apps_sfdl_gen/mr_f2_map_v_inp_gen.h>
#include <apps_sfdl_hw/mr_f2_map_v_inp_gen_hw.h>
#include <apps_sfdl_gen/mr_f2_map_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_f2_mapVerifierInpGenHw::mr_f2_mapVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/mr_f2_map_cons.h for constants to use when generating input.
void mr_f2_mapVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  mpz_t *exo_input;
  alloc_init_vec(&exo_input, SIZE_INPUT);

  v->get_random_vec_priv(SIZE_INPUT, exo_input, 32);
  MapperIn mapper_in;
  for (int i=0; i<SIZE_INPUT; i++) {
    mapper_in.vec[i] = mpz_get_ui(exo_input[i]);
  }

  hash_t digest;
  export_exo_inputs(&mapper_in, sizeof(MapperIn), &digest);

  for (int i=0; i<num_inputs; i++) {
    mpz_set_ui(mpq_numref(input_q[i]), digest.bit[i]);
  }
  clear_vec(SIZE_INPUT, exo_input);
}
