#include <apps_sfdl_gen/ramput_micro_v_inp_gen.h>
#include <apps_sfdl_hw/ramput_micro_v_inp_gen_hw.h>
#include <apps_sfdl_gen/ramput_micro_cons.h>
#include <storage/ram_impl.h>
#include <storage/hasher.h>
#include <storage/configurable_block_store.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

ramput_microVerifierInpGenHw::ramput_microVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/ramput_micro_cons.h for constants to use when generating input.
void ramput_microVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  ConfigurableBlockStore bs;
  RAMImpl ram(&bs);
  HashType* hash = ram.getRootHash();

  int i = 0;
  for (HashType::HashVec::const_iterator itr = hash->GetFieldElts().begin();
        itr != hash->GetFieldElts().end(); ++itr) {
        mpz_set(mpq_numref(input_q[i]), (*itr).get_mpz_t());
        mpq_canonicalize(input_q[i]);
        i++;
  }
}
