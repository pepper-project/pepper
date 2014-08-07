#include <apps_sfdl_gen/ram_hybrid_micro_v_inp_gen.h>
#include <apps_sfdl_hw/ram_hybrid_micro_v_inp_gen_hw.h>
#include <apps_sfdl_gen/ram_hybrid_micro_cons.h>
#include <storage/ram_impl.h>
#include <storage/hasher.h>
#include <storage/configurable_block_store.h>


//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

ram_hybrid_microVerifierInpGenHw::ram_hybrid_microVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/ram_hybrid_micro_cons.h for constants to use when generating input.
void ram_hybrid_microVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif
  
  // setup the Merkle root when the hybrid choose Merkle-tree based
  // solution
  if (num_inputs > 1) {
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

  // set the address for RAM operation to be 0
  mpq_set_ui(input_q[num_inputs-1], 0, 1);
  
  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }
}
