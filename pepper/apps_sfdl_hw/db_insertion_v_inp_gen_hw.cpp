#include <apps_sfdl_hw/db_insertion_v_inp_gen_hw.h>
#include <storage/configurable_block_store.h>
#include <storage/ram_impl.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.
db_insertionVerifierInpGenHw::db_insertionVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/db_insertion_cons.h for constants to use when generating input.
void db_insertionVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs) {
  //Default implementation is provided by compiler
//  compiler_implementation.create_input(input_q, num_inputs);

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

  // the address to get and put in the database
  mpq_set_ui(input_q[i], 0, 1);
  i++;
  // the value to put
//  mpq_set_ui(input_q[i], 42, 1);
}
