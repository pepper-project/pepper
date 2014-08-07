#include <apps_sfdl_gen/shared_store_v_inp_gen.h>
#include <apps_sfdl_hw/shared_store_v_inp_gen_hw.h>
#include <apps_sfdl_gen/shared_store_cons.h>
#include <storage/ram_impl.h>
#include <storage/hasher.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>

#pragma pack(push)
#pragma pack(1)
//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

shared_storeVerifierInpGenHw::shared_storeVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/share_store_cons.h for constants to use when generating input.
void shared_storeVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  char db_file_path[BUFLEN];
  snprintf(db_file_path, BUFLEN - 1, "%s/block_stores/%s", FOLDER_STATE, shared_bstore_file_name.c_str());
  HashBlockStore* bs = new ConfigurableBlockStore(db_file_path);
  RAMImpl* ram = new RAMImpl(bs);

  {
    hash_t hash;
    uint32_t x = 15;

    hashput2(bs, &hash, &x);
    ramput2(ram, 0, &hash);
    x = 20329;
    ramput2(ram, 1, &x);
  }
  HashType* hash = ram->getRootHash();

  int i = 0;
  for (HashType::HashVec::const_iterator itr = hash->GetFieldElts().begin();
        itr != hash->GetFieldElts().end(); ++itr) {
        mpz_set(mpq_numref(input_q[i]), (*itr).get_mpz_t());
        mpq_canonicalize(input_q[i]);
        i++;
  }
  delete ram;
  delete bs;
}

#pragma pack(pop)
