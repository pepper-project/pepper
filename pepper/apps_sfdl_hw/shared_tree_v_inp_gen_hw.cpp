#include <apps_sfdl_gen/shared_tree_v_inp_gen.h>
#include <apps_sfdl_hw/shared_tree_v_inp_gen_hw.h>
#include <apps_sfdl_gen/shared_tree_cons.h>
#include <storage/ram_impl.h>
#include <storage/hasher.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>
#include <include/binary_tree_int_hash_t.h>
#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

shared_treeVerifierInpGenHw::shared_treeVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/shared_tree_cons.h for constants to use when generating input.
void shared_treeVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  char db_file_path[BUFLEN];
  snprintf(db_file_path, BUFLEN - 1, "%s/block_stores/%s", FOLDER_STATE, shared_bstore_file_name.c_str());
  HashBlockStore* bs = new ConfigurableBlockStore(db_file_path);
  MerkleRAM* ram = new RAMImpl(bs);
  setBlockStoreAndRAM(bs, ram);

  tree_t Age_index;
  uint32_t tempAge;
  hash_t tempHash;

  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  tree_init(&Age_index);
  tempAge = 15;
  hashput(&tempHash, &tempAge);
  tree_insert(&Age_index, tempAge, tempHash);
  for(int i = 0; i < num_inputs; i++) {
    //gmp_printf("hash verifier: %d %lX\n", i, Age_index.root.bit[i]);
    mpq_set_ui(input_q[i], Age_index.root.bit[i], 1);
  }

  deleteBlockStoreAndRAM();
}

#pragma pack(pop)
