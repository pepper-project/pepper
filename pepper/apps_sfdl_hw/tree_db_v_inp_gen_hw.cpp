#include <apps_sfdl_gen/tree_db_v_inp_gen.h>
#include <apps_sfdl_hw/tree_db_v_inp_gen_hw.h>
#include <apps_sfdl_gen/tree_db_cons.h>
#include <storage/ram_impl.h>
#include <storage/hasher.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>
#include <include/binary_tree_int_hash_t.h>
#pragma pack(push)
#pragma pack(1)


//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

tree_dbVerifierInpGenHw::tree_dbVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/tree_db_cons.h for constants to use when generating input.
void tree_dbVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  char db_file_path[BUFLEN];
  snprintf(db_file_path, BUFLEN - 1, "%s/block_stores/%s", FOLDER_STATE, shared_bstore_file_name.c_str());
  HashBlockStore* bs = new ConfigurableBlockStore(db_file_path);
  MerkleRAM* ram = new RAMImpl(bs);
  setBlockStoreAndRAM(bs, ram);

  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif
  tree_t Age_index;
  Student_t tempStudent;
  hash_t tempHash;

  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  // need to find a way to let the tree push to the correct block store.
  tree_init(&Age_index);

  tempStudent.KEY = 1;
  tempStudent.Age = 20;
  tempStudent.FName = 1298384231432L;
  tempStudent.LName = 2380943023039L;
  tempStudent.Major = 30;
  tempStudent.State = 12;
  tempStudent.PhoneNum = 512800;
  tempStudent.Credits = 100;
  tempStudent.Average = 89;
  tempStudent.Class = 2009;
  hashput(&tempHash, &tempStudent);
  tree_insert(&Age_index, tempStudent.Age, tempHash);

  tempStudent.KEY = 2;
  tempStudent.Age = 21;
  tempStudent.FName = 1298384231432L;
  tempStudent.LName = 2380943023039L;
  tempStudent.Major = 34;
  tempStudent.State = 15;
  tempStudent.PhoneNum = 619800;
  tempStudent.Credits = 101;
  tempStudent.Average = 90;
  tempStudent.Class = 2009;
  hashput(&tempHash, &tempStudent);
  tree_insert(&Age_index, tempStudent.Age, tempHash);

  for(int i = 0; i < num_inputs; i++) {
    //gmp_printf("hash verifier: %d %lX\n", i, Age_index.root.bit[i]);
    mpq_set_ui(input_q[i], Age_index.root.bit[i], 1);
  }

  deleteBlockStoreAndRAM();
}
#pragma pack(pop)
