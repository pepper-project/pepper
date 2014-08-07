#include <apps_sfdl_gen/pure_hashget_v_inp_gen.h>
#include <apps_sfdl_hw/pure_hashget_v_inp_gen_hw.h>
#include <apps_sfdl_gen/pure_hashget_cons.h>
#include <storage/configurable_block_store.h>
#include <storage/ram_impl.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/pure_hashget.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

pure_hashgetVerifierInpGenHw::pure_hashgetVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/pure_hashget_cons.h for constants to use when generating input.
void pure_hashgetVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  struct In input;
  struct Out output;
  cout << "creating input..." << endl;
  char db_file_path[BUFLEN];
  snprintf(db_file_path, BUFLEN - 1, "%s/block_stores/prover_1_%s", FOLDER_STATE, shared_bstore_file_name.c_str());
  HashBlockStore* bs = new ConfigurableBlockStore(db_file_path);
  MerkleRAM* ram = new RAMImpl(bs);
  setBlockStoreAndRAM(bs, ram);

  srand(time(NULL));
  for (int i = 0; i < NUM_OF_BLOCKS; i++) {
    for (int j = 0; j < BLOCKLEN; j++) {
      output.blocks[i].block[j] = rand();
    }
    hashput(&(input.hashes[i]), &(output.blocks[i]));
    hashget(&(output.blocks[i]), &(input.hashes[i]));
  }
  uint64_t* input_ptr = (uint64_t*)&input;
  for (int i = 0; i < num_inputs; i++) {
    mpq_set_ui(input_q[i], input_ptr[i], 1);
  }
  deleteBlockStoreAndRAM();
}

#pragma pack(pop)
