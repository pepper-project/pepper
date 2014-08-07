#include <apps_sfdl_gen/email_history_v_inp_gen.h>
#include <apps_sfdl_hw/email_history_v_inp_gen_hw.h>
#include <apps_sfdl_gen/email_history_cons.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

#include "apps_sfdl/email_history.c"

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

email_historyVerifierInpGenHw::email_historyVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/email_history_cons.h for constants to use when generating input.
void email_historyVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  char db_file_path[BUFLEN];
  snprintf(db_file_path, BUFLEN - 1, "%s/block_stores/%s", FOLDER_STATE,
shared_bstore_file_name.c_str());
  ConfigurableBlockStore bs(db_file_path);

  hash_t hash;
  {
    struct emailDB db;
    uint8_t* db_typepun = (uint8_t*)&db;
    for(uint32_t i = 0; i < sizeof(struct emailDB); i++){
      db_typepun[i] = (uint8_t) rand();
    }

    hashput2(&bs, &hash, &db);
  }

  int i;
  for(i = 0; i < NUM_HASH_CHUNKS; i++){
    mpz_set_ui(mpq_numref(input_q[i]), hash.bit[i]);
  }

  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }
}

#pragma pack(pop)
