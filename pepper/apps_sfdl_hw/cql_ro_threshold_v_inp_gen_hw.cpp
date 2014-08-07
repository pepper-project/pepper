#include <apps_sfdl_gen/cql_ro_threshold_v_inp_gen.h>
#include <apps_sfdl_hw/cql_ro_threshold_v_inp_gen_hw.h>
#include <apps_sfdl_gen/cql_ro_threshold_cons.h>
#include <include/avl_tree.h>
#include <storage/ram_impl.h>
#include <storage/hasher.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>
#include <storage/db_util.h>
#include <stdlib.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/cql_ro_threshold.h>

#define SIZE 1024

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

cql_ro_thresholdVerifierInpGenHw::cql_ro_thresholdVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/cql_ro_threshold_cons.h for constants to use when generating input.
void cql_ro_thresholdVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  if (generate_states) {
    // SIZE should be a power of 2.
    int number_of_rows = SIZE - 1;

    struct In input;

    input.handle = create_db(number_of_rows, ("prover_1_" + shared_bstore_file_name).c_str());

    uint64_t* input_ptr = (uint64_t*)&input.handle;
    for(int i = 0; i < num_inputs; i++) {
      mpq_set_ui(input_q[i], input_ptr[i], 1);
    }

    dump_vector(num_inputs, input_q, "db_handle", FOLDER_PERSIST_STATE);
  } else {
    // import the root hash from a place.
    load_vector(num_inputs, input_q, "db_handle", FOLDER_PERSIST_STATE);
  }
}

#pragma pack(pop)
