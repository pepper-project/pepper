#include <apps_sfdl_gen/cql_ro_range_v_inp_gen.h>
#include <apps_sfdl_hw/cql_ro_range_v_inp_gen_hw.h>
#include <apps_sfdl_gen/cql_ro_range_cons.h>
#include <include/avl_tree.h>
#include <storage/ram_impl.h>
#include <storage/hasher.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>
#include <storage/db_util.h>
#include <stdlib.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/cql_ro_range.h>

#define SIZE 1024

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

cql_ro_rangeVerifierInpGenHw::cql_ro_rangeVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/cql_ro_range_cons.h for constants to use when generating input.
void cql_ro_rangeVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  mpq_t* full_db_handle;
  int num_ints = sizeof(Student_handle_t) / sizeof(uint64_t);
  alloc_init_vec(&full_db_handle, num_ints);
  Student_handle_t handle;

  if (generate_states) {
    cout << "generating input" << endl;
    // SIZE should be a power of 2.
    int number_of_rows = SIZE - 1;

    // get the full handle of a DB.
#ifdef USE_DB_BLOCK_STORE
    handle = create_compressed_db(number_of_rows, ("prover_1_" + shared_bstore_file_name).c_str());
#else
    handle = create_db(number_of_rows, ("prover_1_" + shared_bstore_file_name).c_str());
#endif

    uint64_t* input_ptr = (uint64_t*)&handle;
    for(int i = 0; i < num_ints; i++) {
      mpq_set_ui(full_db_handle[i], input_ptr[i], 1);
    }

    dump_vector(num_ints, full_db_handle, "db_handle", FOLDER_PERSIST_STATE);
    cout << "input generated." << endl;
  } else {
    // import the root hash from a place.
    load_vector(num_ints, full_db_handle, "db_handle", FOLDER_PERSIST_STATE);

    uint64_t* input_ptr = (uint64_t*)&handle;
    for(int i = 0; i < num_ints; i++) {
      input_ptr[i] = mpz_get_ui(mpq_numref(full_db_handle[i]));
    }
  }

  struct In input;

  // get a succinct handle of a DB using hashput.
  char db_file_path[BUFLEN];
  snprintf(db_file_path, BUFLEN - 1, "%s/block_stores/prover_1_%s", FOLDER_STATE, shared_bstore_file_name.c_str());
  HashBlockStore* bs = new ConfigurableBlockStore(db_file_path);
  hashput2(bs, &(input.db_handle), &handle);
  delete bs;

  // assign it to input_q
  uint64_t* input_ptr = (uint64_t*)&input;
  for(int i = 0; i < num_inputs; i++) {
    mpq_set_ui(input_q[i], input_ptr[i], 1);
  }

  clear_del_vec(full_db_handle, num_ints);
}

#pragma pack(pop)
