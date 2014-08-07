#include <apps_sfdl_gen/tree_balance_v_inp_gen.h>
#include <apps_sfdl_hw/tree_balance_v_inp_gen_hw.h>
#include <apps_sfdl_gen/tree_balance_cons.h>
#include <include/avl_tree.h>
#include <storage/ram_impl.h>
#include <storage/hasher.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>
#include <storage/db_util.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/tree_balance.h>

#define SIZE 1024

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

tree_balanceVerifierInpGenHw::tree_balanceVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/tree_balance_cons.h for constants to use when generating input.
void tree_balanceVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
    // SIZE should be a power of 2.
    int number_of_rows = SIZE - 1;
    struct In input;
    Student_handle_t handle = create_db(number_of_rows, ("prover_1_" + shared_bstore_file_name).c_str());

    input.root = handle.KEY_index;
    int number_of_hash_elements = sizeof(hash_t) / sizeof(uint64_t);
    uint64_t* input_ptr = (uint64_t*)&input.root;
    for(int i = 0; i < number_of_hash_elements; i++) {
      mpq_set_ui(input_q[i], input_ptr[i], 1);
    }
    FILE* path_file = fopen(FOLDER_PERSIST_STATE "/tree_path", "r");
    int path, path_depth;
    if (path_file != NULL) {
      fscanf(path_file, "%d %d", &path, &path_depth);
    } else {
      path_depth = 1;
      path = 0;
    }
    mpq_set_ui(input_q[number_of_hash_elements], path_depth, 1);
    mpq_set_ui(input_q[number_of_hash_elements + 1], path, 1);

    //dump_vector(num_inputs, input_q, "db_handle", FOLDER_PERSIST_STATE);
  } else {
    // import the root hash from a place.
    int number_of_hash_elements = sizeof(hash_t) / sizeof(uint64_t);
    load_vector(number_of_hash_elements, input_q, "db_handle", FOLDER_PERSIST_STATE);

    FILE* path_file = fopen(FOLDER_PERSIST_STATE "/tree_path", "r");
    int path, path_depth;
    if (path_file != NULL) {
      fscanf(path_file, "%d %d", &path, &path_depth);
    } else {
      path_depth = 1;
      path = 0;
    }
    mpq_set_ui(input_q[number_of_hash_elements], path_depth, 1);
    mpq_set_ui(input_q[number_of_hash_elements + 1], path, 1);
  }
}
