#include <common/measurement.h>
#include <common/utility.h>
#include <storage/db_util.h>
#include <storage/exo.h>
#include <storage/external_sort.h>

//#define MAX_TREE_LEVEL 2
//#define NUM_OF_SUBTREE (1 << MAX_TREE_LEVEL)

//#define NUM_OF_CHUNKS 4

enum stage {
  SETUP = 0,
  PREPARE = 1,
  GENERATE_ROWS = 2,
  SORT_ROWS_KV_PAIRS = 3,
  SORT_INDEX = 4,
  PARITION_TREES = 5,
  BUILD_TREE = 6,
  MERGE_TREES = 7,
  SORT_TREE = 8,
  MERGE_KV_PAIRS = 9,
  GET_DB_HANDLE = 10,
  TEST = 11,
  BUILD_AND_MERGE_TREE = 12,
  CONVERT_DB_HANDLE = 13,
};

int main(int argc, char **argv) {
  omp_set_dynamic(0);
  omp_set_num_threads(16);
  int stage = atoi(argv[1]);
  switch(stage) {
    case SETUP:
      {
        cout << "setup" << endl;
        const char* bstore_file_name = argv[2];
        setup(FOLDER_STATE, bstore_file_name);
        break;
      }
    case PREPARE:
      {
        cout << "preparing" << endl;
        int number_of_rows = atoi(argv[2]);
        const char* filename = argv[3];
        int seed = atoi(argv[4]);
        srand(seed);
        generate_random_permutation_to_file(number_of_rows, filename, FOLDER_TMP);
        break;
      }
    case GENERATE_ROWS:
      {
        cout << "generating rows" << endl;
        int starting_row_id = atoi(argv[2]);
        int chunk_id = atoi(argv[3]);
        int number_of_entries_in_chunk = atoi(argv[4]);
        int entry_size = sizeof(hash_t) + sizeof(Student_t);
        generate_one_chunk_of_db_row(starting_row_id, chunk_id, number_of_entries_in_chunk, entry_size, FOLDER_TMP, FOLDER_STATE);
        break;
      }
    case SORT_ROWS_KV_PAIRS:
      {
        cout << "sorting rows" << endl;
        int number_of_chunks = atoi(argv[2]);
        int entry_size = sizeof(hash_t) + sizeof(Student_t);
        external_sort("hash_to_rows", FOLDER_TMP, number_of_chunks, entry_size, hash_t_comparator, false);
        break;
      }
    case SORT_INDEX:
      {
        cout << "sorting indices" << endl;
        const char* filename = argv[2];
        int number_of_chunks = atoi(argv[3]);
        int entry_size = sizeof(int) + sizeof(hash_t);
        external_sort(filename, FOLDER_TMP, number_of_chunks, entry_size, int_comparator, false);
        break;
      }
    case PARITION_TREES:
      {
        cout << "paritioning trees" << endl;
        const char* tree_name = argv[2];
        int tree_size = atoi(argv[3]);
        int max_tree_level = atoi(argv[4]);
        partition_tree_to_file(tree_name, FOLDER_TMP, tree_size, max_tree_level);
        break;
      }
    case BUILD_TREE:
      {
        cout << "building tree" << endl;
        const char* tree_name = argv[2];
        int subtree_id = atoi(argv[3]);
        const char* kv_pairs_filename = argv[4];
        build_one_subtree_bottom_up(tree_name, subtree_id, kv_pairs_filename, FOLDER_TMP);
        break;
      }
    case MERGE_TREES:
      {
        cout << "merging trees" << endl;
        const char* tree_name = argv[2];
        const char* kv_pairs_filename = argv[3];
        int number_of_subtrees = atoi(argv[4]);
        merge_trees(tree_name, kv_pairs_filename, FOLDER_TMP, number_of_subtrees);
        break;
      }
    case SORT_TREE:
      {
        cout << "sorting tree" << endl;
        const char* tree_name = argv[2];
        int number_of_chunks = atoi(argv[3]);
        int entry_size = sizeof(hash_t) + sizeof(tree_node_t);
        for (int i = 0; i < number_of_chunks; i++) {
          cout << "sorting subtree " << i << endl;
          char subtree_name[BUFLEN];
          snprintf(subtree_name, BUFLEN - 1, "%s_%d", tree_name, i);
          if (i == number_of_chunks - 1) {
            external_sort(subtree_name, FOLDER_TMP, 1, entry_size, hash_t_comparator, true);
          } else {
            external_sort(subtree_name, FOLDER_TMP, 8, entry_size, hash_t_comparator, true);
          }
          cout << "subtree " << i << " sorted" << endl;
        }
        //external_sort(tree_name, FOLDER_TMP, number_of_chunks, entry_size, hash_t_comparator, false);
        merge_sorted_chunks(tree_name, FOLDER_TMP, number_of_chunks, entry_size, hash_t_comparator);
        break;
      }
    case MERGE_KV_PAIRS:
      {
        cout << "merging" << endl;
        const char* bstore_file_name = argv[2];
        int number_of_chunks = atoi(argv[3]);
        merge_into_one_db(FOLDER_STATE, number_of_chunks, bstore_file_name);
        break;
      }
    case GET_DB_HANDLE:
      {
        cout << "getting DB handle" << endl;
        Student_handle_t handle;
        memset(&handle, 0, sizeof(Student_handle_t));
        // read each DB handle from disk
        //load_array((char*)&(handle.KEY_index), sizeof(hash_t), "key_tree_handle", FOLDER_TMP);
        load_array((char*)&(handle.Average_index), sizeof(hash_t), "average_tree_handle", FOLDER_TMP);
        load_array((char*)&(handle.Class_index), sizeof(hash_t), "class_tree_handle", FOLDER_TMP);
        load_array((char*)&(handle.Age_index), sizeof(hash_t), "age_tree_handle", FOLDER_TMP);
        // output the handle to db_handle
        dump_array((char*)&handle, sizeof(Student_handle_t), "db_handle_raw", FOLDER_PERSIST_STATE);
        break;
      }
    case TEST:
      {
        cout << "testing" << endl;
        int number_of_rows = atoi(argv[2]);
        const char* bstore_file_name = argv[3];
        Student_handle_t handle;
        load_array((char*)&handle, sizeof(Student_handle_t), "db_handle_raw", FOLDER_PERSIST_STATE);

        test_trees(handle, number_of_rows, FOLDER_STATE, bstore_file_name);
        break;
      }
    case BUILD_AND_MERGE_TREE:
      {
        cout << "building trees" << endl;
        const char* tree_name = argv[2];
        const char* filename = argv[3];
        int number_of_entries = atoi(argv[4]);
        int max_tree_level = atoi(argv[5]);
        int entry_size = sizeof(hash_t) + sizeof(tree_node_t);
        tree_t tree;
        int number_of_chunks = build_tree_bottom_up(&tree, tree_name, filename, FOLDER_TMP, number_of_entries, max_tree_level);
        external_sort(tree_name, FOLDER_TMP, number_of_chunks, entry_size, hash_t_comparator, false);
        // dump the hash to disk.
        char hash_filename[BUFLEN];
        snprintf(hash_filename, BUFLEN - 1, "%s_handle", tree_name);
        dump_array((char*)&(tree.root), sizeof(hash_t), hash_filename, FOLDER_TMP);
        break;
      }
    case CONVERT_DB_HANDLE:
      {
        cout << "converting DB handle to GMP format " << endl;
        Student_handle_t handle;
        load_array((char*)&handle, sizeof(Student_handle_t), "db_handle_raw", FOLDER_PERSIST_STATE);
        mpq_t* input_q;
        int num_inputs = sizeof(Student_handle_t) / sizeof(uint64_t);
        alloc_init_vec(&input_q, num_inputs);
        uint64_t* input_ptr = (uint64_t*)&handle;
        for(int i = 0; i < num_inputs; i++) {
          mpq_set_ui(input_q[i], input_ptr[i], 1);
        }
        dump_vector(num_inputs, input_q, "db_handle", FOLDER_PERSIST_STATE);
        clear_del_vec(input_q, num_inputs);
        break;
      }
    default:
      break;
  }
  return 0;
}
