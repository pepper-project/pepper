#include <common/measurement.h>
#include <common/utility.h>
#include <crypto/crypto.h>
#include <include/avl_tree.h>
#include <storage/configurable_block_store.h>
#include <storage/exo.h>
#include <storage/ggh_hash.h>
#include <storage/hash_block_store.h>
#include <storage/hasher.h>
#include <storage/ram_impl.h>

#define BUFLEN 10240
#define NUM_SAMPLES 10
#define MICROBENCHMARKS 1

// global state
Crypto *crypto;

double do_hash(Hasher* hasher, const Bits& input) {
  Measurement m;

  m.begin_with_init();
  for (int i = 0; i < 1000; i++) {
    hasher->hash(input);
  }
  m.end();

  return m.get_ru_elapsed_time() / 1000;
}

void measure_hash_ops(int bits) {
  Bits input;
  input.resize(bits);
  for(int i = 0; i < bits; i++) {
    input[i] = i % 2;
  }
  Hasher * hasher = new GGHHash();

  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++)
    measurements[i] = do_hash(hasher, input);

  snprintf(scratch_str, BUFLEN-1, "ggh_hash_%d", bits);
  print_stats(scratch_str, measurements);
}

double do_storage(HashBlockStore* store, HashBlockStore::Key& key, HashBlockStore::Value& block) {
  Measurement m;

  m.begin_with_init();
  for (int i = 0; i < 1000; i++) {
    store->put(key, block);
  }
  m.end();

  return m.get_ru_elapsed_time() / 1000;
}

void measure_storage_ops(int bits) {
  HashBlockStore::Key key;
  HashBlockStore::Value block;
  key.resize(768);
  block.resize(bits);
  for (int i = 0; i < 768; i++)
    key[i] = i % 2;
  for (int i = 0; i < bits; i++)
    block[i] = i % 2;

  HashBlockStore* store = new ConfigurableBlockStore();

  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++)
    measurements[i] = do_storage(store, key, block);

  snprintf(scratch_str, BUFLEN-1, "block_store_%d", bits);
  print_stats(scratch_str, measurements);
}

double do_hashput(HashBlockStore* store, void* data, int bytes) {
  Measurement m;
  hash_t hash;

  m.begin_with_init();
  for (int i = 0; i < 1000; i++) {
    crypto->get_randomb_priv((char*)data, bytes);
    __hashput(store, &hash, data, bytes);
  }
  m.end();

  return m.get_ru_elapsed_time() / 1000;
}

void measure_hashput_ops(int bits) {
  int bytes = bits / 8;
  HashBlockStore* store = new ConfigurableBlockStore();
  char* data = new char[bytes];

  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i=0; i<NUM_SAMPLES; i++)
    measurements[i] = do_hashput(store, (void*)data, bytes);

  snprintf(scratch_str, BUFLEN-1, "exo_hashput_%d", bits);
  print_stats(scratch_str, measurements);
  delete[] data;
}

double do_hashget(HashBlockStore* store, hash_t* hashes, int bytes) {
  Measurement m;
  char* data = new char[bytes];

  m.begin_with_init();
  for (int i = 0; i < 1000; i++) {
    __hashget(store, (void*)data, &(hashes[i]), bytes);
  }
  m.end();

  delete[] data;
  return m.get_ru_elapsed_time() / 1000;
}

void measure_hashget_ops(int bits) {
  int bytes = bits / 8;
  HashBlockStore* store = new ConfigurableBlockStore();
  char* data = new char[bytes];
  hash_t hashes[1000];

  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  for (int i = 0; i < 1000; i++) {
    crypto->get_randomb_priv((char*)data, bytes);
    __hashput(store, &(hashes[i]), (void*)data, bytes);
  }

  for (int i=0; i<NUM_SAMPLES; i++) {
    measurements[i] = do_hashget(store, hashes, bytes);
  }

  snprintf(scratch_str, BUFLEN-1, "exo_hashget_%d", bits);
  print_stats(scratch_str, measurements);
  delete[] data;
}

int inorderTraversal(int input[], int output[], int rootIndex, int i, int subtreeSize) {
  if (subtreeSize < 0) {
    return i;
  }
  subtreeSize = (subtreeSize + 1) / 2 - 1;
  int leftIndex = rootIndex - subtreeSize - 1;
  int rightIndex = rootIndex + subtreeSize + 1;
  output[i] = input[rootIndex];
  i++;
  i = inorderTraversal(input, output, leftIndex, i, subtreeSize);
  i = inorderTraversal(input, output, rightIndex, i, subtreeSize);
  return i;
}

// n needs to be 2^k - 1
void generateInOrderBinaryTreeTraversal(int output[], int n) {
  int* input = new int[n];
  for (int i = 0; i < n; i++) {
    input[i] = i;
  }
  inorderTraversal(input, output, (n + 1) / 2 - 1, 0, (n + 1) / 2 - 1);
  delete[] input;
}

double do_tree_insert(tree_t* tree, int* keys, hash_t* values) {
  Measurement m;

  m.begin_with_init();
  for (int i = 0; i < 1000; i++) {
    //cout << "Tree insertion." << endl;
    int success = tree_insert(tree, keys[i], values[i]);
    if (!success) {
      cout << "Error during exogenous tree insertion. Please make sure the binary tree is big enough to hold the node." << endl;
    }
  }
  m.end();

  return m.get_ru_elapsed_time() / 1000;
}

void measure_tree_insert_ops() {
  int numberOfRows = 1024;
  int* index = new int[numberOfRows];
  hash_t* values = new hash_t[numberOfRows];
  generateInOrderBinaryTreeTraversal(index, numberOfRows);

  vector<double> measurements(NUM_SAMPLES, 0);
  char scratch_str[BUFLEN];

  tree_t tree;

  for (int i=0; i<NUM_SAMPLES; i++) {
    HashBlockStore* store = new ConfigurableBlockStore();
    MerkleRAM* ram = new RAMImpl(store);
    setBlockStoreAndRAM(store, ram);
    tree_init(&tree);

    measurements[i] = do_tree_insert(&tree, index, values);

    deleteBlockStoreAndRAM();
  }

  snprintf(scratch_str, BUFLEN-1, "exo_tree_insert");
  print_stats(scratch_str, measurements);
  delete[] index;
  delete[] values;
}

void pressure_hashput(HashBlockStore* store, void* data, int bits) {
  int bytes = bits / 8;
  Measurement m;
  hash_t hash;
  char scratch_str[BUFLEN];
  snprintf(scratch_str, BUFLEN-1, "presure_hashput_%d", bits);

  m.begin_with_init();
  for (int i = 0; i < 1000; i++) {
    crypto->get_randomb_priv((char*)data, bits);
    __hashput(store, &hash, data, bytes);
  }
  m.end();

  printf("%s: %lf\n", scratch_str, m.get_ru_elapsed_time() / 1000);
  printf("%s_latency: %lf\n\n", scratch_str, m.get_papi_elapsed_time() / 1000);
}

void pressure_test_hashput_ops(int bits) {
  int bytes = bits / 8;
  HashBlockStore* store = new ConfigurableBlockStore();
  char* data = new char[bytes];

  for(int i = 0; i < 10000; i++) {
    pressure_hashput(store, (void*)data, bits);
  }
  delete[] data;
}

void pressure_file_io(const char* folder, const char* buf, int bytes) {
  Measurement m;
  char scratch_str[BUFLEN];
  char file_path[BUFLEN];
  FILE *fp;

  snprintf(scratch_str, BUFLEN-1, "presure_file_io_%d", bytes * 8);

  m.begin_with_init();
  for (int i = 0; i < 10000; i++) {
    snprintf(file_path, BUFLEN-1, "%s/b%lld_%d", folder, PAPI_get_real_nsec(), i);
    fp = fopen(file_path, "w");
    fwrite(buf, 1, bytes, fp);
    fclose(fp);
  }
  m.end();

  printf("%s: %lf\n", scratch_str, m.get_ru_elapsed_time() / 10000);
  printf("%s_latency: %lf\n\n", scratch_str, m.get_papi_elapsed_time() / 10000);
}

void pressure_test_file_io_ops(int bits) {
  int bytes = bits / 8;
  char scratch_str[BUFLEN];
  char *buf = new char[bytes];
  snprintf(scratch_str, BUFLEN-1, "temp_folder_%d", PAPI_get_real_nsec());
  mkdir(scratch_str, S_IRWXU);

  for(int i = 0; i < 10000; i++) {
    crypto->get_randomb_priv(buf, bits);
    pressure_file_io(scratch_str, buf, bytes);
  }
  delete[] buf;
}

int main(int argc, char **argv) {
  crypto = new Crypto(CRYPTO_TYPE_PRIVATE, CRYPTO_ELGAMAL, PNG_CHACHA, false, 128, 1024, 160);
  //measure_hash_ops(768);
  //measure_hash_ops(1024);
  //measure_hash_ops(4096);
  //measure_storage_ops(768);
  //measure_storage_ops(1024);
  //measure_storage_ops(4096);
  //measure_hashput_ops(768);
  //measure_hashput_ops(1024);
  //measure_hashput_ops(4096);
  //measure_hashget_ops(768);
  //measure_hashget_ops(1024);
  //measure_hashget_ops(4096);

  // 2344 is the size of a tree_node_t struct of a int_hash_t binary tree.
  //measure_hash_ops(2344);
  //measure_storage_ops(2344);
  measure_hashput_ops(2344);
  //measure_hashget_ops(2344);
  //measure_tree_insert_ops();

  pressure_test_hashput_ops(2344);
  //pressure_test_file_io_ops(768);

  delete crypto;
  return 0;
}
