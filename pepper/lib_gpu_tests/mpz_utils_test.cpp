#include "lib_gpu/mpz_utils.h"
#include "test_utils.h"

#define ARRY_SIZE 1024
#define NBITS 1024

int main() {
  gmpz_array arry;
  mpz_t array[ARRY_SIZE];
  mpz_t r_array[ARRY_SIZE];

  rand_init(array, ARRY_SIZE, NBITS);
  init(r_array, ARRY_SIZE);

  arry.fromMPZArray(array, ARRY_SIZE);
  arry.toMPZArray(r_array);

  START_TEST("from/to MPZArray");
  RUN_TEST(cmp(array, r_array, ARRY_SIZE));

  arry.fromMPZ(array[0], ARRY_SIZE);
  arry.toMPZArray(r_array);

  for (int i = 1; i < ARRY_SIZE; i++) {
    mpz_set(array[i], array[0]);
  }

  START_TEST("from/to MPZ");
  RUN_TEST(cmp(array, r_array, ARRY_SIZE));

  WORD *a = to_gpu_format(ARRY_SIZE, array, NBITS / sizeof(WORD) / 8);
  arry.fromMPZArray(array, ARRY_SIZE, false);

  assert(arry.arraySize() == ARRY_SIZE * NBITS / 8);

  START_TEST("MPZ to raw bytes.");
  RUN_TEST(cmp((char*)a, (char*)arry.getElemHost(0), arry.arraySize()));
}

