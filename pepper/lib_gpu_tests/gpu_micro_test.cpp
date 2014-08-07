#include <cassert>
#include <iostream>
#include <cuda_runtime_api.h>
#include <ctime>
#include "lib_gpu/mpz_utils.h"
#include "lib_gpu_tests/test_utils.h"
#include <common/measurement.h>

#define NBITS 1024
#define NBITS_EXP 192 
#define ARRY_CHUNK_SIZE 32
#define ARRY_TOTAL_SIZE (32*4)

using namespace std;

int main()
{
  Measurement m;
  int size = ARRY_TOTAL_SIZE;
  for (int i=0; i<10; i++) { 
    size = 2 * size;
    mpz_t A[size];
    gmpz_array g_A;
    rand_init(A, ARRY_TOTAL_SIZE, NBITS);
    
    m.begin_with_init();
    g_A.fromMPZArray(A, ARRY_TOTAL_SIZE, true, 1);
    m.end();
    cout<<"Array transfer to GPU ("<<size<<"): "<<(m.get_papi_elapsed_time())<<endl;
  }
  
  return 0;
}
