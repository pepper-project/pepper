#include <apps_sfdl_gen/sparse_matvec_flat_v_inp_gen.h>
#include <apps_sfdl_hw/sparse_matvec_flat_v_inp_gen_hw.h>
#include <apps_sfdl_gen/sparse_matvec_flat_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

sparse_matvec_flatVerifierInpGenHw::sparse_matvec_flatVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/sparse_matvec_flat_cons.h for constants to use when generating input.
using sparse_matvec_flat_cons::N;
using sparse_matvec_flat_cons::K;
void sparse_matvec_flatVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif
  srand(time(NULL));
  // randomly distribute elements into rows
  int buckets[N] = {0,};
  int ptrs[N+1] = {0,};
  // first figure out how many go into each row
  for(int i=0; i < K; i++) {
      int r;
      do {
          r = rand() % N;
      } while (buckets[r] >= N);
      buckets[r]++;
  }
  // then compute the row pointers
  ptrs[0] = 0;
  for(int i=0; i < N; i++) {
      ptrs[i+1] = ptrs[i] + buckets[i];
  }

  int inds[K] = {0,};
  // distribute elemnts within each row
  for(int i=0; i < N; i++) {
      memset(buckets, 0, N * sizeof(int));
      int nelms = ptrs[i+1] - ptrs[i];

      // yes, this is quick and dirty
      while(nelms > 0) {
          int r = rand() % N;
          if (buckets[r] == 0) {
              buckets[r] = 1;
              nelms--;
          }
      }

      int k = ptrs[i];
      for(int j=0; j < N; j++) {
          if (buckets[j] != 0) {
              inds[k++] = j;
          }
      }
  }

  // input_q layout is
  // vector[N] (random values % 1024)
  // elms[K]   (random values % 1024)
  // inds[K]   (computed above)
  // ptrs[N+1] (compuated above)
  for(int i=0; i < N + K; i++) {
      mpq_set_ui(input_q[i], rand() % 1024, 1);
  }
  for(int i=0; i < K; i++) {
      mpq_set_ui(input_q[N+K+i], inds[i], 1);
  }
  for(int i=0; i < N+1; i++) {
      mpq_set_ui(input_q[N+K+K+i], ptrs[i], 1);
  }
}
