#ifndef COMMON_PAIRING_POW_TABLE_H
#define COMMON_PAIRING_POW_TABLE_H
#include <common/pairing_util.h>

#if NONINTERACTIVE == 1

//Reason for 8: This is the parameter that was decided on in 
//lib_gpu/mp_modexp.h (WINDOWSIZE)

template <class T, int N, int K=8> class pairing_pow_table {
  //Parameters: the table will support powering with an N bit exponent
  //and the cost of an exponentiation will be ceil(N / K) multiplications
  //and the storage space will be 2^K * ceil(N / K)
public:
  pairing_pow_table(){
    k = K;
    num_cols = 1 << K;
    num_rows = (N + K - 1) / K;
    num_cells = (( N + K - 1) / K) * ( 1 << K );
  }
  int k;
  int num_cols;
  int num_rows;
  int num_cells;
  T tbl [ (( N + K - 1) / K) * (1 << K) ];
};

#endif

#endif //COMMON_PAIRING_POW_TABLE_H
