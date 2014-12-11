#include <stdint.h>

#define N 1150
#define K 2300

// sparse matrix * vector multiplication
// N x N matrix with K nonzero elements
//
// input matrix is in CSR representation:
//     elms = <K-length array>
//     inds = <K-length array>
//     ptrs = <N+1-length array>

struct In { int vector[N]; int elms[K]; int inds[K]; int ptrs[N+1]; };
struct Out { int out[N]; };

void compute( struct In *input, struct Out *output ) {
    int i;
    int j;

    for(i = 0; i < N; i++) {
        output->out[i] = 0;
    }

    // N+K-1 trips through the loop
    // in the worst case, the first N-1 rows are empty
    // so we have a useless trip through the loop
    // then the last row would have K multiplications to do
    // total is N + K - 1
    [[buffet::fsm(N + K - 1)]]
    for(i = 0; i < N; i++) {
        int ip0 = input->ptrs[i];
        int ip1 = input->ptrs[i+1];
        for(j = ip0; j < ip1; j++) {
                int iej = input->elms[j];
                int inj = input->inds[j];
                int vij = input->vector[inj];
                output->out[i] += iej * vij;
        }
    }
}
