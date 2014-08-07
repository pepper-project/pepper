#include <iostream>
#include <stdio.h>

#include "lib_gpu/common.h"
#include "test_utils.h"

using namespace std;

int main()
{
    size_t n = 64;
    size_t size = n * sizeof(int);
    
    int *A = (int*)malloc(size);
    int *B = (int*)malloc(size);
    int *C = (int*)malloc(size);

    if (!A || !B || !C) {
        printf("Malloc failed.\n");
        if (A) free(A);
        if (B) free(A);
        if (C) free(A);

        return 1; 
    }

    init(A, n);
    init(B, n);

    vecadd(C, A, B, n);

    for (size_t i = 0; i < n; i++) {
        printf("%d ", C[i]);
        if (i % 10 == 9)
            printf("\n");

        if (C[i] != A[i] + B[i]) {
          cout << "FAILED" << endl;
          exit(1);
        }
    }
    printf("\n");

    free(C);
    free(A);
    free(B);
}

