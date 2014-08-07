#include <iostream>
#include <stdio.h>

#include "lib_gpu/common.h"
#include "test_utils.h"

using namespace std;

int dot_product_local(const int *A, const int *B, size_t n)
{
    int result = 0;
    for (size_t i = 0; i < n; i++) {
        result += A[i] * B[i];
    }
    return result;
}

int main()
{
    size_t n = 64 * BLOCK_SIZE;
    size_t size = n * sizeof(int);
    bool failed = false;
    
    int *A = (int*)malloc(size);
    int *B = (int*)malloc(size);

    if (!A || !B) {
        cout << "Malloc failed." << endl;
        if (A) free(A);
        if (B) free(A);

        return 1; 
    }

    rand_init(A, n);
    rand_init(B, n);

    for (int j = 0; j < 64; j++) {
        printf("Trying n = %05d to %05d (%d * BLOCK_SIZE)\n", j * BLOCK_SIZE, (j + 1) * BLOCK_SIZE - 1, j);
        for (int i = 0; i < BLOCK_SIZE; i++) {
            if (dot_product_local(A, B, i) != dot_product(A, B, i)) {
                printf("Incorrect for %d\n", i);
                failed = true;
            }
        }
    }

    printf("Answer   : %d\n", dot_product(A, B, n));
    printf("Should be: %d\n", dot_product(A, B, n));

    free(A);
    free(B);

    return (failed ? 1 : 0);
}

