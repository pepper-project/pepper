#ifndef EXAMPLES_COMMON_H
#define EXAMPLES_COMMON_H
#include <stdlib.h>

#define BLOCK_SIZE 256

int dot_product(const int *A, const int *B, size_t n);
void vecadd(int *C, const int *A, const int *B, size_t n);

#endif
