#include <cuda.h>
#include <helper_cuda.h>

#include "common.h"

__global__ void vecadd_kernel(int *C, int *A, int *B, size_t n)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
        C[i] = A[i] + B[i];
}

void vecadd(int *C, const int *A, const int *B, size_t n)
{
    if (n == 0)
        return;

    int size = sizeof(int) * n;
    int *d_A, *d_B;
    checkCudaErrors(cudaMalloc(&d_A, size));
    checkCudaErrors(cudaMalloc(&d_B, size));

    checkCudaErrors(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = BLOCK_SIZE;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vecadd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_A, d_B, n);

    checkCudaErrors(cudaMemcpy(C, d_A, size, cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
}

