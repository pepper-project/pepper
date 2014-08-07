#include "cuda_mem_pool.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>

cuda_mem_pool::~cuda_mem_pool()
{
        if(mem_)
        {
                checkCudaErrors(cudaFree(mem_));
                mem_ = 0;
        }
}

bool cuda_mem_pool::init(unsigned long maxsize)
{
        maxsize_ = maxsize;
        checkCudaErrors(cudaMalloc((void**)&mem_, maxsize));
        return true;
}


void cuda_mem_pool::destroy()
{
	if (mem_) {
		checkCudaErrors(cudaFree(mem_));
		mem_ = NULL;
	}
}
