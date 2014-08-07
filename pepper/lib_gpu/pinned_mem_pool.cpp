#include "pinned_mem_pool.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <assert.h>

static void *alloc_pinned_mem(int size)
{
	void *ret;

	checkCudaErrors(cudaHostAlloc(&ret, size, cudaHostAllocPortable));

	return ret;
}

pinned_mem_pool::~pinned_mem_pool()
{
        if (mem_) {
                cudaFreeHost(mem_);
                mem_ = NULL;
        }
}

bool pinned_mem_pool::init(unsigned long maxsize)
{
        maxsize_ = maxsize;
        mem_ = (uint8_t *) alloc_pinned_mem(maxsize);
	assert(mem_ != NULL);
        return true;
}

void pinned_mem_pool::destroy()
{
	if (mem_) {
		cudaFreeHost(mem_);
		mem_ = NULL;
	}
}
