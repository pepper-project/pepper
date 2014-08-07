#ifndef __CUDA_MEMORY_POOL__
#define __CUDA_MEMORY_POOL__
#include "memory_pool.h"

/**
 * class cuda_mem_pool
 *
 * memory pool for device memory.
 */
class cuda_mem_pool : public mem_pool
{
public:
        virtual ~cuda_mem_pool();

	/**
	 * Allocate the pool buffer in the device memory.
	 *
	 * @param maxsize Total amount of buffer for the pool.
	 *
	 * @return true if successful, false otherwise.
	 */
        virtual bool init(unsigned long maxsize);

	/**
	 * Free allocated memory.
	 *
	 */
	virtual void destroy();
};


#endif
