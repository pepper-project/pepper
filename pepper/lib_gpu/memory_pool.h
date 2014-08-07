#ifndef __MEM_POOL__
#define __MEM_POOL__
#include <stdint.h>
#define ALIGN(x,a) (((x)+(a)-1)&~((a)-1))

/**
 * class mem_pool
 *
 * memory allocator to share a pool between streams.
 */
class mem_pool
{
public:
	mem_pool() : mem_(0), maxsize_(0), curpos_(0) {}

	virtual ~mem_pool() {}

	/**
	 *
	 *
	 * @param maxsize
	 *
	 * @return
	 */
	virtual bool init(unsigned long maxsize) = 0;

	/**
	 * Allocate the given size of memory.
	 *
	 * @param size size of buffer to allocate.
	 *
	 * @return Starting pointer to the buffer.
	 */
	void * alloc(unsigned long size)
        {
		if(size+curpos_ > maxsize_)
			return 0;
                curpos_ = ALIGN(curpos_, 64);
                void * ret = (void *) (mem_ + curpos_);
                curpos_ += size;
                curpos_ = ALIGN(curpos_, 64);
                return ret;
        }

	/**
	 * Reset buffer pool.
	 * In other words invalidate all allocation from the pool
	 */
        __inline__ void reset()
        {
                curpos_ = 0;
        }

	/**
	 * Retrieve the current position in the pool.
	 * In other words, size of total amount of memory allocated.
	 *
	 * @return Current position in the memory pool.
	 */
        __inline__ unsigned long get_curpos() { return curpos_; }

	/**
	 * Retrieve the starting point of the pool buffer.
	 *
	 * @return Starting point of the pool buffer.
	 */
        __inline__ void* get_mem() { return mem_; }

protected:
        uint8_t * mem_;
        unsigned long maxsize_;
        unsigned long curpos_;
};

#undef ALIGN

#endif
