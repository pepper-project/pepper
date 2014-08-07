#ifndef __PINNED_MEM_POOL__
#define __PINNED_MEM_POOL__
#include "memory_pool.h"

/**
 * class pinned_mem_pool
 *
 * memory pool for pinned page.
 */
class pinned_mem_pool : public mem_pool
{
public:
        virtual ~pinned_mem_pool();
	/**
	 * Allocate the pool buffer in the pinned page.
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
        __inline__ void store_pointer(uint8_t * pos) { curpointer_ = pos; }
        __inline__ void* get_stored_pointer() { return (void *) curpointer_; }
        __inline__ void store_pos() { pos_ = curpos_; }
        __inline__ void* get_stored_pos() { return (void *) ((uint8_t*) mem_ + pos_);}
private:
        unsigned long pos_;
        uint8_t * curpointer_;
};

#endif
