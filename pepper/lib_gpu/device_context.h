#ifndef __DEVICE_CONTEXT__
#define __DEVICE_CONTEXT__

#include "pinned_mem_pool.h"
#include "cuda_mem_pool.h"

#include <cuda_runtime.h>
#include <assert.h>

#define MAX_STREAM 16
#define MAX_BLOCKS 8192

/**
 * Enum for stream state
 **/
enum STATE {
	READY,
	WAIT_KERNEL,
	WAIT_COPY,
};


struct stream_context{
	cuda_mem_pool pool;
	cudaStream_t stream;

	STATE state;
	bool finished;

	uint8_t *checkbits;
	uint8_t *checkbits_d;
	unsigned int num_blks;

	uint64_t begin_usec;
	uint64_t end_usec;
};

/**
 * device_context class.
 *
 * help managing cuda stream state and memory allocation
 **/
class device_context{
public:
	device_context();
	~device_context();

	/**
	 * Initialize device context. This function will allocate device memory and streams.
	 *
	 * @param size Total amount of memory per stream
	 * @param nstream Number of streams to use. Maximum 16 and minimum is 0.
         *                Use 0 if you want use default stream only.
	 *                Be aware that using a single stream and no stream is different.
	 *                For more information on how they differ, refer to CUDA manual.
	 * @return true when succesful, false otherwise
	 **/
	bool init(const unsigned long size, const unsigned nstream);

	/**
	 * Check whether current operation has finished.
	 *
	 * @param stream_id Index of stream. If initilized to 0 stream then 0,
	 *                  otherwise from 1 to number of streams initialized.
	 * @param block Wait for current operation to finish. true by default
	 * @return true if stream is idle, and false if data copy or execution is in progress
	 **/
	bool sync(const unsigned stream_id, const bool block=true);

	/**
	 * Set the state of the stream. States indicates what is the current operation on-going.
	 *
	 * @param stream_id Index of stream. If initilized to 0 stream then 0, otherwise from 1 to number of streams initialized.
	 * @param state Describe the current state of stream. Currently there are three different states: READY, WAIT_KERNEL, WAIT_COPY.
	 **/
	void set_state(const unsigned stream_id, const STATE state);

	/**
	 * retrieve the state of the stream
	 *
	 * @param stream_id Index of stream. If initilized to 0 stream then 0, otherwise from 1 to number of streams initialized.
	 * @return Current state of the stream
	 **/
	STATE get_state(const unsigned stream_id);

	/**
	 * Retreive the buffer storing the kernel execution finish check bits.
	 *
	 * This buffer is used in sync function to check whether the kernel has finished or not.
	 * Executed kernel will set this each byte to 1 at the end of execution.
	 * We use checkbits instead of cudaStreamSynchronize because
	 * Calling cudaStreamSynchronize to a stream will block other streams
	 * from launching new kernels and it won't return before
	 * all previos launched kernel start execution according to CUDA manual 3.2.
	 * We use host mapped memory to avoid calling cudaStreamSyncrhonize and
	 * check whether kernel execution has finished or not so that
	 * multiple stream can run simultaneously without blocking others.
	 *
	 * @param stream_id Index of stream. If initilized to 0 stream then 0, otherwise from 1 to number of streams initialized.
	 * @return pointer to a buffer
	 **/
	uint8_t *get_dev_checkbits(const unsigned stream_id);

	/**
	 * Reset checkbits to 0 for new kernel execution.
	 *
	 * @param stream_id Index of stream. If initilized to 0 stream then 0, otherwise from 1 to number of streams initialized.
	 * @param num_blks Length in bytes to be reset to 0. It should be same as the number of cuda blocks for kernel execution.
	 **/
	void clear_checkbits(const unsigned stream_id, const unsigned num_blks);


	/**
	 * Retrieves cudaStream from stream index.
	 *
	 * @param stream_id Index of stream. If initilized to 0 stream then 0, otherwise from 1 to number of streams initialized.
	 * @return cudaStream
	 **/
	cudaStream_t get_stream(const unsigned stream_id);

	/**
	 * Query whether the device context is initialized to use stream or not.
	 *
	 * @return
	 */
	bool use_stream() { return (nstream_ != 0); };

	/**
	 * Retrieves device memory pool to allocate memory for stream
	 *
	 * @param stream_id index of stream 1 to 16 for streams, and 0 for default stream
	 *
	 * @return
	 */
	class cuda_mem_pool *get_cuda_mem_pool(const unsigned stream_id);

	/**
	 * Retrieves the time took for the processing in the stream
	 * It will only return correct value when all the processing in the stream is finished
	 * and state of the stream is back to READY again.
	 *
	 * @param stream_id index of stream 1 to 16 for streams, and 0 for default stream
	 *
	 * @return
	 */
	uint64_t get_elapsed_time(const unsigned stream_id)
	{
		assert(0 <= stream_id && stream_id <= nstream_);
		assert((stream_id == 0) ^ (nstream_ > 0));
		return stream_ctx_[stream_id].end_usec - stream_ctx_[stream_id].begin_usec;
	}
private:
	struct stream_context stream_ctx_[MAX_STREAM + 1]; //stream_ctx 0 is for default stream
        unsigned int nstream_;
	bool init_;
};


#endif /* __GPU_DEVICE_CONTEXT__ */
