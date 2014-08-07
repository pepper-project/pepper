#include "device_context.h"

#include <helper_cuda.h>
#include <sys/time.h>
#include <assert.h>

static uint64_t get_now() {
	struct timeval tv;
	assert(gettimeofday(&tv, NULL) == 0);
	return tv.tv_sec * 1000000 + tv.tv_usec;
};

device_context::device_context()
{
	init_ = false;
}

device_context::~device_context()
{
	if (!init_)
		return;

	for (unsigned i = 1; i <= nstream_; i++) {
		checkCudaErrors(cudaStreamDestroy(stream_ctx_[i].stream));
		checkCudaErrors(cudaFreeHost(stream_ctx_[i].checkbits));
		stream_ctx_[i].pool.destroy();
	}

	if (nstream_ == 0) {
		checkCudaErrors(cudaFreeHost((void*)stream_ctx_[0].checkbits));
		stream_ctx_[0].pool.destroy();
	}
}

bool device_context::init(const unsigned long size, const unsigned nstream)
{
	void *ret = NULL;
	assert(nstream >= 0 && nstream <= MAX_STREAM);
	assert(init_ == false);
	init_ = true;

	nstream_ = nstream;

	if (nstream_ > 0) {
		for (unsigned i = 1; i <= nstream; i++) {
			checkCudaErrors(cudaStreamCreate(&stream_ctx_[i].stream));

			if (!stream_ctx_[i].pool.init(size))
				return false;
			stream_ctx_[i].state = READY;

			checkCudaErrors(cudaHostAlloc(&ret, MAX_BLOCKS, cudaHostAllocMapped));
			stream_ctx_[i].checkbits = (uint8_t*)ret;
			checkCudaErrors(cudaHostGetDevicePointer((void **)&stream_ctx_[i].checkbits_d, ret, 0));
		}
	} else {
		stream_ctx_[0].stream = 0;
		if (!stream_ctx_[0].pool.init(size))
			return false;

		stream_ctx_[0].state = READY;

		checkCudaErrors(cudaHostAlloc(&ret, MAX_BLOCKS, cudaHostAllocMapped));
		stream_ctx_[0].checkbits = (uint8_t*)ret;
		checkCudaErrors(cudaHostGetDevicePointer((void **)&stream_ctx_[0].checkbits_d, ret, 0));
	}
	return true;
}

bool device_context::sync(const unsigned stream_id, const bool block)
{
	assert(stream_id >= 0 && stream_id <= nstream_);
	assert((stream_id == 0) ^ (nstream_ > 0));
	if (!block) {
		if (stream_ctx_[stream_id].finished)
			return true;

		if (stream_ctx_[stream_id].state == WAIT_KERNEL && stream_ctx_[stream_id].num_blks > 0) {
			volatile uint8_t *checkbits = stream_ctx_[stream_id].checkbits;
			for (unsigned i = 0; i < stream_ctx_[stream_id].num_blks; i++) {
				if (checkbits[i] == 0)
					return false;
			}
		} else if (stream_ctx_[stream_id].state != READY) {
			cudaError_t ret = cudaStreamQuery(stream_ctx_[stream_id].stream);
			if (ret == cudaErrorNotReady)
				return false;
			assert(ret == cudaSuccess);
		}
		stream_ctx_[stream_id].finished = true;
	} else {
		cudaStreamSynchronize(stream_ctx_[stream_id].stream);
	}

	return true;
}

void device_context::set_state(const unsigned stream_id, const STATE state)
{
	assert(stream_id >= 0 && stream_id <= nstream_);
	assert((stream_id == 0) ^ (nstream_ > 0));

	if (state == READY) {
		stream_ctx_[stream_id].end_usec = get_now();
		stream_ctx_[stream_id].num_blks = 0;
		stream_ctx_[stream_id].pool.reset();
	} else if (state == WAIT_KERNEL) {
		stream_ctx_[stream_id].begin_usec = get_now();
		stream_ctx_[stream_id].finished = false;
	} else if (state == WAIT_COPY) {
		stream_ctx_[stream_id].finished = false;
	}
	stream_ctx_[stream_id].state = state;
}

enum STATE device_context::get_state(const unsigned stream_id)
{
	assert(stream_id >= 0 && stream_id <= nstream_);
	assert((stream_id == 0) ^ (nstream_ > 0));

	return stream_ctx_[stream_id].state;
}

cudaStream_t device_context::get_stream(const unsigned stream_id)
{
	assert(stream_id >= 0 && stream_id <= nstream_);
	assert((stream_id == 0) ^ (nstream_ > 0));

	return stream_ctx_[stream_id].stream;
}

uint8_t *device_context::get_dev_checkbits(const unsigned stream_id)
{
	assert(stream_id >= 0 && stream_id <= nstream_);
	assert((stream_id == 0) ^ (nstream_ > 0));

	return stream_ctx_[stream_id].checkbits_d;
}
void device_context::clear_checkbits(const unsigned stream_id, const unsigned num_blks)
{
	assert(stream_id >= 0 && stream_id <= nstream_);
	assert((stream_id == 0) ^ (nstream_ > 0));
	assert(num_blks >= 0 && num_blks <= MAX_BLOCKS);

	stream_ctx_[stream_id].num_blks = num_blks;
	volatile uint8_t *checkbits = stream_ctx_[stream_id].checkbits;
	for (unsigned i = 0; i < num_blks; i++)
		checkbits[i] = 0;
	stream_ctx_[stream_id].finished = false;
}


cuda_mem_pool *device_context::get_cuda_mem_pool(const unsigned stream_id)
{
	assert(stream_id >= 0 && stream_id <= nstream_);
	assert((stream_id == 0) ^ (nstream_ > 0));

	return &stream_ctx_[stream_id].pool;
}
