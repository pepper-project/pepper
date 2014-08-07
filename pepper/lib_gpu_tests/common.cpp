#include <sys/time.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
uint64_t get_usec()
{
	struct timeval tv;
	assert(gettimeofday(&tv, NULL) == 0);
	return tv.tv_sec * 1000000 + tv.tv_usec;
}

void set_random(unsigned char *buf, int len)
{
	for (int i = 0; i < len; i++)
		buf[i] = rand() % 256;
}
