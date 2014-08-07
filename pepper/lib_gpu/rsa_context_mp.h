#ifndef RSA_CONTEXT_MP_HH
#define RSA_CONTEXT_MP_HH

#include "rsa_context.h"
#include "mp_modexp.h"
#include "mp_modexp_gpu.h"
#include "device_context.h"

#include <cuda_runtime.h>
#include <helper_cuda.h>

/**
 * class rsa_context
 *
 * Interface for RSA processing in GPU.
 *
 */
class rsa_context_mp : public rsa_context
{
public:
	/**
	* Constructor.
	* It will randomly generate the RSA key pair with the given key length.
	*
	* @param keylen Length of key in bits. Supported length are 512, 1024, 2048, and 4096 bits.
	*/
	rsa_context_mp(int keylen);

	/**
	 * Constructor.
	 * It will load key from the file using the given password.
	 * Currently supports PEM format only
	 *
	 * @param filename file path that contains the rsa private key
	 * @param passwd password used to encrypt the private key
	 */
	rsa_context_mp(const std::string &filename, const std::string &passwd);

	/**
	 * Constructor.
	 * It will load key from the file using the given password.
	 * Currently supports PEM format only
	 *
	 * @param filename file path that contains the rsa private key
	 * @param passwd password used to encrypt the private key
	 */
	rsa_context_mp(const char *filename, const char *passwd);

	virtual ~rsa_context_mp();

	virtual void dump();

	/**
	 * Decrypt the data with RSA algorithm using private key.
	 * All encryption/decryption methods assume RSA_PKCS1_PADDING
	 *
	 * @param out Buffer for output.
	 * @param out_len In: allocated buffer space for output result, Out: output size.
	 * @param in Buffer that stores cipher text.
	 * @param in_len Length of cipher text
	 */
	virtual void priv_decrypt(unsigned char *out, int *out_len,
			const unsigned char *in, int in_len);

	/**
	 * Decrypt the data with RSA algorithm using private key in a batch
	 * All encryption/decryption methods assume RSA_PKCS1_PADDING
	 *
	 * @param out Buffers for plain text.
	 * @param out_len In: allocated buffer space for output results, Out: output sizes.
	 * @param in Buffers that stores ciphertext.
	 * @param in_len Length of cipher texts.
	 * @param n Ciphertexts count.
	 */
	virtual void priv_decrypt_batch(unsigned char **out, int *out_len,
			const unsigned char **in, const int *in_len,
			int n);

	/**
	 * Decrypt the data with RSA algorithm using private key in a batch
	 * All encryption/decryption methods assume RSA_PKCS1_PADDING
	 * It runs asynchronously. Use sync() for completion check.
	 *
	 * @param out Buffers for plain text.
	 * @param out_len In: allocated buffer space for output results, Out: output sizes.
	 * @param in Buffers that stores ciphertext.
	 * @param in_len Length of cipher texts.
	 * @param n Ciphertexts count.
	 * @param stream_id Stream index. 1 <= stream_id <= max_stream
	 */
	void priv_decrypt_stream(unsigned char **out, int *out_len,
			const unsigned char **in, const int *in_len,
			int n, unsigned int stream_id);

	/**
	 * Synchronize/query the execution on the stream.
	 * This function can be used to check whether the current execution
	 * on the stream is finished or also be used to wait until
	 * the execution to be finished.
	 *
	 * @param stream Stream index.
	 * @param block Wait for the execution to finish or not. true by default.
	 * @param copy_result If false, it will not copy result back to CPU.
	 *
	 * @return true if the current operation on the stream is finished
	 * otherwise false.
	 */
	bool sync(unsigned int stream_id, bool block = true, bool copy_result = true);

	static const unsigned int max_stream = MAX_STREAMS;

	/**
	 * Sets the device context for the rsa context.
	 * TODO: Move dev_ctx initialization to constructor.
	 *
	 * @param dev_ctx device context.
	 */
	void set_device_context(device_context *dev_ctx) {dev_ctx_ = dev_ctx;};

protected:

private:
	void gpu_setup();

	/* for single-stream only */
	cudaEvent_t begin_evt;
	cudaEvent_t end_evt;
	device_context *dev_ctx_;

	struct {
		bool post_launched;

		unsigned char **out;
		int *out_len;
		int n;

		WORD *a;
		WORD *ret;
		WORD *dbg;

		WORD *a_d;
		WORD *ret_d;
		WORD *dbg_d;
	} streams[max_stream + 1];

	struct mp_sw *sw_d;
	WORD *n_d;
	WORD *np_d;
	WORD *r_sqr_d;
	WORD *iqmp_d;

	WORD mp_e[2][MAX_S];
	WORD mp_n[2][MAX_S];
	WORD mp_np[2][MAX_S];
	WORD mp_r_sqr[2][MAX_S];
	WORD mp_iqmp[MAX_S];

	BIGNUM *r;
	BIGNUM *r_inv[2];

	BIGNUM *in_bn_p;
	BIGNUM *in_bn_q;
	BIGNUM *out_bn_p;
	BIGNUM *out_bn_q;
};

#endif
