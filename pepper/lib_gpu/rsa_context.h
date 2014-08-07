#ifndef RSA_CONTEXT_HH
#define RSA_CONTEXT_HH

#include <string>

#include <openssl/rsa.h>

/**
 * class rsa_context
 *
 * Interface for RSA processing.
 *
 */
class rsa_context
{
public:
	/**
	* Constructor.
	* It will randomly generate the RSA key pair with the given key length.
	*
	* @param keylen Length of key in bits. Supported length are 512, 1024, 2048, and 4096 bits.
	*/
	rsa_context(int keylen);

	/**
	 * Constructor.
	 * It will load key from the file using the given password.
	 * Currently supports PEM format only
	 *
	 * @param filename file path that contains the rsa private key
	 * @param passwd password used to encrypt the private key
	 */
	rsa_context(const std::string &filename, const std::string &passwd);

	/**
	 * Constructor.
	 * It will load key from the file using the given password.
	 * Currently supports PEM format only
	 *
	 * @param filename file path that contains the rsa private key
	 * @param passwd password used to encrypt the private key
	 */
	rsa_context(const char *filename, const char *passwd);

	virtual ~rsa_context();

	/**
	 * Return key length.
	 *
	 * @return key length in bits.
	 */
	int get_key_bits();

	/**
	 * Return a maximum amount of plain text that can be encrypted.
	 *
	 * @return maximum plain text size in bytes.
	 */
	int max_ptext_bytes();

	/**
	 * Return whether chinese remainder theorem (CRT) is used for processing or not.
	 *
	 * @return true if CRT is enabled, false otherwise.
	 */
	bool is_crt_available();

	virtual void dump();

	/**
	 * Encrypts the data with RSA algorithm using public key.
	 * All encryption/decryption methods assume RSA_PKCS1_PADDING
	 *
	 * @param out Buffer for output.
	 * @param out_len In: allocated buffer space for output result, Out: out put size.
	 * @param in Intput plain text.
	 * @param in_len Intpu plain text size.
	 */
	virtual void pub_encrypt(unsigned char *out, int *out_len,
			const unsigned char *in, int in_len);

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

	float get_elapsed_ms_kernel();

	static const int max_batch = 2048 / 2;

protected:
	void dump_bn(BIGNUM *bn, const char *name);

	// returns -1 if it fails
	int remove_padding(unsigned char *out, int *out_len, BIGNUM *bn);

	RSA *rsa;
	BN_CTX *bn_ctx;

	float elapsed_ms_kernel;

private:
	void set_crt();

	bool crt_available;
};

#endif
