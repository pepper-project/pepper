#include <cassert>
#include <cstring>
#include <sys/time.h>

#include <openssl/pem.h>
#include <openssl/err.h>
#include <openssl/evp.h>

#include "rsa_context.h"

rsa_context::rsa_context(int keylen)
{
	assert(keylen == 512 || keylen == 1024 || keylen == 2048 || keylen == 4096);

	BIGNUM *e = BN_new();
	BN_set_word(e, RSA_F4 /* 65537 */);

	rsa = RSA_new();
	RSA_generate_key_ex(rsa, keylen, e, NULL);
	assert(RSA_check_key(rsa));
	BN_free(e);

	set_crt();

	bn_ctx = BN_CTX_new();
	elapsed_ms_kernel = 0.0f;
};

rsa_context::rsa_context(const std::string &filename, const std::string &passwd)
{
	rsa_context(filename.c_str(), passwd.c_str());
}

rsa_context::rsa_context(const char *filename, const char *passwd)
{
	BIO *key;
	OpenSSL_add_all_algorithms();

	key = BIO_new(BIO_s_file());
	assert(key != NULL);
	assert(BIO_read_filename(key, filename) == 1);
	rsa = PEM_read_bio_RSAPrivateKey(key, NULL, NULL, (void *)passwd);
	ERR_print_errors_fp(stdout);
	BIO_free(key);

	set_crt();

	bn_ctx = BN_CTX_new();
	elapsed_ms_kernel = 0.0f;
};

rsa_context::~rsa_context()
{
	RSA_free(rsa);
	BN_CTX_free(bn_ctx);
}

int rsa_context::get_key_bits()
{
	return RSA_size(rsa) * 8;
}

int rsa_context::max_ptext_bytes()
{
	return RSA_size(rsa) - 11;
}

bool rsa_context::is_crt_available()
{
	return crt_available;
}

void rsa_context::dump()
{
	dump_bn(rsa->e, "e");
	dump_bn(rsa->d, "d");
	dump_bn(rsa->n, "n");

	if (crt_available) {
		dump_bn(rsa->p, "p");
		dump_bn(rsa->q, "q");
	}
}

void rsa_context::pub_encrypt(unsigned char *out, int *out_len, 
		const unsigned char *in, int in_len)
{
	int bytes_needed = get_key_bits() / 8;
	assert(*out_len >= bytes_needed);

	assert(in_len <= max_ptext_bytes());

	*out_len = RSA_public_encrypt(in_len, in, out, rsa, RSA_PKCS1_PADDING);
	assert(*out_len != -1);
}

void rsa_context::priv_decrypt(unsigned char *out, int *out_len, 
		const unsigned char *in, int in_len)
{
#if 0
	if (is_crt_available()) {
		// This is only for debugging purpose.
		// RSA_private_decrypt() is enough.
		BIGNUM *c = BN_bin2bn(in, in_len, NULL);
		assert(c != NULL);
		assert(BN_cmp(c, rsa->n) == -1);

		BIGNUM *m1 = BN_new();
		BIGNUM *m2 = BN_new();
		BIGNUM *t = BN_new();

		BN_nnmod(t, c, rsa->p, bn_ctx);
		BN_mod_exp(m1, t, rsa->dmp1, rsa->p, bn_ctx);
		BN_nnmod(t, c, rsa->q, bn_ctx);
		BN_mod_exp(m2, t, rsa->dmq1, rsa->q, bn_ctx);
		
		BN_sub(t, m1, m2);
		BN_mod_mul(t, t, rsa->iqmp, rsa->p, bn_ctx);
		BN_mul(t, t, rsa->q, bn_ctx);
		BN_add(t, m2, t);

		int ret = remove_padding(out, out_len, t);
		assert(ret != -1);

		BN_free(c);
		BN_free(m1);
		BN_free(m2);
		BN_free(t);
	} else {
#endif
		int bytes_needed = get_key_bits() / 8;
		assert(*out_len >= bytes_needed);

		*out_len = RSA_private_decrypt(in_len, in, out, rsa, 
				RSA_PKCS1_PADDING);
		assert(*out_len != -1);
#if 0
	}
#endif
}

void rsa_context::priv_decrypt_batch(unsigned char **out, int *out_len,
		const unsigned char **in, const int *in_len,
		int n)
{
	assert(0 < n && n <= max_batch);

	for (int i = 0; i < n; i++)
		priv_decrypt(out[i], &out_len[i], in[i], in_len[i]);
}

float rsa_context::get_elapsed_ms_kernel()
{
	return elapsed_ms_kernel;
}

void rsa_context::dump_bn(BIGNUM *bn, const char *name)
{
	printf("%s (%4d bits): ", name, BN_num_bits(bn));
	BN_print_fp(stdout, bn);
	printf("\n");
}

int rsa_context::remove_padding(unsigned char *out, int *out_len, BIGNUM *bn)
{
	int bytes_needed = get_key_bits() / 8;
	assert(*out_len >= bytes_needed);

	unsigned char bn_bin[1024];
	int bn_bin_len;
	int ret;

	bn_bin_len = BN_bn2bin(bn, bn_bin);
	ret = RSA_padding_check_PKCS1_type_2(out, *out_len, bn_bin, bn_bin_len, bytes_needed);
	if (ret == -1) {
#if 1
		// for debugging purposes...
		ERR_print_errors_fp(stderr);
		return ret;
#else
		assert(false);
#endif
	}

	*out_len = ret;
	return 0;
}

void rsa_context::set_crt()
{
	crt_available = (
			rsa->p != NULL &&
			rsa->q != NULL &&
			rsa->dmp1 != NULL &&
			rsa->dmq1 != NULL &&
			rsa->iqmp != NULL
			);

// turn this on to disable CRT
#if 0
	crt_available = false;
#endif

	if (!crt_available) {
		fprintf(stderr, "-----------------------------------------------------\n");
		fprintf(stderr, "WARNING: Chinese remainder theorem is not applicable!\n");
		fprintf(stderr, "-----------------------------------------------------\n");
	}
}
