#include <cassert>
#include <sys/time.h>
#include <openssl/bn.h>

#include "rsa_context_mp.h"

rsa_context_mp::rsa_context_mp(int keylen)
	: rsa_context(keylen)
{
	gpu_setup();
}

rsa_context_mp::rsa_context_mp(const std::string &filename,
		const std::string &passwd)
	: rsa_context(filename, passwd)
{
	gpu_setup();
}

rsa_context_mp::rsa_context_mp(const char *filename, const char *passwd)
	: rsa_context(filename, passwd)
{
	gpu_setup();
}

rsa_context_mp::~rsa_context_mp()
{
	BN_free(r);
	BN_free(r_inv[0]);
	BN_free(r_inv[1]);

	BN_free(in_bn_p);
	BN_free(in_bn_q);
	BN_free(out_bn_p);
	BN_free(out_bn_q);

	checkCudaErrors(cudaFree(sw_d));
	checkCudaErrors(cudaFree(n_d));
	checkCudaErrors(cudaFree(np_d));
	checkCudaErrors(cudaFree(r_sqr_d));
	checkCudaErrors(cudaFree(iqmp_d));

	for (unsigned int i = 0; i <= max_stream; i++) {
		checkCudaErrors(cudaFree(streams[i].a_d));
		checkCudaErrors(cudaFree(streams[i].ret_d));
		checkCudaErrors(cudaFree(streams[i].dbg_d));

		checkCudaErrors(cudaFreeHost(streams[i].a));
		checkCudaErrors(cudaFreeHost(streams[i].ret));
		checkCudaErrors(cudaFreeHost(streams[i].dbg));
	}
}

void rsa_context_mp::dump()
{
	if (is_crt_available()) {
		mp_print("p'", mp_np[0], (get_key_bits() / 2) / BITS_PER_WORD);
		mp_print("q'", mp_np[1], (get_key_bits() / 2) / BITS_PER_WORD);
	} else {
		mp_print("n'", mp_np[0], get_key_bits() / BITS_PER_WORD);
	}

	rsa_context::dump();
}

void rsa_context_mp::priv_decrypt(unsigned char *out, int *out_len,
		const unsigned char *in, int in_len)
{
	priv_decrypt_batch(&out, out_len, &in, &in_len, 1);
}

void rsa_context_mp::priv_decrypt_batch(unsigned char **out, int *out_len,
		const unsigned char **in, const int *in_len,
		int n)
{
	// by default, stream is not used
	priv_decrypt_stream(out, out_len, in, in_len, n, 0);
	sync(0);
}

void rsa_context_mp::priv_decrypt_stream(unsigned char **out, int *out_len,
		const unsigned char **in, const int *in_len,
		int n, unsigned int stream_id)
{
	assert(is_crt_available());
	assert(0 < n && n <= max_batch);
	assert(n <= MP_MAX_NUM_PAIRS);
	assert(stream_id <= max_stream);
	assert(dev_ctx_ != NULL);
	assert(dev_ctx_->get_state(stream_id) == READY);
	dev_ctx_->set_state(stream_id, WAIT_KERNEL);

	int word_len = (get_key_bits() / 2) / BITS_PER_WORD;
	int S = word_len;
	int num_blks = ((n + MP_MSGS_PER_BLOCK - 1) / MP_MSGS_PER_BLOCK) * 2;
	dev_ctx_->clear_checkbits(stream_id, num_blks);
	streams[stream_id].post_launched = false;

	for (int i = 0; i < n; i++) {
		BN_bin2bn(in[i], in_len[i], in_bn_p);
		BN_bin2bn(in[i], in_len[i], in_bn_q);
		assert(in_bn_p != NULL);
		assert(in_bn_q != NULL);

		//assert(BN_cmp(in_bn_p, rsa->n) < 0);

		BN_nnmod(in_bn_p, in_bn_p, rsa->p, bn_ctx);	// TODO: test BN_nnmod
		BN_nnmod(in_bn_q, in_bn_q, rsa->q, bn_ctx);

		mp_bn2mp(streams[stream_id].a + (i * 2 * MAX_S), in_bn_p, word_len);
		mp_bn2mp(streams[stream_id].a + (i * 2 * MAX_S) + MAX_S, in_bn_q, word_len);
	}

	//copy in put and execute kernel
	mp_modexp_crt((WORD *) streams[stream_id].a,
			n, word_len,
			(WORD *)streams[stream_id].ret_d, (WORD *) streams[stream_id].a_d,
			sw_d,
			(WORD *) n_d,
			(WORD *) np_d,
			(WORD *) r_sqr_d,
		      dev_ctx_->get_stream(stream_id),
		      stream_id,
		      dev_ctx_->get_dev_checkbits(stream_id));


	streams[stream_id].n = n;
	streams[stream_id].out = out;
	streams[stream_id].out_len = out_len;
}

bool rsa_context_mp::sync(unsigned int stream_id, bool block, bool copy_result)
{
	assert(stream_id <= max_stream);
	int word_len = (get_key_bits() / 2) / BITS_PER_WORD;

	if (dev_ctx_->get_state(stream_id) == READY)
		return true;

	//blocing case
	if (block) {
		//wait for previous operation to finish
		dev_ctx_->sync(stream_id, true);
		if (dev_ctx_->get_state(stream_id) == WAIT_KERNEL &&
			 streams[stream_id].post_launched == false) {
			//post kernel launch
			int S = word_len;
			int num_blks = ((streams[stream_id].n + MP_MSGS_PER_BLOCK - 1) / MP_MSGS_PER_BLOCK);
			dev_ctx_->clear_checkbits(stream_id, num_blks);
			mp_modexp_crt_post_kernel(streams[stream_id].ret,
						  streams[stream_id].ret_d,
						  n_d,
						  np_d,
						  r_sqr_d,
						  iqmp_d,
						  streams[stream_id].n,
						  word_len,
						  block,
						  dev_ctx_->get_stream(stream_id),
						  dev_ctx_->get_dev_checkbits(stream_id));
			streams[stream_id].post_launched = true;
			dev_ctx_->sync(stream_id, true);
		}

		if (dev_ctx_->get_state(stream_id) == WAIT_KERNEL &&
		    streams[stream_id].post_launched == true) {
			//copy result
			dev_ctx_->set_state(stream_id, WAIT_COPY);
			checkCudaErrors(cudaMemcpyAsync(streams[stream_id].ret,
						      streams[stream_id].ret_d,
						      sizeof(WORD[2][MAX_S]) * streams[stream_id].n,
						      cudaMemcpyDeviceToHost,
						      dev_ctx_->get_stream(stream_id)));
			dev_ctx_->sync(stream_id, true);
		}

		if (dev_ctx_->get_state(stream_id) == WAIT_COPY) {
			dev_ctx_->set_state(stream_id, READY);
		}

		//move result to out from gathred buffer
		for (int i = 0; i < streams[stream_id].n; i++) {
			int rsa_bytes = get_key_bits() / 8;

			int ret = RSA_padding_check_PKCS1_type_2(streams[stream_id].out[i],
								 streams[stream_id].out_len[i],
								 (unsigned char *)(streams[stream_id].ret + (i * 2 * MAX_S)) + 1,
								 rsa_bytes - 1,
								 rsa_bytes);
			if (ret == -1) {
				for (int j = 0; j < 2 * word_len * (int)sizeof(WORD); j++)
					printf("%02x ", *(((unsigned char *)(streams[stream_id].ret + (i * 2 * MAX_S)) + j)));
				printf("\n");
				assert(false);
			}
			streams[stream_id].out_len[i] = ret;
		}

		return true;
	}


	//nonblocking case
	if (dev_ctx_->get_state(stream_id) == WAIT_KERNEL) {
		if (!dev_ctx_->sync(stream_id, false))
			return false;
		if (!streams[stream_id].post_launched) {
			//start post kernel execution
			int S = word_len;
			int num_blks = ((streams[stream_id].n + MP_MSGS_PER_BLOCK - 1) / MP_MSGS_PER_BLOCK);

			streams[stream_id].post_launched = true;
			dev_ctx_->clear_checkbits(stream_id, num_blks);
			mp_modexp_crt_post_kernel(streams[stream_id].ret,
						  streams[stream_id].ret_d,
						  n_d,
						  np_d,
						  r_sqr_d,
						  iqmp_d,
						  streams[stream_id].n,
						  word_len,
						  block,
						  dev_ctx_->get_stream(stream_id),
						  dev_ctx_->get_dev_checkbits(stream_id));
			streams[stream_id].post_launched = true;
			return false;
		} else {
			//start copying result
			dev_ctx_->set_state(stream_id, WAIT_COPY);
			checkCudaErrors(cudaMemcpyAsync(streams[stream_id].ret,
						      streams[stream_id].ret_d,
						      sizeof(WORD[2][MAX_S]) * streams[stream_id].n,
						      cudaMemcpyDeviceToHost,
						      dev_ctx_->get_stream(stream_id)));
			return false;
		}

	} else if (dev_ctx_->get_state(stream_id) == WAIT_COPY) {
		if (!dev_ctx_->sync(stream_id, false))
			return false;

		//move result to out from gathred buffer
		for (int i = 0; i < streams[stream_id].n; i++) {
			int rsa_bytes = get_key_bits() / 8;

			int ret = RSA_padding_check_PKCS1_type_2(streams[stream_id].out[i],
								 streams[stream_id].out_len[i],
								 (unsigned char *)(streams[stream_id].ret + (i * 2 * MAX_S)) + 1,
								 rsa_bytes - 1,
								 rsa_bytes);
			if (ret == -1) {
				for (int j = 0; j < 2 * word_len * (int)sizeof(WORD); j++)
					printf("%02x ", *(((unsigned char *)(streams[stream_id].ret + (i * 2 * MAX_S)) + j)));
				printf("\n");
				assert(false);
			}
			streams[stream_id].out_len[i] = ret;
		}

		dev_ctx_->set_state(stream_id, READY);
		return true;
	}
	return false;
}

void rsa_context_mp::gpu_setup()
{
	assert(is_crt_available());
	assert(get_key_bits() == 512 ||
			get_key_bits() == 1024 ||
			get_key_bits() == 2048 ||
			get_key_bits() == 4096);

	int word_len = (get_key_bits() / 2) / BITS_PER_WORD;
	dev_ctx_ = NULL;

	in_bn_p = BN_new();
	in_bn_q = BN_new();
	out_bn_p = BN_new();
	out_bn_q = BN_new();

	{
		struct mp_sw sw[2];

		mp_bn2mp(mp_e[0], rsa->dmp1, word_len);
		mp_bn2mp(mp_e[1], rsa->dmq1, word_len);

		mp_get_sw(&sw[0], mp_e[0], word_len);
		mp_get_sw(&sw[1], mp_e[1], word_len);

		checkCudaErrors(cudaMalloc(&sw_d, sizeof(struct mp_sw) * 2));
		checkCudaErrors(cudaMemcpy(sw_d, sw, sizeof(struct mp_sw) * 2, cudaMemcpyHostToDevice));
	}

	{
		mp_bn2mp(mp_n[0], rsa->p, word_len);
		mp_bn2mp(mp_n[1], rsa->q, word_len);

		checkCudaErrors(cudaMalloc(&n_d, sizeof(mp_n[0]) * 2));
		checkCudaErrors(cudaMemcpy(n_d, mp_n, sizeof(mp_n[0]) * 2, cudaMemcpyHostToDevice));
	}

	{
		mp_bn2mp(mp_iqmp, rsa->iqmp, word_len);

		checkCudaErrors(cudaMalloc(&iqmp_d, sizeof(mp_iqmp)));
		checkCudaErrors(cudaMemcpy(iqmp_d, mp_iqmp, sizeof(mp_iqmp), cudaMemcpyHostToDevice));
	}

	r = BN_new();
	BN_set_bit(r, get_key_bits() / 2);

	{
		BIGNUM *NP = BN_new();

		r_inv[0] = BN_new();
		BN_mod_inverse(r_inv[0], r, rsa->p, bn_ctx);
		BN_mul(NP, r, r_inv[0], bn_ctx);
		BN_sub_word(NP, 1);
		BN_div(NP, NULL, NP, rsa->p, bn_ctx);
		mp_bn2mp(mp_np[0], NP, word_len);

		r_inv[1] = BN_new();
		BN_mod_inverse(r_inv[1], r, rsa->q, bn_ctx);
		BN_mul(NP, r, r_inv[1], bn_ctx);
		BN_sub_word(NP, 1);
		BN_div(NP, NULL, NP, rsa->q, bn_ctx);
		mp_bn2mp(mp_np[1], NP, word_len);

		BN_free(NP);
	}

	{
		BIGNUM *R_SQR = BN_new();

		BN_mod_mul(R_SQR, r, r, rsa->p, bn_ctx);
		mp_bn2mp(mp_r_sqr[0], R_SQR, word_len);

		BN_mod_mul(R_SQR, r, r, rsa->q, bn_ctx);
		mp_bn2mp(mp_r_sqr[1], R_SQR, word_len);

		checkCudaErrors(cudaMalloc(&r_sqr_d, sizeof(mp_r_sqr[0]) * 2));
		checkCudaErrors(cudaMemcpy(r_sqr_d, mp_r_sqr, sizeof(mp_r_sqr[0]) * 2, cudaMemcpyHostToDevice));

		BN_free(R_SQR);
	}

	checkCudaErrors(cudaMalloc(&np_d, sizeof(mp_np[0]) * 2));
	checkCudaErrors(cudaMemcpy(np_d, mp_np, sizeof(mp_np[0]) * 2, cudaMemcpyHostToDevice));

	checkCudaErrors(cudaEventCreate(&begin_evt));
	checkCudaErrors(cudaEventCreate(&end_evt));

	for (unsigned int i = 0; i <= max_stream; i++) {
		checkCudaErrors(cudaMalloc(&streams[i].dbg_d, sizeof(WORD[max_batch][2][MAX_S])));
		checkCudaErrors(cudaMalloc(&streams[i].ret_d, sizeof(WORD[max_batch][2][MAX_S])));
		checkCudaErrors(cudaMalloc(&streams[i].a_d, sizeof(WORD[max_batch][2][MAX_S])));

		checkCudaErrors(cudaHostAlloc(&streams[i].a,
					sizeof(WORD[max_batch][2][MAX_S]),
					cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc(&streams[i].ret,
					sizeof(WORD[max_batch][2][MAX_S]),
					cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc(&streams[i].dbg,
					sizeof(WORD[max_batch][2][MAX_S]),
					cudaHostAllocPortable));
	}
}
