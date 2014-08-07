#ifndef __BN_CUDA__
#define __BN_CUDA__

#include "bn_kn.h" 

float BN_mod_exp_mont_batch_cu(BIGNUM *r[], BIGNUM *b[], int n, RNS_CTX *rns_ctx[]);

RNS_CTX * RNS_CTX_new(BIGNUM *N, BIGNUM *d);

void cpyRNSCTX2Dev();

void RNS_CTX_free(RNS_CTX *rns_ctx_h);

#endif
