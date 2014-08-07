/* This file includes BN device function declarations and related data structure definitions */
#ifndef __BN_KN_H__
#define __BN_KN_H__

#include <openssl/bn.h>
#include <stdint.h>

#define MSGS_PER_BLOCK	1
#define MAX_RNS_CTXS	8
#define MAX_BS 80 // 64 for 512bit, 128 for 1024bit
#define MODULI_BITS 14
#define R_BOUND_BITS 11

#define MAX_NUM_MSG 2048

#define MAX_WIN 64

#define MODULI uint32_t

typedef struct RNS_CTX_st {
  // Bi in set A
  MODULI Bi_A[MAX_BS][MAX_BS];
  // Ai in set B
  MODULI Ai_B[MAX_BS][MAX_BS];

  MODULI a[MAX_BS]; 
  MODULI b[MAX_BS]; 
  
  // B in set A
  MODULI B_A[MAX_BS];
  // A in set B
  MODULI A_B[MAX_BS];

  MODULI Bsqr_modN_A[MAX_BS]; 
  MODULI Bsqr_modN_B[MAX_BS]; 

  MODULI N_A[MAX_BS]; 
  MODULI Np_B[MAX_BS];

  MODULI BI_modA_A[MAX_BS]; // B^1 mod A

  MODULI AiI_mod_ai[MAX_BS];
  MODULI BiI_mod_bi[MAX_BS];
 
  // constants for 1
  MODULI ONE_A[MAX_BS]; MODULI ONE_B[MAX_BS];

  int bs;

  // exponent d
  #define MAX_D_LEN 128 // big enough for 4096 bits
  int d_num_bits; // #bits
  int d_len; // d array length
  BN_ULONG d[MAX_D_LEN]; // exponent array

  // Vars used in CLNW alrorithm
  int CLNW_num; // #windows
  int CLNW[1024]; // windows 
  int CLNW_len[1024]; // lengthes of windows
  int CLNW_maxwin; // maximum window value

/*
  // Vars used in VLNW alrorithm
  int VLNW_num; // #windows
  int VLNW[4096]; // windows 
  int VLNW_len[4096]; // lengthes of windows
  int VLNW_maxwin; // maximum window value
*/

  int index;

#if 0
  // Pre-computed values before doing exponentiation, M, M^2, M^3, M^5, M^7
  MODULI M_A[MAX_NUM_MSG][MAX_WIN][MAX_BS];
  MODULI M_B[MAX_NUM_MSG][MAX_WIN][MAX_BS];
#endif

  /* These BN objects are used only on host for test purpose */
  BIGNUM *A, *B;
  BIGNUM *Bsqr_modN;
  BIGNUM *Ai[MAX_BS]; 
  BIGNUM *Bi[MAX_BS];
} __attribute__((aligned (64))) RNS_CTX;

#endif
