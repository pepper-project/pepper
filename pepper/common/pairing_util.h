#ifndef CODE_PEPPER_COMMON_PAIRING_UTIL_H_  
#define CODE_PEPPER_COMMON_PAIRING_UTIL_H_  

#include <string>
#include <crypto/prng.h>
#include <dirent.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <time.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdint.h>
#include <algorithm>
#include <list>
#include <map>
#include <vector>
#include <string>
#include <math.h>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <storage/exo.h>
#include <common/sha1.h>
#include <common/utility.h>

#if NONINTERACTIVE == 1

//Pairing libraries
#define PAIRING_PBC 1
#define PAIRING_LIBZM 2

#define PAIRING_LIB PAIRING_LIBZM
//#define PAIRING_LIB PAIRING_PBC

#define MULTIEXP

#if PAIRING_LIB == PAIRING_PBC
//#define PBC_DEBUG
#define PAIRING_PARAM "static_state/f_256.param"

#ifdef PBC_DEBUG
#define DEBUG_TEST_ENABLE
#endif

#include <pbc.h>

typedef element_t G1_t;
typedef element_t G2_t;
typedef element_t GT_t;

#elif PAIRING_LIB == PAIRING_LIBZM
#define PAIRING_PARAM "static_state/libzm.param"

//#define DEBUG_TEST_ENABLE

#include <common/pairing_pow_table.h>

#include <bn.h>

#ifdef assert
#undef assert
#endif

class G1_element;
class G2_element;
class GT_element;

typedef G1_element G1_t[1];
typedef G2_element G2_t[1];
typedef GT_element GT_t[1];

typedef pairing_pow_table<G1_t, 256> G1_pow_table_t;
typedef pairing_pow_table<G2_t, 256> G2_pow_table_t;

class G1_element {
public:
  G1_element();
  virtual ~G1_element();
  void build_powers();
  void pp_exp(G1_t out, mpz_t exp);
  /*
  Needs to be called after any 
  modification to value
  */
  void invalidate();
  bn::Fp value[3];
  G1_pow_table_t* powers;
};

class G2_element {
public:
  G2_element();
  virtual ~G2_element();
  void build_powers();
  void pp_exp(G2_t out, mpz_t exp);
  /*
  Needs to be called after any
  modification to value
  */
  void invalidate();
  bn::Fp2 value[3];
  G2_pow_table_t* powers;
};

class GT_element{
public:
  GT_element();
  virtual ~GT_element();
  bn::Fp12 value[1];
};

#endif

//using namespace std;
using std::cout;
using std::endl;
using std::string;
using std::vector;
using std::pair;
using std::ifstream;
using std::ofstream;

void alloc_init_vec_G1(G1_t **arr, uint32_t size);
void alloc_init_vec_G2(G2_t **arr, uint32_t size);
void alloc_init_vec_GT(GT_t **arr, uint32_t size);
void alloc_init_scalar_G1(G1_t s);
void alloc_init_scalar_G2(G2_t s);
void alloc_init_scalar_GT(GT_t s);
void clear_scalar_G1(G1_t s);
void clear_scalar_G2(G2_t s);
void clear_scalar_GT(GT_t s);
void clear_vec_G1(int size, G1_t *arr);
void clear_vec_G2(int size, G2_t *arr);
void clear_vec_GT(int size, GT_t *arr);

void dump_scalar_G1(G1_t q, char *scalar_name, const char *folder_name = NULL);
void dump_scalar_G2(G2_t q, char *scalar_name, const char *folder_name = NULL);
void dump_scalar_GT(GT_t q, char *scalar_name, const char *folder_name = NULL);
void dump_vector_G1(int size, G1_t *v, const char *vec_name, const char
  *folder_name = NULL);
void dump_vector_G2(int size, G2_t *v, const char *vec_name, const char
  *folder_name = NULL);
void dump_vector_GT(int size, GT_t *v, const char *vec_name, const char
  *folder_name = NULL);
void load_scalar_G1(G1_t q, const char *scalar_name, const char *folder_name = NULL);
void load_scalar_G2(G2_t q, const char *scalar_name, const char *folder_name = NULL);
void load_scalar_GT(GT_t q, const char *scalar_name, const char *folder_name = NULL);
void load_vector_G1(int size, G1_t *v, const char *vec_name, const char
  *folder_name = NULL);
void load_vector_G2(int size, G2_t *v, const char *vec_name, const char
  *folder_name = NULL);
void load_vector_GT(int size, GT_t *v, const char *vec_name, const char
  *folder_name = NULL);

void init_pairing_from_file(const char * filename, mpz_t prime);

void multi_exponentiation_G1(int size, G1_t *base, mpz_t *exponents, G1_t result);
void multi_exponentiation_G2(int size, G2_t *base, mpz_t *exponents, G2_t result);

void tri_multi_exponentiation_G1(G1_t *base, int size1, mpz_t *exp1, int size2, mpz_t *exp2, int size3, mpz_t *exp3, G1_t result);
void tri_multi_exponentiation_G2(G2_t *base, int size1, mpz_t *exp1, int size2, mpz_t *exp2, int size3, mpz_t *exp3, G2_t result);

void G1_set(G1_t, G1_t);
void G2_set(G2_t, G2_t);
//void GT_set(GT_t, GT_t);

void G1_set1(G1_t);
void G2_set1(G2_t);

void G1_mul(G1_t, G1_t, G1_t);
void G2_mul(G2_t, G2_t, G2_t);
void GT_mul(GT_t, GT_t, GT_t);

void G1_random(G1_t);
void G2_random(G2_t);
//void GT_random(GT_t);

void G1_exp(G1_t, G1_t, mpz_t);
void G2_exp(G2_t, G2_t, mpz_t);
//void GT_exp(GT_t, GT_t, mpz_t);

void G1_fixed_exp(G1_t*, G1_t, mpz_t*, int);
void G2_fixed_exp(G2_t*, G2_t, mpz_t*, int);

void G1_geom_fixed_exp(G1_t*, G1_t, mpz_t, mpz_t, int);
void G2_geom_fixed_exp(G2_t*, G2_t, mpz_t, mpz_t, int);

void G1_mul_fixed_exp(G1_t*, G1_t, mpz_t*, int);
void G2_mul_fixed_exp(G2_t*, G2_t, mpz_t*, int);

int G1_cmp(G1_t op1, G1_t op2);
int G2_cmp(G2_t op1, G2_t op2);
int GT_cmp(GT_t op1, GT_t op2);

void G1_print(G1_t op);
void G2_print(G2_t op);

void do_pairing(GT_t, G1_t, G2_t);
#endif

#endif  // CODE_PEPPER_COMMON_PAIRING_UTIL_H_
