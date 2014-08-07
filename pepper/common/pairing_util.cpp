#include <dirent.h>
#include <common/pairing_util.h>
#include <cassert>

#if NONINTERACTIVE == 1

#if PAIRING_LIB == PAIRING_PBC
pairing_t pairing;
#elif PAIRING_LIB == PAIRING_LIBZM

//If 1, don't output (X,Y) pairs,
//but just output X and which square root of X^3 + (b) to use for Y
#define COMPRESS_PROOF 1

using namespace bn;

namespace {
  G1_t g1, g1_one;
  G2_t g2, g2_one;
  mpz_t prime;
  Prng * prng;
}
#endif

void alloc_init_vec_G1(G1_t **arr, uint32_t size) {
  *arr = new G1_t[size];
  for (uint32_t i=0; i<size; i++) {
    alloc_init_scalar_G1((*arr)[i]);
  }
}

void alloc_init_vec_G2(G2_t **arr, uint32_t size) {
  *arr = new G2_t[size];
  for (uint32_t i=0; i<size; i++) {
    alloc_init_scalar_G2((*arr)[i]);
  }
}

void alloc_init_vec_GT(GT_t **arr, uint32_t size) {
  *arr = new GT_t[size];
  for (uint32_t i=0; i<size; i++) {
    alloc_init_scalar_GT((*arr)[i]);
  }
}

void clear_vec_G1(int size, G1_t *arr) {
  for (int i=0; i<size; i++)
    clear_scalar_G1(arr[i]);
  delete [] arr; //use delete [] to complement new []
}
void clear_vec_G2(int size, G2_t *arr) {
  for (int i=0; i<size; i++)
    clear_scalar_G2(arr[i]);
  delete [] arr;
}
void clear_vec_GT(int size, GT_t *arr) {
  for (int i=0; i<size; i++)
    clear_scalar_GT(arr[i]);
  delete [] arr;
}

void tri_multi_exponentiation_G1(G1_t *base, int size1, mpz_t *exp1, int size2, mpz_t *exp2, int size3, mpz_t *exp3, G1_t result) {
  G1_t tmp1;
  alloc_init_scalar_G1(tmp1);
  multi_exponentiation_G1(size1, base, exp1, result);
  multi_exponentiation_G1(size2, base+size1, exp2, tmp1);
  G1_mul(result, result, tmp1);
  multi_exponentiation_G1(size3, base+size1+size2, exp3, tmp1);
  G1_mul(result, result, tmp1);
  clear_scalar_G1(tmp1);
}
void tri_multi_exponentiation_G2(G2_t *base, int size1, mpz_t *exp1, int size2, mpz_t *exp2, int size3, mpz_t *exp3, G2_t result) {
  G2_t tmp1;
  alloc_init_scalar_G2(tmp1);
  multi_exponentiation_G2(size1, base, exp1, result);
  multi_exponentiation_G2(size2, base+size1, exp2, tmp1);
  G2_mul(result, result, tmp1);
  multi_exponentiation_G2(size3, base+size1+size2, exp3, tmp1);
  G2_mul(result, result, tmp1);
  clear_scalar_G2(tmp1);
}



#if PAIRING_LIB == PAIRING_PBC

void alloc_init_scalar_G1(G1_t s) {
  element_init_G1(s, pairing);
}

void alloc_init_scalar_G2(G2_t s) {
  element_init_G2(s, pairing);
}

void alloc_init_scalar_GT(GT_t s) {
  element_init_GT(s, pairing);
}

void clear_scalar_G1(G1_t s) {
  element_clear(s);
}
void clear_scalar_G2(G2_t s) {
  element_clear(s);
}
void clear_scalar_GT(GT_t s) {
  element_clear(s);
}

void load_scalar_G1(G1_t q, const char *scalar_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, scalar_name, (char *)"rb", folder_name);
  if (fp == NULL) return;
  // call element_from_bytes to read it from file.
  size_t element_size = element_length_in_bytes(q);
  unsigned char *data = (unsigned char*) pbc_malloc(element_size);
  if (element_size != fread(data, 1, element_size, fp)) {
    cout<<"Cannot read enough bytes to initilize an element_t."<<endl;
    exit(1);
  }
  element_from_bytes(q, data);

  pbc_free(data);

  fclose(fp);
}
void load_scalar_G2(G2_t q, const char *scalar_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, scalar_name, (char *)"rb", folder_name);
  if (fp == NULL) return;
  // call element_from_bytes to read it from file.
  size_t element_size = element_length_in_bytes(q);
  unsigned char *data = (unsigned char*) pbc_malloc(element_size);
  if (element_size != fread(data, 1, element_size, fp)) {
    cout<<"Cannot read enough bytes to initilize an element_t."<<endl;
    exit(1);
  }
  element_from_bytes(q, data);

  pbc_free(data);

  fclose(fp);
}
void load_scalar_GT(GT_t q, const char *scalar_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, scalar_name, (char *)"rb", folder_name);
  if (fp == NULL) return;
  // call element_from_bytes to read it from file.
  size_t element_size = element_length_in_bytes(q);
  unsigned char *data = (unsigned char*) pbc_malloc(element_size);
  if (element_size != fread(data, 1, element_size, fp)) {
    cout<<"Cannot read enough bytes to initilize an element_t."<<endl;
    exit(1);
  }
  element_from_bytes(q, data);

  pbc_free(data);

  fclose(fp);
}

void load_vector_G1(int size, G1_t *v, const char *vec_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, vec_name, (char *)"rb", folder_name);
  if (fp == NULL) return;
  size_t element_size = element_length_in_bytes(v[0]);
  unsigned char *data = (unsigned char*) pbc_malloc(element_size);
  for (int i=0; i<size; i++) {
    // read the raw file and call element_from_bytes to convert to group element.
    if (element_size != fread(data, 1, element_size, fp)) {
      cout<<"Cannot read enough bytes to initilize an element_t."<<endl;
      exit(1);
    }
    element_from_bytes(v[i], data);
  }
  pbc_free(data);

  fclose(fp);
}
void load_vector_G2(int size, G2_t *v, const char *vec_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, vec_name, (char *)"rb", folder_name);
  if (fp == NULL) return;
  size_t element_size = element_length_in_bytes(v[0]);
  unsigned char *data = (unsigned char*) pbc_malloc(element_size);
  for (int i=0; i<size; i++) {
    // read the raw file and call element_from_bytes to convert to group element.
    if (element_size != fread(data, 1, element_size, fp)) {
      cout<<"Cannot read enough bytes to initilize an element_t."<<endl;
      exit(1);
    }
    element_from_bytes(v[i], data);
  }
  pbc_free(data);

  fclose(fp);
}
void load_vector_GT(int size, GT_t *v, const char *vec_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, vec_name, (char *)"rb", folder_name);
  if (fp == NULL) return;
  size_t element_size = element_length_in_bytes(v[0]);
  unsigned char *data = (unsigned char*) pbc_malloc(element_size);
  for (int i=0; i<size; i++) {
    // read the raw file and call element_from_bytes to convert to group element.
    if (element_size != fread(data, 1, element_size, fp)) {
      cout<<"Cannot read enough bytes to initilize an element_t."<<endl;
      exit(1);
    }
    element_from_bytes(v[i], data);
  }
  pbc_free(data);

  fclose(fp);
}

void dump_scalar_G1(G1_t q, char *scalar_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, scalar_name, (char *)"wb", folder_name);
  if (fp == NULL) return;
  int element_size = element_length_in_bytes(q);
  unsigned char *data = (unsigned char*) pbc_malloc(element_size);
  // call element_to_bytes to write the element into a buffer
  element_to_bytes(data, q);
  // write the buffer to file
  fwrite(data, 1, element_size, fp);
  pbc_free(data);
  fclose(fp);
}
void dump_scalar_G2(G2_t q, char *scalar_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, scalar_name, (char *)"wb", folder_name);
  if (fp == NULL) return;
  int element_size = element_length_in_bytes(q);
  unsigned char *data = (unsigned char*) pbc_malloc(element_size);
  // call element_to_bytes to write the element into a buffer
  element_to_bytes(data, q);
  // write the buffer to file
  fwrite(data, 1, element_size, fp);
  pbc_free(data);
  fclose(fp);
}
void dump_scalar_GT(GT_t q, char *scalar_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, scalar_name, (char *)"wb", folder_name);
  if (fp == NULL) return;
  int element_size = element_length_in_bytes(q);
  unsigned char *data = (unsigned char*) pbc_malloc(element_size);
  // call element_to_bytes to write the element into a buffer
  element_to_bytes(data, q);
  // write the buffer to file
  fwrite(data, 1, element_size, fp);
  pbc_free(data);
  fclose(fp);
}

void dump_vector_G1(int size, G1_t *v, const char *vec_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, vec_name, (char *)"wb", folder_name);
  if (fp == NULL) return;

  int element_size = element_length_in_bytes(v[0]);
  unsigned char *data = (unsigned char*)pbc_malloc(element_size);
  for (int i=0; i<size; i++) {
    // call element_to_bytes to write the element into a buffer
    element_to_bytes(data, v[i]);
    // write the buffer to file
    fwrite(data, 1, element_size, fp);
  }
  pbc_free(data);

  fclose(fp);
}
void dump_vector_G2(int size, G2_t *v, const char *vec_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, vec_name, (char *)"wb", folder_name);
  if (fp == NULL) return;

  int element_size = element_length_in_bytes(v[0]);
  unsigned char *data = (unsigned char*)pbc_malloc(element_size);
  for (int i=0; i<size; i++) {
    // call element_to_bytes to write the element into a buffer
    element_to_bytes(data, v[i]);
    // write the buffer to file
    fwrite(data, 1, element_size, fp);
  }
  pbc_free(data);

  fclose(fp);
}
void dump_vector_GT(int size, GT_t *v, const char *vec_name, const char *folder_name) {
  FILE *fp;
  open_file(&fp, vec_name, (char *)"wb", folder_name);
  if (fp == NULL) return;

  int element_size = element_length_in_bytes(v[0]);
  unsigned char *data = (unsigned char*)pbc_malloc(element_size);
  for (int i=0; i<size; i++) {
    // call element_to_bytes to write the element into a buffer
    element_to_bytes(data, v[i]);
    // write the buffer to file
    fwrite(data, 1, element_size, fp);
  }
  pbc_free(data);

  fclose(fp);
}

void init_pairing_from_file(const char * filename, mpz_t _prime) {
  //init pairing
  char s[16384];
  FILE *fp = stdin;
  fp = fopen(filename, "r");
  if (!fp) {
    cout<<"Cannot find "<<filename<<endl;
    exit(1);
  }
  size_t count = fread(s, 1, 16384, fp);
  if (!count) {
    cout<<"Cannot read "<<filename<<endl;
    exit(1);
  }
  fclose(fp);
  if (pairing_init_set_buf(pairing, s, count)) {
    cout<<"Cannot initialize pairing with file content "<<filename<<endl;
    exit(1);
  }

}

// using multi-exponentiation
void multi_exponentiation_G1(int size, G1_t *base, mpz_t *exponents, G1_t result) {

  // make sure that the result can be passed back to caller: yes.
  element_t tmp;
  // init each data structure. assume result is already initialized.
  alloc_init_scalar_G1(tmp);
  G1_set1(result);
#ifdef MULTIEXP
  for (int i = 0; i + 2 < size; i += 3) {
    element_pow3_mpz(tmp, base[i], exponents[i], base[i + 1], exponents[i + 1], base[i + 2], exponents[i + 2]);
    //is this correct? yes, it is result = result * tmp
    element_mul(result, result, tmp);
  }
  if ((size % 3) == 1) {
    element_pow_mpz(tmp, base[size - 1], exponents[size - 1]);
    element_mul(result, result, tmp);
  } else if ((size % 3) == 2) {
    element_pow2_mpz(tmp, base[size - 2], exponents[size - 2], base[size - 1], exponents[size - 1]);
    element_mul(result, result, tmp);
  }
#else
  for (int i = 0; i < size; i++) {
    element_pow_mpz(tmp, base[i], exponents[i]);
    element_mul(result, result, tmp);
  }
#endif
  clear_scalar_G1(tmp);

}
void multi_exponentiation_G2(int size, G2_t *base, mpz_t *exponents, G2_t result) {

  // make sure that the result can be passed back to caller: yes.
  element_t tmp;
  // init each data structure. assume result is already initialized.
  alloc_init_scalar_G2(tmp);
  G2_set1(result);
#ifdef MULTIEXP
  for (int i = 0; i + 2 < size; i += 3) {
    element_pow3_mpz(tmp, base[i], exponents[i], base[i + 1], exponents[i + 1], base[i + 2], exponents[i + 2]);
    //is this correct? yes, it is result = result * tmp
    element_mul(result, result, tmp);
  }
  if ((size % 3) == 1) {
    element_pow_mpz(tmp, base[size - 1], exponents[size - 1]);
    element_mul(result, result, tmp);
  } else if ((size % 3) == 2) {
    element_pow2_mpz(tmp, base[size - 2], exponents[size - 2], base[size - 1], exponents[size - 1]);
    element_mul(result, result, tmp);
  }
#else
  for (int i = 0; i < size; i++) {
    element_pow_mpz(tmp, base[i], exponents[i]);
    element_mul(result, result, tmp);
  }
#endif
  clear_scalar_G2(tmp);

}

void G1_set(G1_t e1, G1_t e2) {
  element_set(e1, e2);
}

void G2_set(G2_t e1, G2_t e2) {
  element_set(e1, e2);
}

void GT_set(GT_t e1, GT_t e2) {
  element_set(e1, e2);
}

void G1_set1(G1_t e1) {
  element_set1(e1);
}

void G2_set1(G2_t e1) {
  element_set1(e1);
}

void G1_mul(G1_t rop, G1_t op1, G1_t op2) {
  element_mul(rop, op1, op2);
}

void G2_mul(G2_t rop, G2_t op1, G2_t op2) {
  element_mul(rop, op1, op2);
}

void GT_mul(GT_t rop, GT_t op1, GT_t op2) {
  element_mul(rop, op1, op2);
}

void G1_random(G1_t e) {
  element_random(e);
}

void G2_random(G2_t e) {
  element_random(e);
}

void G1_exp(G1_t rop, G1_t op1, mpz_t exp) {
  element_pow_mpz(rop, op1, exp);
}

void G2_exp(G2_t rop, G2_t op1, mpz_t exp) {
  element_pow_mpz(rop, op1, exp);
}

int G1_cmp(G1_t op1, G1_t op2) {
  return element_cmp(op1, op2);
}
int G2_cmp(G2_t op1, G2_t op2) {
  return element_cmp(op1, op2);
}
int GT_cmp(GT_t op1, GT_t op2) {
  return element_cmp(op1, op2);
}


void G1_fixed_exp(G1_t* result, G1_t base, mpz_t* exp, int size) {
  element_pp_t fixed_exp;
  element_pp_init(fixed_exp, base);
  for (int i = 0; i < size; i++) {
    element_pp_pow(result[i], exp[i], fixed_exp);
  }
}

void G2_fixed_exp(G2_t* result, G2_t base, mpz_t* exp, int size) {
  element_pp_t fixed_exp;
  element_pp_init(fixed_exp, base);
  for (int i = 0; i < size; i++) {
    element_pp_pow(result[i], exp[i], fixed_exp);
  }
  element_pp_clear(fixed_exp);
}

void G1_geom_fixed_exp(G1_t* result, G1_t base, mpz_t r, mpz_t prime, int size) {
  element_pp_t fixed_exp;
  mpz_t tmp;

  alloc_init_scalar(tmp);
  mpz_set_ui(tmp, 1);
  element_pp_init(fixed_exp, base);
  for (int i = 0; i < size; i++) {
    element_pp_pow(result[i], tmp, fixed_exp);
    mpz_mul(tmp, tmp, r);
    mpz_mod(tmp, tmp, prime);
  }
  element_pp_clear(fixed_exp);
  clear_scalar(tmp);
}
void G2_geom_fixed_exp(G2_t* result, G2_t base, mpz_t r, mpz_t prime, int size) {
  element_pp_t fixed_exp;
  mpz_t tmp;

  alloc_init_scalar(tmp);
  mpz_set_ui(tmp, 1);
  element_pp_init(fixed_exp, base);
  for (int i = 0; i < size; i++) {
    element_pp_pow(result[i], tmp, fixed_exp);
    mpz_mul(tmp, tmp, r);
    mpz_mod(tmp, tmp, prime);
  }
  element_pp_clear(fixed_exp);
  clear_scalar(tmp);
}

void G1_mul_fixed_exp(G1_t* result, G1_t base, mpz_t* exp, int size) {
  element_pp_t fixed_exp;
  G1_t tmp;
  alloc_init_scalar_G1(tmp);
  element_pp_init(fixed_exp, base);
  for (int i = 0; i < size; i++) {
    element_pp_pow(tmp, exp[i], fixed_exp);
    G1_mul(result[i], result[i], tmp);
  }
  element_pp_clear(fixed_exp);
  clear_scalar_G1(tmp);
}
void G2_mul_fixed_exp(G2_t* result, G2_t base, mpz_t* exp, int size) {
  element_pp_t fixed_exp;
  G2_t tmp;
  alloc_init_scalar_G2(tmp);
  element_pp_init(fixed_exp, base);
  for (int i = 0; i < size; i++) {
    element_pp_pow(tmp, exp[i], fixed_exp);
    G2_mul(result[i], result[i], tmp);
  }
  element_pp_clear(fixed_exp);
  clear_scalar_G2(tmp);
}

void do_pairing(GT_t gt, G1_t g1, G2_t g2) {
  element_pairing(gt, g1, g2);
}







#elif PAIRING_LIB == PAIRING_LIBZM

//G1 element
G1_element::G1_element()
: powers(0)
{
}
G1_element::~G1_element(){
  invalidate();
}
void G1_element::invalidate(){
  if (powers != NULL){
    delete powers;
    powers = NULL;
  }
}
void G1_element::build_powers(){
  if (powers != NULL){
    return;
  }

  powers = new G1_pow_table_t();
  G1_t entry;
  G1_t base2;
  G1_set(base2, this);
  for(int i = 0; i < powers->num_rows; i++){
    G1_set1(entry);
    for(int j = 0; j < powers->num_cols; j++){
      G1_set(powers->tbl[i*powers->num_cols + j],entry);
      G1_mul(entry, entry, base2);
    }
    G1_set(base2, entry);
  }
}

void G1_element::pp_exp(G1_t result, mpz_t exp){
  if (mpz_sgn(exp) == 0){
    G1_set1(result);
    return;
  }

  //Code inspired by element_pow_base_table in libpbc
  int bitno = 0;
  G1_set1(result);
  for(int i = 0; i < powers->num_rows; i++){
    int word = 0;
    for(int j = 0; j < powers->k; j++){
      word |= mpz_tstbit(exp, bitno) << j;
      bitno++;
    }
    if (word > 0){
      G1_mul(result, result, powers->tbl[i*powers->num_cols + word]);
    }
  }
}

//G2_element
G2_element::G2_element()
: powers(0)
{
}
G2_element::~G2_element(){
  invalidate();
}
void G2_element::invalidate(){
  if (powers != NULL){
    delete powers;
    powers = NULL;
  }
}

void G2_element::build_powers(){
  if (powers != NULL){
    return;
  }

  powers = new G2_pow_table_t();
  G2_t entry;
  G2_t base2;
  G2_set(base2, this);
  for(int i = 0; i < powers->num_rows; i++){
    G2_set1(entry);
    for(int j = 0; j < powers->num_cols; j++){
      G2_set(powers->tbl[i*powers->num_cols + j],entry);
      G2_mul(entry, entry, base2);
    }
    G2_set(base2, entry);
  }
}

void G2_element::pp_exp(G2_t result, mpz_t exp){
  if (mpz_sgn(exp) == 0){
    G2_set1(result);
    return;
  }

  //Code inspired by element_pow_base_table in libpbc
  int bitno = 0;
  G2_set1(result);
  for(int i = 0; i < powers->num_rows; i++){
    int word = 0;
    for(int j = 0; j < powers->k; j++){
      word |= mpz_tstbit(exp, bitno) << j;
      bitno++;
    }
    if (word > 0){
      G2_mul(result, result, powers->tbl[i*powers->num_cols + word]);
    }
  }
}

//GT_element
GT_element::GT_element(){
}
GT_element::~GT_element(){
}

void alloc_init_scalar_G1(G1_t s) {
}

void alloc_init_scalar_G2(G2_t s) {
}

void alloc_init_scalar_GT(GT_t s) {
}

void clear_scalar_G1(G1_t s) {
}
void clear_scalar_G2(G2_t s) {
}
void clear_scalar_GT(GT_t s) {
}

template <typename T> void load_zmobject(FILE *fp, T* t, mpz_class tmp){
  mpz_inp_raw(tmp.get_mpz_t(), fp);
  stringstream plaintext;
  plaintext << tmp;
  plaintext >> *t;
}

template <typename T> void dump_zmobject(FILE *fp, T* t, mpz_class tmp){
  stringstream plaintext;
  plaintext << *t;
  plaintext >> tmp;
  mpz_out_raw(fp, tmp.get_mpz_t());
}

template <typename T> void Fpsqrt(T& out, T& in){
  mpz_t power;
  alloc_init_scalar(power);
  //Fp::getModulus() =
  //16798108731015832284940804142231733909889187121439069848933715426072753864723
  //Which is 3 mod 4, so we can do the following.
  //Let (p + 1) / 4 =
  //4199527182753958071235201035557933477472296780359767462233428856518188466181 
  mpz_set_str(power,
  "4199527182753958071235201035557933477472296780359767462233428856518188466181",
  10);

  out = T(1);
  for(int i = 254; i >= 0; i--){
    out = out * out;
    if (mpz_tstbit(power, i)){
      out = out * in;
    }
  }

  clear_scalar(power);
}

template <typename T> void Fp2sqrt(T& out, T& in){
  //Fp::getModulus() =
  //16798108731015832284940804142231733909889187121439069848933715426072753864723
  // then
  //p^2 = 
  //282176456939030335248791854420814157247578133698464773674441028577770932557802396053087865359350024719594752360717198714502904631799257758828132583866729 
  // is indeed 9 mod 16! 
  // So we can uses the "Generalized Atkin-algorithm for q=9 mod 16"
  //from "On the Computation of Square Roots in Finite Fields" 2002 
  // by Signua Muller.

  //Compute (p^2 - 1)/4 = 
  //70544114234757583812197963605203539311894533424616193418610257144442733139450599013271966339837506179898688090179299678625726157949814439707033145966682

  //Compute (p^2 - 9)/16 = 
  //17636028558689395953049490901300884827973633356154048354652564286110683284862649753317991584959376544974672022544824919656431539487453609926758286491670

  //Furthermore, I verified using zmlib that
  //Fp2(1,1)^((p^2 - 1)/2) = -1
  //which means that Fp2(1,1) is a quadratic nonresidue
  //And, of course, Fp2(1) is a quadratic residue. So we're set.

  mpz_t tmp;
  alloc_init_scalar(tmp);

  T twoQ = in + in;
  mpz_set_str(tmp,
  "70544114234757583812197963605203539311894533424616193418610257144442733139450599013271966339837506179898688090179299678625726157949814439707033145966682",
  10);

  T s = T(1);
  for(int i = 508; i >= 0; i--){
    s = s * s;
    if (mpz_tstbit(tmp, i)){
      s = s * twoQ;
    }
  }

  T d;
  if (s == T(1)){
    d = T(1,1); // Use a quadratic nonresidue
  } else if (s == T(-1)){
    d = T(1,0); // Use a quadratic residue
  } else {
    cout << in << endl;
    cout << s << endl;
    cout << "2Q^((p^2-1)/4) was not in {-1,1}!" << endl;
    //exit(1);
  }

  T twoQdSquared = twoQ * d * d;
  mpz_set_str(tmp,
  "17636028558689395953049490901300884827973633356154048354652564286110683284862649753317991584959376544974672022544824919656431539487453609926758286491670",
  10);

  T z = T(1);
  for(int i = 508; i >= 0; i--){
    z = z * z;
    if (mpz_tstbit(tmp, i)){
      z = z * twoQdSquared;
    }
  }

  T eye = twoQdSquared*z*z;

  //Answers are +/- zdQ(eye-1)
  out = z * d * in * (eye - T(1));

  clear_scalar(tmp);
}

typedef struct {
  uint32_t bit[8]; //254 + 2 bits
} compressedG1_t;

typedef struct {
  uint32_t bit[16]; //254 * 2 + 2 bits
} compressedG2_t;

void load_compressed_G1(FILE* fp, G1_t q){
  compressedG1_t cg;

  if (fread(&cg, sizeof(compressedG1_t), 1, fp) <= 0) {
    cout << "ERROR: cannot read from G1" << endl;
  }

  mpz_class tmp;
  int bitptr = 0;
  for(unsigned int i = 0; i < 8; i++){
    for(unsigned int j = 0; j < 32; j++){
      if ((cg.bit[i] >> j) & 1){
        mpz_setbit(tmp.get_mpz_t(), bitptr);
      }
      bitptr++;
    }
  }

  int whichSqrRoot = mpz_tstbit(tmp.get_mpz_t(), 254);
  int Z = mpz_tstbit(tmp.get_mpz_t(), 255);
  mpz_clrbit(tmp.get_mpz_t(), 254);
  mpz_clrbit(tmp.get_mpz_t(), 255);

  stringstream plaintext;
  plaintext << tmp;
  plaintext >> q->value[0];

  if (Z == 0){
    q->value[2] = Fp(0);
  } else {
    q->value[2] = Fp(1);
  }

  Fp X = q->value[0];
  if (Z){
    Fp Ysquared = X * X * X + Fp(2);
    //The same
    //cout << Ysquared << endl;
    //cout << v[i]->value[1] * v[i]->value[1] << endl;
    Fp Y;
    Fpsqrt(Y, Ysquared);
    //cout << Y << endl;
    //cout << Fp(0) - Y << endl;
    //cout << v[i]->value[1] << endl;
    //cout << endl;
    if (whichSqrRoot == 0){
      q->value[1] = Y;
    } else {
      q->value[1] = Fp(0) - Y;
    }
  } else {
    q->value[1] = Fp(0);
  }

  q->invalidate();

}


void dump_compressed_G1(FILE* fp, G1_t q){
  ecop::NormalizeJac(q->value, q->value);

  //q->value[2] is either 0 or 1
  int Z = 0;
  if (q->value[2] == Fp(0)){
    Z = 0;
  } else if (q->value[2] == Fp(1)){
    Z = 1;
  } else {
    cout << "ERROR: normalizing jacobian failed" << endl;
    exit(1);
  }
  Fp X = q->value[0];
  int whichSqrRoot = 0;
  if (Z){
    Fp Ysquared = X * X * X + Fp(2);
    //The same
    //cout << Ysquared << endl;
    //cout << v[i]->value[1] * v[i]->value[1] << endl;
    Fp Y;
    Fpsqrt(Y, Ysquared);
    //cout << Y << endl;
    //cout << Fp(0) - Y << endl;
    //cout << v[i]->value[1] << endl;
    //cout << endl;
    if (Y == q->value[1]){
      whichSqrRoot = 0;
    } else if (Fp(0) - Y == q->value[1]){
      whichSqrRoot = 1;
    } else {
      cout << "ERROR: Square root failure loading G1" << endl;
      exit(1);
    }
  }

  mpz_class tmp;
  stringstream plaintext;
  plaintext << X;
  plaintext >> tmp;

  //Set the 255'th bit (index 254) to whichSqrRoot
  //and the 256'th bit (index 255) to Z
  if (whichSqrRoot)
    mpz_setbit(tmp.get_mpz_t(), 254);
  if (Z)
    mpz_setbit(tmp.get_mpz_t(), 255);

  compressedG1_t cg;
  int bitptr = 0;
  for(unsigned int i = 0; i < 8; i++){
    cg.bit[i] = 0;
    for(unsigned int j = 0; j < 32; j++){
      cg.bit[i] |= (mpz_tstbit(tmp.get_mpz_t(),bitptr) << j);
      mpz_clrbit(tmp.get_mpz_t(), bitptr);
      bitptr++;
    }
  }

  //Assertion
  if (tmp != 0){
    cout << "ERROR: Did not print out all bits of temporary mpz_t" <<
    endl;
    exit(1);
  }

  //mpz_out_raw(fp, tmp.get_mpz_t());
  fwrite(&cg, sizeof(compressedG1_t), 1, fp);
}

void load_compressed_G2(FILE* fp, G2_t q){
  compressedG2_t cg;
  if (fread(&cg, sizeof(compressedG2_t), 1, fp) <= 0) {
    cout << "ERROR: cannot read from G1" << endl;
  }

  mpz_class tmpX1;
  mpz_class tmpX2;
  int whichSqrRoot = 0;
  int Z = 0;

  int bitptr = 0;
  for(unsigned int i = 0; i < 16; i++){
    for(unsigned int j = 0; j < 32; j++){
      if ((cg.bit[i] >> j) & 1){
        if (bitptr < 254){
          mpz_setbit(tmpX1.get_mpz_t(), bitptr);
        } else if (bitptr < 508){
          mpz_setbit(tmpX2.get_mpz_t(), bitptr - 254);
        } else if (bitptr == 508){
          whichSqrRoot = 1;
        } else if (bitptr == 509){
          Z = 1;
        }
      }
      bitptr++;
    }
  }

  {
    stringstream plaintext;
    plaintext << tmpX1;
    plaintext >> q->value[0].a_;
  }
  {
    stringstream plaintext;
    plaintext << tmpX2;
    plaintext >> q->value[0].b_;
  }

  if (Z == 0){
    q->value[2] = Fp2(0);
  } else {
    q->value[2] = Fp2(1);
  }

  Fp2 X = q->value[0];
  if (Z){
    Fp2 Ysquared = X * X * X + ParamT<Fp2>::b_invxi;
    //The same
    //cout << Ysquared << endl;
    //cout << v[i]->value[1] * v[i]->value[1] << endl;
    Fp2 Y;
    Fp2sqrt(Y, Ysquared);
    //cout << Y << endl;
    //cout << Fp(0) - Y << endl;
    //cout << v[i]->value[1] << endl;
    //cout << endl;
    if (whichSqrRoot == 0){
      q->value[1] = Y;
    } else {
      q->value[1] = Fp2(0) - Y;
    }
  } else {
    q->value[1] = Fp2(0);
  }

  q->invalidate();
}


void dump_compressed_G2(FILE* fp, G2_t q){
  ecop::NormalizeJac(q->value, q->value);

  //q->value[2] is either 0 or 1
  int Z = 0;
  if (q->value[2] == Fp2(0)){
    Z = 0;
  } else if (q->value[2] == Fp2(1)){
    Z = 1;
  } else {
    cout << "ERROR: normalizing jacobian failed" << endl;
    exit(1);
  }
  Fp2 X = q->value[0];
  int whichSqrRoot = 0;
  if (Z){
    Fp2 Ysquared = X * X * X + ParamT<Fp2>::b_invxi;
    if (Ysquared != q->value[1] * q->value[1]){
      cout << q->value[0] << endl << q->value[1] << endl
      <<q->value[2] << endl;
      cout << "Error in computing Ysquared!" << endl;
    }
    //The same
    //cout << Ysquared << endl;
    //cout << q->value[1] * q->value[1] << endl;
    //cout << endl;
    Fp2 Y;
    Fp2sqrt(Y, Ysquared);
    //cout << Y << endl;
    //cout << Fp2(0) - Y << endl;
    //cout << q->value[1] << endl;
    //cout << endl;
    if (Y == q->value[1]){
      whichSqrRoot = 0;
    } else if (Fp2(0) - Y == q->value[1]){
      whichSqrRoot = 1;
    } else {
      cout << "ERROR: Square root failure loading G2" << endl;
      exit(1);
    }
  }

  mpz_class tmpX1;
  mpz_class tmpX2;
  {
    stringstream plaintext;
    plaintext << X.a_;
    plaintext >> tmpX1;
  }
  {
    stringstream plaintext;
    plaintext << X.b_;
    plaintext >> tmpX2;
  }

  compressedG2_t cg;
  int bitptr = 0;
  for(unsigned int i = 0; i < 16; i++){
    cg.bit[i] = 0;
    for(unsigned int j = 0; j < 32; j++){
      uint32_t bit = 0;

      if (bitptr < 254){
        bit = mpz_tstbit(tmpX1.get_mpz_t(),bitptr);
      } else if (bitptr < 508){
        bit = mpz_tstbit(tmpX2.get_mpz_t(),bitptr - 254);
      } else if (bitptr == 508){
        bit = whichSqrRoot;
      } else if (bitptr == 509){
        bit = Z;
      }
      cg.bit[i] |= bit << j;
      bitptr++;
    }
  }

  //mpz_out_raw(fp, tmp.get_mpz_t());
  fwrite(&cg, sizeof(compressedG2_t), 1, fp);
}

void load_scalar_G1(G1_t q, const char *scalar_name, const char *folder_name) {
  FILE* fp;
  open_file(&fp, scalar_name, "r", folder_name);

  mpz_class tmp;

  load_zmobject(fp, &q->value[0], tmp);
  load_zmobject(fp, &q->value[1], tmp);
  load_zmobject(fp, &q->value[2], tmp);
  q->invalidate();

  fclose(fp);
}
void load_scalar_G2(G2_t q, const char *scalar_name, const char *folder_name) {
  FILE* fp;
  open_file(&fp, scalar_name, "r", folder_name);

  mpz_class tmp;

  load_zmobject(fp, &q->value[0], tmp);
  load_zmobject(fp, &q->value[1], tmp);
  load_zmobject(fp, &q->value[2], tmp);
  q->invalidate();

  fclose(fp);
}
void load_scalar_GT(GT_t q, const char *scalar_name, const char *folder_name) {
  FILE* fp;
  open_file(&fp, scalar_name, "r", folder_name);

  mpz_class tmp;

  load_zmobject(fp, &q->value[0], tmp);

  fclose(fp);
}


void dump_scalar_G1(G1_t q, char *scalar_name, const char *folder_name) {
  FILE* fp;
  open_file(&fp, scalar_name, "w", folder_name);

  mpz_class tmp;

  ecop::NormalizeJac(q->value, q->value);
  dump_zmobject(fp, &q->value[0], tmp);
  dump_zmobject(fp, &q->value[1], tmp);
  dump_zmobject(fp, &q->value[2], tmp);

  fclose(fp);
}

void dump_scalar_G2(G2_t q, char *scalar_name, const char *folder_name) {
  ecop::NormalizeJac(q->value, q->value);

  FILE* fp;
  open_file(&fp, scalar_name, "w", folder_name);

  mpz_class tmp;

  dump_zmobject(fp, &q->value[0], tmp);
  dump_zmobject(fp, &q->value[1], tmp);
  dump_zmobject(fp, &q->value[2], tmp);

  fclose(fp);
}
void dump_scalar_GT(GT_t q, char *scalar_name, const char *folder_name) {
  FILE* fp;
  open_file(&fp, scalar_name, "w", folder_name);

  mpz_class tmp;

  dump_zmobject(fp, &q->value[0], tmp);

  fclose(fp);
}

void load_vector_G1(int size, G1_t *v, const char *vec_name, const char *folder_name) {

  FILE* fp;
  open_file(&fp, vec_name, "r", folder_name);

  mpz_class tmp;
  for (int i = 0; i < size; i++) {
#if COMPRESS_PROOF == 1
    load_compressed_G1(fp, v[i]);
#else
    load_zmobject(fp, &v[i]->value[0], tmp);
    load_zmobject(fp, &v[i]->value[1], tmp);
    load_zmobject(fp, &v[i]->value[2], tmp);
    v[i]->invalidate();

/*
    if (test->value[0] != v[i]->value[0]
    || test->value[1] != v[i]->value[1]
    || test->value[2] != v[i]->value[2]
    ){
      cout << test->value[0] << endl
          << test->value[1] << endl
          << test->value[2] << endl;
      cout << v[i]->value[0] << endl
          << v[i]->value[1] << endl
          << v[i]->value[2] << endl;
      cout << "DIFFERENCE" << endl;
      exit(1);
    }
    */
#endif
  }

  fclose(fp);
}
void load_vector_G2(int size, G2_t *v, const char *vec_name, const char *folder_name) {

  FILE* fp;
  open_file(&fp, vec_name, "r", folder_name);

  mpz_class tmp;
  for (int i = 0; i < size; i++) {
#if COMPRESS_PROOF == 1
    load_compressed_G2(fp, v[i]);
#else
    load_zmobject(fp, &v[i]->value[0].a_, tmp);
    load_zmobject(fp, &v[i]->value[0].b_, tmp);
    load_zmobject(fp, &v[i]->value[1].a_, tmp);
    load_zmobject(fp, &v[i]->value[1].b_, tmp);
    load_zmobject(fp, &v[i]->value[2].a_, tmp);
    load_zmobject(fp, &v[i]->value[2].b_, tmp);
    v[i]->invalidate();

/*
    if (test->value[0] != v[i]->value[0]
    || test->value[1] != v[i]->value[1]
    || test->value[2] != v[i]->value[2]
    ){
      cout << test->value[0] << endl
          << test->value[1] << endl
          << test->value[2] << endl;
      cout << v[i]->value[0] << endl
          << v[i]->value[1] << endl
          << v[i]->value[2] << endl;
      cout << "DIFFERENCE" << endl;
      exit(1);
    }*/
#endif
  }

  fclose(fp);
}
void load_vector_GT(int size, GT_t *v, const char *vec_name, const char *folder_name) {

  FILE* fp;
  open_file(&fp, vec_name, "r", folder_name);

  mpz_class tmp;
  for (int i = 0; i < size; i++) {
    load_zmobject(fp, &v[i]->value[0], tmp);
  }

  fclose(fp);
}


void dump_vector_G1(int size, G1_t *v, const char *vec_name, const char *folder_name) {

  FILE* fp;
  open_file(&fp, vec_name, "w", folder_name);

  mpz_class tmp;

  for (int i = 0; i < size; i++) {
#if COMPRESS_PROOF == 1
    dump_compressed_G1(fp, v[i]);
#else
    ecop::NormalizeJac(v[i]->value, v[i]->value);
    dump_zmobject(fp, &v[i]->value[0], tmp);
    dump_zmobject(fp, &v[i]->value[1], tmp);
    dump_zmobject(fp, &v[i]->value[2], tmp);

#endif
  }
  fclose(fp);
}
void dump_vector_G2(int size, G2_t *v, const char *vec_name, const char *folder_name) {

  FILE* fp;
  open_file(&fp, vec_name, "w", folder_name);

  mpz_class tmp;

  for (int i = 0; i < size; i++) {
#if COMPRESS_PROOF == 1
    dump_compressed_G2(fp, v[i]);
#else
    ecop::NormalizeJac(v[i]->value, v[i]->value);
    dump_zmobject(fp, &v[i]->value[0].a_, tmp);
    dump_zmobject(fp, &v[i]->value[0].b_, tmp);
    dump_zmobject(fp, &v[i]->value[1].a_, tmp);
    dump_zmobject(fp, &v[i]->value[1].b_, tmp);
    dump_zmobject(fp, &v[i]->value[2].a_, tmp);
    dump_zmobject(fp, &v[i]->value[2].b_, tmp);
#endif
  }
  fclose(fp);
}
void dump_vector_GT(int size, GT_t *v, const char *vec_name, const char *folder_name) {
  FILE* fp;
  open_file(&fp, vec_name, "w", folder_name);

  mpz_class tmp;

  for (int i = 0; i < size; i++) {
    dump_zmobject(fp, &v[i]->value[0], tmp);
  }
  fclose(fp);
}

void init_pairing_from_file(const char * filename, mpz_t _prime) {
  // load prime
  mpz_set(prime, _prime);

  prng = new Prng(PNG_CHACHA);

  // init my library
  Param::init(-1);

  mpz_class prime1_(prime);
  stringstream prime2;
  prime2 << Param::r;
  mpz_class prime2_(prime2.str());
  if (prime1_ != prime2_){
    cout << "Wrong prime for zmlib. Use " << prime2.str() << endl;
  }

  // prepair a generator
  g2->value[0] =
        Fp2(
                Fp("12723517038133731887338407189719511622662176727675373276651903807414909099441"),
                Fp("4168783608814932154536427934509895782246573715297911553964171371032945126671")
        );
  g2->value[1]= Fp2(
                Fp("13891744915211034074451795021214165905772212241412891944830863846330766296736"),
                Fp("7937318970632701341203597196594272556916396164729705624521405069090520231616")
        );
  g2->value[2] = Fp2(
                Fp("1"),
                Fp("0")
        );
  g1->value[0] =
        Fp("1674578968009266105367653690721407808692458796109485353026408377634195183292");
  g1->value[1] =
        Fp("8299158460239932124995104248858950945965255982743525836869552923398581964065");
  g1->value[2] = Fp("1");

  ecop::ScalarMult(g1_one->value, g1->value, Param::r);
  //ecop::NormalizeJac(g1_one->value, g1_one->value);
  ecop::ScalarMult(g2_one->value, g2->value, Param::r);
  //ecop::NormalizeJac(g2_one->value, g2_one->value);

  g1->invalidate();
  g2->invalidate();
  g1_one->invalidate();
  g2_one->invalidate();
}

template <int N> void multi_exponentiation_G1_N(G1_t result, G1_t* base, mpz_t* exponents) {
  G1_set1(result);

  int s = 0;
  int allzero = 1;
  for(int i = 0; i < N; i++){
    int si = mpz_sizeinbase(exponents[i],2) - 1;
    if (si > s){
      s = si;
    }
    if (mpz_sgn(exponents[i]) != 0){
      allzero = 0;
    }
  }
  if (allzero){
    return;
  }

  G1_t tmp;
  alloc_init_scalar_G1(tmp);

  int table_length = 1 << N;
  G1_t table [table_length];
  if (N == 2){
    G1_set1(table[0]);
    G1_set(table[1], base[0]);
    G1_set(table[2], base[1]);
    G1_mul(table[3], table[1], table[2]);
  } else if (N == 3){
    G1_set1(table[0]);
    G1_set(table[1], base[0]);
    G1_set(table[2], base[1]);
    G1_mul(table[3], table[1], table[2]);
    G1_set(table[4], base[2]);
    G1_mul(table[5], table[1], table[4]);
    G1_mul(table[6], table[2], table[4]);
    G1_mul(table[7], table[1], table[6]);
  } else {
    for(int i = 0; i < table_length; i++){
      G1_set1(table[i]);
    }
    //TODO: This could be optimized.
    for(int i = 0; i < table_length; i++){
      for(int j = 0; j < N; j++){
        int mask = 1 << j;
        if (i & mask){
          G1_mul(table[i], table[i-mask], base[j]);
        }
      }
    }
  }

  for(; s >= 0; s--){
    G1_mul(result, result, result);

    int w = 0;
    for(int i = 0; i < N; i++){
      w |= mpz_tstbit(exponents[i], s) << i;
    }

    if (w > 0){
      G1_mul(result, result, table[w]);
    }
  }

  /*
  for(int i = 0; i < N; i++){
    G1_exp(tmp, base[i], exponents[i]);
    G1_mul(result, result, tmp);
  }
  */

  clear_scalar_G1(tmp);
}


template <int N> void multi_exponentiation_G2_N(G2_t result, G2_t* base, mpz_t* exponents) {
  G2_set1(result);

  int s = 0;
  int allzero = 1;
  for(int i = 0; i < N; i++){
    int si = mpz_sizeinbase(exponents[i],2) - 1;
    if (si > s){
      s = si;
    }
    if (mpz_sgn(exponents[i]) != 0){
      allzero = 0;
    }
  }
  if (allzero){
    return;
  }

  G2_t tmp;
  alloc_init_scalar_G2(tmp);

  int table_length = 1 << N;
  G2_t table [table_length];
  if (N == 2){
    G2_set1(table[0]);
    G2_set(table[1], base[0]);
    G2_set(table[2], base[1]);
    G2_mul(table[3], table[1], table[2]);
  } else if (N == 3){
    G2_set1(table[0]);
    G2_set(table[1], base[0]);
    G2_set(table[2], base[1]);
    G2_mul(table[3], table[1], table[2]);
    G2_set(table[4], base[2]);
    G2_mul(table[5], table[1], table[4]);
    G2_mul(table[6], table[2], table[4]);
    G2_mul(table[7], table[1], table[6]);
  } else {
    for(int i = 0; i < table_length; i++){
      G2_set1(table[i]);
    }
    //TODO: This could be optimized.
    for(int i = 0; i < table_length; i++){
      for(int j = 0; j < N; j++){
        int mask = 1 << j;
        if (i & mask){
          G2_mul(table[i], table[i-mask], base[j]);
        }
      }
    }
  }

  for(; s >= 0; s--){
    G2_mul(result, result, result);

    int w = 0;
    for(int i = 0; i < N; i++){
      w |= mpz_tstbit(exponents[i], s) << i;
    }

    if (w > 0){
      G2_mul(result, result, table[w]);
    }
  }

  /*
  for(int i = 0; i < N; i++){
    G2_exp(tmp, base[i], exponents[i]);
    G2_mul(result, result, tmp);
  }
  */

  clear_scalar_G2(tmp);
}

// using multi-exponentiation
void multi_exponentiation_G1(int size, G1_t *base, mpz_t *exponents, G1_t result) {

  G1_t tmp;
  alloc_init_scalar_G1(tmp);
  G1_set1(result);
#ifdef MULTIEXP
  int i = 0;
  for (; i + 2 < size; i += 3) {
    multi_exponentiation_G1_N<3>(tmp, base + i, exponents + i);
    //is this correct? yes, it is result = result * tmp
    G1_mul(result, result, tmp);
  }
  for (; i + 1 < size; i += 2) {
    multi_exponentiation_G1_N<2>(tmp, base + i, exponents + i);
    //is this correct? yes, it is result = result * tmp
    G1_mul(result, result, tmp);
  }
  for (; i < size; i += 1) {
    //multi_exponentiation_G1_N<1>(tmp, base + i, exponents + i);
    G1_exp(tmp, base[i], exponents[i]);
    //is this correct? yes, it is result = result * tmp
    G1_mul(result, result, tmp);
  }
#else
  for (int i = 0; i < size; i++) {
    G1_exp(tmp, base[i], exponents[i]);
    G1_mul(result, result, tmp);
  }
#endif
  clear_scalar_G1(tmp);
}

void multi_exponentiation_G2(int size, G2_t *base, mpz_t *exponents, G2_t result) {

  G2_t tmp;
  alloc_init_scalar_G2(tmp);
  G2_set1(result);
#ifdef MULTIEXP
  int i = 0;
  for (; i + 2 < size; i += 3) {
    multi_exponentiation_G2_N<3>(tmp, base + i, exponents + i);
    //is this correct? yes, it is result = result * tmp
    G2_mul(result, result, tmp);
  }
  for (; i + 1 < size; i += 2) {
    multi_exponentiation_G2_N<2>(tmp, base + i, exponents + i);
    //is this correct? yes, it is result = result * tmp
    G2_mul(result, result, tmp);
  }
  for (; i < size; i += 1) {
    //multi_exponentiation_G2_N<1>(tmp, base + i, exponents + i);
    G2_exp(tmp, base[i], exponents[i]);
    //is this correct? yes, it is result = result * tmp
    G2_mul(result, result, tmp);
  }
#else
  for (int i = 0; i < size; i++) {
    G2_exp(tmp, base[i], exponents[i]);
    G2_mul(result, result, tmp);
  }
#endif
  clear_scalar_G2(tmp);

}

void G1_set(G1_t e1, G1_t e2) {
  for(int i = 0; i < 3; i++){
    e1->value[i] = e2->value[i];
  }
  e1->invalidate();
}

void G2_set(G2_t e1, G2_t e2) {
  for(int i = 0; i < 3; i++){
    e1->value[i] = e2->value[i];
  }
  e1->invalidate();
}

void GT_set(GT_t e1, GT_t e2) {
  e1->value[0] = e2->value[0];
}

void G1_set1(G1_t e1) {
  G1_set(e1, g1_one);
}

void G2_set1(G2_t e1) {
  G2_set(e1, g2_one);
}

void G1_mul(G1_t rop, G1_t op1, G1_t op2) {
  //libzm does not allow the destination to be the same
  //as either source.
  G1_t out;
  if (op1 == op2){
    ecop::ECDouble(out->value, op1->value);
  } else {
    ecop::ECAdd(out->value, op1->value, op2->value);
  }
  //ecop::NormalizeJac(rop->value, out->value);
  G1_set(rop, out);
  rop->invalidate();
}

void G2_mul(G2_t rop, G2_t op1, G2_t op2) {
  //libzm does not allow the destination to be the same
  //as either source.
  G2_t out;
  if (op1 == op2){
    ecop::ECDouble(out->value, op1->value);
  } else {
    ecop::ECAdd(out->value, op1->value, op2->value);
  }
  //ecop::NormalizeJac(rop->value, out->value);
  G2_set(rop, out);
  rop->invalidate();
}

void GT_mul(GT_t rop, GT_t op1, GT_t op2) {
  rop->value[0] = op1->value[0] * op2->value[0];
}

void G1_exp(G1_t rop, G1_t op1, mpz_t exp) {
  mpz_class exp_(exp);

//libzm does not allow the destination to be the same
//as either source.
  G1_t out;
  ecop::ScalarMult(out->value, op1->value, exp_);
  //ecop::NormalizeJac(rop->value, out->value);
  G1_set(rop, out);
  rop->invalidate();
}

void G2_exp(G2_t rop, G2_t op1, mpz_t exp) {
  mpz_class exp_(exp);

//libzm does not allow the destination to be the same
//as either source.
  G2_t out;
  ecop::ScalarMult(out->value, op1->value, exp_);
  G2_set(rop, out);
  rop->invalidate();
}



void G1_random(G1_t e) {
  // g1 is a generator
  mpz_t exp;
  alloc_init_scalar(exp);
  // gen random exp
  prng->get_random(exp, prime);
  G1_exp(e, g1, exp);
  clear_scalar(exp);
}

void G2_random(G2_t e) {
  // g2 is a generator
  mpz_t exp;
  alloc_init_scalar(exp);
  // gen random exp
  prng->get_random(exp, prime);
  G2_exp(e, g2, exp);
  clear_scalar(exp);
}

//void GT_random(GT_t e) {
  //element_random(e);
//}

void G1_print(G1_t op){
  cout << op->value[0] << endl << op->value[1] << endl << op->value[2] << endl;
}

void G2_print(G2_t op){
  cout << op->value[0] << endl << op->value[1] << endl << op->value[2] << endl;
}

int G1_cmp(G1_t op1, G1_t op2) {
  ecop::NormalizeJac(op1->value, op1->value);
  ecop::NormalizeJac(op2->value, op2->value);
  if (op1->value[2] != 0){
    for(int i = 0; i < 3; i++){
      if (op1->value[i] != op2->value[i]){
        return 1;
      }
    }
    return 0;
  }
  return !(op1->value[2] == op2->value[2]);
}
int G2_cmp(G2_t op1, G2_t op2) {
  ecop::NormalizeJac(op1->value, op1->value);
  ecop::NormalizeJac(op2->value, op2->value);
  if (op1->value[2] != 0){
    for(int i = 0; i < 3; i++){
      if (op1->value[i] != op2->value[i]){
        return 1;
      }
    }
    return 0;
  }
  return !(op1->value[2] == op2->value[2]);
}

int GT_cmp(GT_t op1, GT_t op2) {
  if (op1->value[0] == op2->value[0]) {
    return 0;
  } else {
    return 1;
  }
}


void G1_fixed_exp(G1_t* result, G1_t base, mpz_t* exp, int size) {
  base->build_powers();

  for (int i = 0; i < size; i++) {
    base->pp_exp(result[i], exp[i]);
    //G1_exp(result[i], base, exp[i]);
  }
}

void G2_fixed_exp(G2_t* result, G2_t base, mpz_t* exp, int size) {
  base->build_powers();

  for (int i = 0; i < size; i++) {
    base->pp_exp(result[i], exp[i]);
    //G2_exp(result[i], base, exp[i]);
  }
}

void G1_mul_fixed_exp(G1_t* result, G1_t base, mpz_t* exp, int size) {
  base->build_powers();

  G1_t tmp;
  alloc_init_scalar_G1(tmp);
  for (int i = 0; i < size; i++) {
    base->pp_exp(tmp, exp[i]);
    //G1_exp(tmp, base, exp[i]);
    G1_mul(result[i], result[i], tmp);
  }
  clear_scalar_G1(tmp);
}
void G2_mul_fixed_exp(G2_t* result, G2_t base, mpz_t* exp, int size) {
  base->build_powers();

  G2_t tmp;
  alloc_init_scalar_G2(tmp);
  for (int i = 0; i < size; i++) {
    base->pp_exp(tmp, exp[i]);
    //G2_exp(tmp, base, exp[i]);
    G2_mul(result[i], result[i], tmp);
  }
  clear_scalar_G2(tmp);
}

void G1_geom_fixed_exp(G1_t* result, G1_t base, mpz_t r, mpz_t prime, int size) {
  base->build_powers();

  mpz_t tmp;
  alloc_init_scalar(tmp);
  mpz_set_ui(tmp, 1);
  for (int i = 0; i < size; i++) {
    base->pp_exp(result[i], tmp);
    //G1_exp(result[i], base, tmp);
    mpz_mul(tmp, tmp, r);
    mpz_mod(tmp, tmp, prime);
  }
  clear_scalar(tmp);
}
void G2_geom_fixed_exp(G2_t* result, G2_t base, mpz_t r, mpz_t prime, int size) {
  base->build_powers();

  mpz_t tmp;
  alloc_init_scalar(tmp);
  mpz_set_ui(tmp, 1);
  for (int i = 0; i < size; i++) {
    base->pp_exp(result[i], tmp);
    //G2_exp(result[i], base, tmp);
    mpz_mul(tmp, tmp, r);
    mpz_mod(tmp, tmp, prime);
  }
  clear_scalar(tmp);
}

void do_pairing(GT_t gt, G1_t g1, G2_t g2) {
  //Normalization is free unless the values aren't normalized
  //This needs to happen because we do an exponentiation in the test...
  ecop::NormalizeJac(g1->value, g1->value);
  ecop::NormalizeJac(g2->value, g2->value);
  //Reversed operands is correct - GT, G2, then G1.
  opt_atePairingJac<Fp>(gt->value[0], g2->value, g1->value);
}

#endif //switch on pairing lib
#endif //noninteractive == 1
