#include <apps_sfdl_hw/sha_1_p_exo.h>
#include <apps_sfdl_gen/sha_1_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

sha_1ProverExo::sha_1ProverExo() { }

//Rotate x m many positions left, in x's big endian encoding.
uint32_t LROT(uint32_t x, uint32_t m){
  return (x << m) | (x >> (32 - m));
}

void sha_1ProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {

  int i;
  uint32_t w[80];
  uint32_t a,b,c,d,e,f,k,temp;
  uint32_t h[5];

  for(i = 0; i < 16; i++){
    w[i] = mpz_get_ui(mpq_numref(input_q[i]));
  }

  //State machine 16 to 79
  for(i = 16; i < 80; i++){
    w[i] = LROT(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);
  }
  //Initialize h
  h[0] = 0x67452301;
  h[1] = 0xEFCDAB89;
  h[2] = 0x98BADCFE;
  h[3] = 0x10325476;
  h[4] = 0xC3D2E1F0;

  //hash values
  a = h[0];
  b = h[1];
  c = h[2];
  d = h[3];
  e = h[4];

//Main loop:
  for(i = 0; i < 80; i++){
    if (0 <= i && i <= 19){
      f = (b & c) | ((~b) & d);
      k = 0x5A827999;
    }
    if (20 <= i && i <= 39){
      f = b ^ c ^ d;
      k = 0x6ED9EBA1;
    }
    if (40 <= i && i <= 59){
      f = (b & c) | (b & d) | (c & d);
      k = 0x8F1BBCDC;
    }
    if (60 <= i && i <= 79){
      f = b ^ c ^ d;
      k = 0xCA62C1D6;
    }

    temp = LROT(a,5) + f + e + k + w[i];
    e = d;
    d = c;
    c = LROT(b,30);
    b = a;
    a = temp;
  }

  //Add a-e to the h
  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;

  //Return value
  mpq_set_si(output_recomputed[0], 0, 1);
  //sha1
  mpq_t* sha1 = output_recomputed + 1;
  for(i = 0; i < 5; i++) {
    mpq_set_ui(sha1[i], h[i], 1);
  }
}

//Refer to apps_sfdl_gen/sha_1_cons.h for constants to use in this exogenous
//check.
bool sha_1ProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {
 bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      passed_test = false;
      break;
    }
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};
