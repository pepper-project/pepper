#include <apps_sfdl_hw/mm_sfdl_p_exo.h>
#include <apps_sfdl_gen/mm_sfdl_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mm_sfdlProverExo::mm_sfdlProverExo() { }

void mm_sfdlProverExo::baseline(const mpq_t* input_q, int num_inputs,
                                mpq_t* output_recomputed, int num_outputs) {

  int m = mm_sfdl_cons::m;
  const mpq_t* A = input_q;
  const mpq_t* B = input_q + m*m;
  const mpq_t* C = output_recomputed;

  mpz_t temp;
  alloc_init_scalar(temp);

  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      mpz_set_ui(temp, 0);
      for(int k = 0; k < m; k++) {
        mpz_mul(temp, mpq_numref(A[i*m + k]), mpq_numref(B[k*m + j]));
        mpz_add(mpq_numref(output_recomputed[i*m+j]), mpq_numref(output_recomputed[i*m+j]), temp);
      }
    }
  }
  clear_scalar(temp);
}

//Refer to apps_sfdl_gen/mm_sfdl_cons.h for constants to use in this exogenous
//check.
bool mm_sfdlProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
                                       int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool success = true;

#ifdef ENABLE_EXOGENOUS_CHECKING
  int m = mm_sfdl_cons::m;
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, m*m);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  const mpq_t* C = output_q;

  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      if (mpz_cmp(mpq_numref(C[i*m + j]), mpq_numref(output_recomputed[i*m+j])) != 0) {
        success = false;
        break;
      }
    }
  }
  clear_vec(m*m, output_recomputed);
#else
  cout<<"Exogeneous checking disabled"<<endl;
#endif
  return success;
};
