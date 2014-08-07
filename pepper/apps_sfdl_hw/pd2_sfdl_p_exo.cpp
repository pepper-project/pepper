#include <apps_sfdl_hw/pd2_sfdl_p_exo.h>
#include <apps_sfdl_gen/pd2_sfdl_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

pd2_sfdlProverExo::pd2_sfdlProverExo() { }

void pd2_sfdlProverExo::baseline(const mpq_t* input_q, int num_inputs,
                                 mpq_t* output_recomputed, int num_outputs) {

  mpq_t t2, f;
  mpq_init(t2);
  mpq_init(f);

  const mpq_t *a = input_q;
  int rctr = 0;
  int m = pd2_sfdl_cons::m;

  mpq_set_si(f, 0, 1);
  for(int i = 0; i < m; i++) {
    mpq_set_si(t2, pd2_sfdl_cons::poly_FUNCTION_STATIC_RANDOM_INT[rctr++], 1);
    mpq_mul(t2, t2, (a[i]));
    mpq_add(f, f, t2);
  }

  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      mpq_set_si(t2, pd2_sfdl_cons::poly_FUNCTION_STATIC_RANDOM_INT[rctr++], 1);
      mpq_mul(t2, t2, (a[i]));
      mpq_mul(t2, t2, (a[j]));
      mpq_add(f, f, t2);
    }
  }

  mpq_set((output_recomputed[0]), f);

  mpq_clear(t2);
  mpq_clear(f);
}

//Refer to apps_sfdl_gen/pd2_sfdl_cons.h for constants to use in this exogenous
//check.
bool pd2_sfdlProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
                                        int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);
  bool success = mpq_cmp((output_recomputed[0]), (output_q[0])) == 0;
  clear_vec(num_outputs, output_recomputed);
  return success;
};
