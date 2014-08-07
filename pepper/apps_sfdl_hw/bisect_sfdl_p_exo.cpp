#include <apps_sfdl_hw/bisect_sfdl_p_exo.h>
#include <apps_sfdl_gen/bisect_sfdl_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

bisect_sfdlProverExo::bisect_sfdlProverExo() { }

void bisect_sfdlProverExo::exogenous_fAtMidpt(mpq_t& f, mpq_t* a, mpq_t* b) {
  mpq_t t2;
  mpq_t t3;

  alloc_init_scalar(t2);
  alloc_init_scalar(t3);

  mpq_set_ui(f, 0, 1);
  int m = bisect_sfdl_cons::m;

  int rctr = 0;
  for(int i = 0; i < m; i++) {
    mpq_add(t2, a[i], b[i]);
    mpq_set_si(t3, bisect_sfdl_cons::fAtMidpt_FUNCTION_STATIC_RANDOM_INT[rctr++], 1);
    mpq_mul(t2, t3, t2);
    mpq_div_2exp(t2, t2, 1);
    mpq_add(f, f, t2);
  }


  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      mpq_add(t2, a[i], b[i]);
      mpq_add(t3, a[j], b[j]);
      mpq_mul(t2, t2, t3);
      mpq_set_si(t3, bisect_sfdl_cons::fAtMidpt_FUNCTION_STATIC_RANDOM_INT[rctr++], 1);
      mpq_mul(t2, t2, t3);
      mpq_div_2exp(t2, t2, 2);
      mpq_add(f, f, t2);
    }
  }

  mpq_clear(t2);
  mpq_clear(t3);
}

void bisect_sfdlProverExo::baseline(const mpq_t* input_q, int num_inputs,
                                    mpq_t* output_recomputed, int num_outputs) {

  int m = bisect_sfdl_cons::m;
  int L = bisect_sfdl_cons::L;

  mpq_t temp_q;
  alloc_init_scalar(temp_q);

  mpq_t* a = output_recomputed;
  mpq_t* b = output_recomputed + m;
  mpq_t& f = temp_q;

  for(int i = 0; i < m; i++) {
    mpq_set(a[i], input_q[i]);
    mpq_set(b[i], input_q[i+m]);
  }

  for(int i = 0; i < L; i++) {
    exogenous_fAtMidpt(f, a, b);
    if (mpq_sgn(f) > 0) {
      for(int j = 0; j < m; j++) {
        mpq_add(b[j], a[j], b[j]);
        mpq_div_2exp(b[j], b[j], 1);
      }
    } else {
      for(int j = 0; j < m; j++) {
        mpq_add(a[j], a[j], b[j]);
        mpq_div_2exp(a[j], a[j], 1);
      }
    }
  }
  clear_scalar(temp_q);
}

//Refer to apps_sfdl_gen/bisect_sfdl_cons.h for constants to use in this exogenous
//check.
bool bisect_sfdlProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
    int input_size, const mpz_t* output, const mpq_t* output_q, int output_size, mpz_t prime) {

  bool success = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  int m = bisect_sfdl_cons::m;

  mpq_t* buffer;
  alloc_init_vec(&buffer, m + m);

  baseline(input_q, input_size, buffer, output_size);

  mpq_t* a = buffer;
  mpq_t* b = buffer + m;

  for(int i = 0; i < m; i++) {
    success &= mpq_equal(a[i], output_q[i]);
    success &= mpq_equal(b[i], output_q[i+m]);
  }

  clear_vec(m+m, buffer);

#else
  gmp_printf("Exogeneous checking disabled\n");
#endif
  return success;
};
