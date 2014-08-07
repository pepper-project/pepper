#include <apps_sfdl_hw/pd2eval_p_exo.h>
#include <apps_sfdl_gen/pd2eval_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

pd2evalProverExo::pd2evalProverExo()
{
}

//Refer to apps_sfdl_gen/pd2eval_cons.h for constants to use in this exogenous
//check.
bool pd2evalProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs){

#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t t2;
  mpq_t f;
  mpq_init(f);
  mpq_init(t2);

  const mpq_t* a = input_q;
  int rctr = 0;
  int m = pd2eval_cons::m;
  mpq_set_si(f, 0, 1);
  for(int i = 0; i < m; i++){
    mpq_set_si(t2, pd2eval_cons::poly_FUNCTION_STATIC_RANDOM_INT[rctr++], 1);
    mpq_mul(t2, t2, a[i]);
    mpq_add(f, f, t2);
  }
  for(int i = 0; i < m; i++){
    for(int j = 0; j < m; j++){
      mpq_set_si(t2, pd2eval_cons::poly_FUNCTION_STATIC_RANDOM_INT[rctr++], 1);
      mpq_mul(t2, t2, a[i]);
      mpq_mul(t2, t2, a[j]);
      mpq_add(f, f, t2);
    }
  }

  bool success = mpq_cmp(f, output_q[0]) == 0;
  
  mpq_clear(f);
  mpq_clear(t2);

  return success;
#endif

  return true;
};

