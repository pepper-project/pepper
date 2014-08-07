#include <apps_sfdl_hw/permanent_p_exo.h>
#include <apps_sfdl_gen/permanent_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

permanentProverExo::permanentProverExo() {
}

static bool GRAY_CODE(int i, int j, int m) {
  return ((i ^ (i >> 1)) & (1 << (m-1-j))) != 0;
}

//Refer to apps_sfdl_gen/permanent_cons.h for constants to use in this exogenous
//check.
bool permanentProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
    int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs) {

#ifdef ENABLE_EXOGENOUS_CHECKING

  int m = permanent_cons::m;
  int signM = permanent_cons::signM;
  int twoToM = permanent_cons::twoToM;

  mpz_t permanent, term, sign;
  mpz_t* rowsums;
  alloc_init_scalar(permanent);
  alloc_init_scalar(term);
  alloc_init_scalar(sign);
  alloc_init_vec(&rowsums, m);

  mpz_set_ui(permanent,0);
  for(int i = 0; i < m; i++) {
    mpz_set_ui(rowsums[i], 0);
  }

  //NOTE the SFDL implementation runs in O(2^n * n) time, while
  //this runs in O(2^n * n^2) time, because the compiler in SFDL
  //has more information than GCC does (it can simulate the following loop)
  for(int i = 1; i < twoToM; i++) {
    //Update row sums
    for(int j = 0; j < m; j++) {
      if (GRAY_CODE(i,j,m) && !GRAY_CODE(i-1, j,m)) {
        for(int k = 0; k < m; k++) {
          mpz_add(rowsums[k], rowsums[k], mpq_numref(input_q[k*m + j]));
        }
      }
      if (!GRAY_CODE(i,j,m) && GRAY_CODE(i-1, j,m)) {
        for(int k = 0; k < m; k++) {
          mpz_sub(rowsums[k], rowsums[k], mpq_numref(input_q[k*m + j]));
        }
      }
    }

    mpz_set_ui(sign,1);
    for(int j = 0; j < m; j++) {
      if (GRAY_CODE(i,j,m)) {
        mpz_neg(sign, sign);
      }
    }

    mpz_set_ui(term, 1);
    for(int j = 0; j < m; j++) {
      mpz_mul(term, term, rowsums[j]);
    }

    mpz_mul(term, term, sign);

    mpz_add(permanent, permanent, term);
  }

  /*
     for(int i = 0; i < m; i++){
     for(int j = 0; j < m; j++){
     gmp_printf("%Qd ", input_q[i*m + j]);
     }
     gmp_printf("\n");
     }
     gmp_printf("%Qd\n\n", output_q[0]);
     */

  return mpz_cmp(permanent, mpq_numref(output_q[0])) == 0;

#else
  return true;
#endif
};

