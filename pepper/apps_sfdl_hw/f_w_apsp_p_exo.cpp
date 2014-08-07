#include <apps_sfdl_hw/f_w_apsp_p_exo.h>
#include <apps_sfdl_gen/f_w_apsp_cons.h>
#include <gmpxx.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

f_w_apspProverExo::f_w_apspProverExo() {}

void f_w_apspProverExo::baseline(const mpq_t* input_q, int num_inputs,
                                 mpq_t* output_recomputed, int num_outputs) {
  int m = f_w_apsp_cons::m;
  int Infinity = f_w_apsp_cons::Infinity;

  mpq_class D[m][m];
  mpq_class D2[m][m];

  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      D[i][j] = mpq_class(input_q[i*m + j]);
    }
  }

  for(int k = 0; k < m; k++) {
    for(int i = 0; i < m; i++) {
      for(int j = 0; j < m; j++) {
        mpq_class sum = D[i][k] + D[k][j];
        if (sum < D[i][j] && D[i][k] != Infinity && D[k][j] != Infinity) {
          D2[i][j] = sum;
        } else {
          D2[i][j] = D[i][j];
        }
      }
    }

    for(int i = 0; i < m; i++) {
      for(int j= 0; j < m; j++) {
        D[i][j] = D2[i][j];
      }
    }
  }

  for (int i=0; i<m; i++) {
    for (int j=0; j<m; j++) {
      mpq_set(output_recomputed[i*m+j], D[i][j].get_mpq_t());
    }
  }
}

//Refer to apps_sfdl_gen/f_w_apsp_cons.h for constants to use in this exogenous
//check.
bool f_w_apspProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
                                        int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool success = true;
#ifdef ENABLE_EXOGENOUS_CHECKING

  int m = f_w_apsp_cons::m;
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, m*m);

  mpz_t *output_z;
  alloc_init_vec(&output_z, m*m);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  convert_to_z(num_outputs, output_z, output_recomputed, prime);

  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      success &= (mpz_class(output_z[i*m+j]) == mpz_class(output[i*m + j]));
    }
  }

  clear_vec(m*m, output_recomputed);
  clear_vec(m*m, output_z);

#else
  cout<<"Exogeneous checking disabled"<<endl;
#endif
  return success;
};

