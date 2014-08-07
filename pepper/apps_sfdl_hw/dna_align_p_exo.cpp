#include <apps_sfdl_hw/dna_align_p_exo.h>
#include <apps_sfdl_gen/dna_align_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

dna_alignProverExo::dna_alignProverExo() {
}

static int getScore(int* LL, int i, int j, int m, int n) {
  if (i < 0 || j < 0 || i >= m || j >= n) {
    return 0;
  }
  return LL[i*n + j];
}

void dna_alignProverExo::baseline(const mpq_t* input_q, int num_inputs,
                                  mpq_t* output_recomputed, int num_outputs) {
  int m = dna_align_cons::A_LEN;
  int n = dna_align_cons::B_LEN;

  const mpq_t* A = input_q;
  const mpq_t* B = input_q + m;

  int LL [m*n];
  int choices [m*n];
  for(int i = m-1; i >= 0; i--) {
    for(int j = n-1; j>=0; j--) {
      if (mpq_equal(A[i],B[j])) {
        LL[i*n + j] = getScore(LL, i+1, j+1, m, n) + 1;
        choices[i*n + j] = 0;
      } else {
        int down = getScore(LL, i+1, j, m, n);
        int right = getScore(LL, i, j+1, m, n);
        if (down == right + 1) {
          LL[i*n + j] = down;
          choices[i*n + j] = 1;
        } else { //same, or right == down+1
          LL[i*n + j] = right;
          choices[i*n + j] = 2;
        }
      }
    }
  }
 
  bool verbose = false; 
  if (verbose){
    int i = 0;
    int j = 0;
    for(i = 0; i < m; i++){
      gmp_printf("%c", (char)mpz_get_ui(mpq_numref(A[i])));
    }
    gmp_printf("\n");
    for(i = 0; i < n; i++){
      gmp_printf("%c", (char)mpz_get_ui(mpq_numref(B[i])));
    }
    gmp_printf("\n");  
  }
  
  int i = 0;
  int j = 0;
  int checkIndex = 1;
  bool match = true;
  while(i < m && j < n) {
    switch(choices[i*n + j]) {
    case 0:
      if (!mpq_equal(A[i], output_recomputed[checkIndex])) {
        match = false;
      }
      checkIndex++;
      if (verbose) gmp_printf("%c", (char)mpz_get_ui(mpq_numref(A[i])));
      i++;
      j++;
      break;
    case 1:
      i++;
      break;
    case 2:
      j++;
      break;
    }
  }
  if (verbose) gmp_printf("\n\n");
  mpq_set_ui(output_recomputed[0], (int) match, 1);
}

//Refer to apps_sfdl_gen/dna_align_cons.h for constants to use in this exogenous
//check.
bool dna_alignProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
    int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  for (int i=0; i<num_outputs; i++)
    mpq_set(output_recomputed[i], output_q[i]);

  baseline(input_q, num_inputs, output_recomputed, num_outputs);
  bool match = mpz_get_ui(mpq_numref(output_recomputed[0]));

  clear_vec(num_outputs, output_recomputed);
  return match;
#else
  cout<<"Exogeneous checking disabled"<<endl;
  return true;
#endif
};

