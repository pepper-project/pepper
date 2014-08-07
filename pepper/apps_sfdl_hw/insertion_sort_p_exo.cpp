#include <apps_sfdl_hw/insertion_sort_p_exo.h>
#include <apps_sfdl_gen/insertion_sort_cons.h>
#include <gmpxx.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

insertion_sortProverExo::insertion_sortProverExo() { }

void insertion_sortProverExo::baseline(const mpq_t* input_q, int num_inputs,
                                       mpq_t* output_recomputed, int num_outputs) {

  for (int i=0; i<num_inputs; i++) {
    mpz_set(mpq_numref(output_recomputed[i]), mpq_numref(input_q[i]));
  }

  mpz_t temp;
  alloc_init_scalar(temp);
  int hole_pos;
  for (int i=0; i<num_inputs; i++) {
    mpz_set(temp, mpq_numref(output_recomputed[i]));
    hole_pos = i;

    while (hole_pos > 0 && mpz_cmp(temp, mpq_numref(output_recomputed[hole_pos - 1])) < 0) {
      mpz_set(mpq_numref(output_recomputed[hole_pos]), mpq_numref(output_recomputed[hole_pos-1]));
      hole_pos = hole_pos - 1;
    }

    mpz_set(mpq_numref(output_recomputed[hole_pos]), temp);
  }
  clear_scalar(temp);
}

//Refer to apps_sfdl_gen/insertion_sort_cons.h for constants to use in this exogenous
//check.
bool insertion_sortProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
    int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool lists_equal = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_q_recomputed;
  alloc_init_vec(&output_q_recomputed, num_inputs);

  // call baseline to compute output
  baseline(input_q, num_inputs, output_q_recomputed, num_outputs);

  mpz_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);

  convert_to_z(num_outputs, output_recomputed, output_q_recomputed, prime);

  for(int j = 0; j < num_outputs; j++) {
    if (mpz_cmp(output[j], output_recomputed[j]) == 0)
      continue;
    else
      lists_equal = false;
  }
  clear_vec(num_outputs, output_recomputed);
  clear_vec(num_outputs, output_q_recomputed);
#else
  gmp_printf("Exogeneous checking disabled\n");
  lists_equal = true;
#endif
  return lists_equal;
};
