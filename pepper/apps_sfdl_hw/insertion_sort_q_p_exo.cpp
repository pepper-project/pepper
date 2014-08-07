#include <apps_sfdl_hw/insertion_sort_q_p_exo.h>
#include <apps_sfdl_gen/insertion_sort_q_cons.h>
#include <gmpxx.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

insertion_sort_qProverExo::insertion_sort_qProverExo() { }

void insertion_sort_qProverExo::baseline(const mpq_t* input_q, int num_inputs,
    mpq_t* output_recomputed, int num_outputs) {

  for (int i=0; i<num_inputs; i++) {
    mpq_set(output_recomputed[i], input_q[i]);
  }

  std::vector<mpq_class> sorted_input(num_inputs);
  for(int j = 0; j < num_inputs; j++) {
    sorted_input[j] = mpq_class(output_recomputed[j]);
  }
  std::sort(sorted_input.begin(), sorted_input.end());

  for(int j = 0; j < num_inputs; j++)
    mpq_set(output_recomputed[j], sorted_input[j].get_mpq_t());

  /*mpq_t temp;
  alloc_init_scalar(temp);
  int hole_pos;
  for (int i=0; i<num_inputs; i++) {
    mpq_set(temp, output_recomputed[i]);
    hole_pos = i;

    while (hole_pos > 0) {
      int ret = mpq_cmp(temp, output_recomputed[hole_pos - 1]);
      if (ret < 0)
        hole_pos = hole_pos - 1;
      else
        break;
    }

    for (int j=i; j>hole_pos; j--)
      mpq_set(output_recomputed[j], output_recomputed[j-1]);

    mpq_set(output_recomputed[hole_pos], temp);
  }

  clear_scalar(temp);
  */
}

//Refer to apps_sfdl_gen/insertion_sort_q_cons.h for constants to use in this exogenous
//check.
bool insertion_sort_qProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
