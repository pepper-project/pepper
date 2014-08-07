#include <apps_sfdl_hw/fannkuch_p_exo.h>
#include <apps_sfdl_gen/fannkuch_cons.h>
#include <gmpxx.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

fannkuchProverExo::fannkuchProverExo() {}

void fannkuchProverExo::baseline(const mpq_t* input_q, int num_inputs,
                                 mpq_t* output_recomputed, int num_outputs) {

  int L = fannkuch_cons::L;

  std::vector<mpq_class> Xa(num_inputs);
  for(int j = 0; j < num_inputs; j++) {
    Xa[j] = mpq_class(input_q[j]);
  }

  int max_flips = 0;
  for(int i = 0; i < L; i++) {
    std::vector<mpq_class> stack = Xa;

    int flips = 0;
    while(stack[0] != 1) {
      std::reverse(stack.begin(), stack.begin() + (int)(stack[0].get_d()));
      flips++;
    }
    if (flips > max_flips) {
      max_flips = flips;
    }

    //Advance Xa to the next permutation
    std::next_permutation(Xa.begin(), Xa.end());
  }
  mpz_set_ui(mpq_numref(output_recomputed[0]), max_flips);
}

//Refer to apps_sfdl_gen/fannkuch_cons.h for constants to use in this exogenous
//check.
bool fannkuchProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
                                        int input_size, const mpz_t* output, const mpq_t* output_q, int output_size, mpz_t prime) {
  bool success = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, 1);

  baseline(input_q, input_size, output_recomputed, output_size);
  success = mpq_class(output_recomputed[0]) == mpq_class(output_q[0]);
  clear_vec(1, output_recomputed);
#else
  gmp_printf("Exogeneous checking disabled");
#endif
  return success;
};

