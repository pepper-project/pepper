#include <apps_sfdl_hw/insertion_sort_1024_p_exo.h>
#include <apps_sfdl_gen/insertion_sort_1024_cons.h>
#include <gmpxx.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

insertion_sort_1024ProverExo::insertion_sort_1024ProverExo()
{
}

//Refer to apps_sfdl_gen/insertion_sort_1024_cons.h for constants to use in this exogenous
//check.
bool insertion_sort_1024ProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs){

#ifdef ENABLE_EXOGENOUS_CHECKING
  //Sort input_q
  std::vector<mpq_class> sorted_input(num_inputs);
  for(int j = 0; j < num_inputs; j++){
    sorted_input[j] = mpq_class(input_q[j]);
  }
  //std::sort(sorted_input.begin(), sorted_input.end());  
 
  mpz_t *output2;
  alloc_init_vec(&output2, num_inputs);

  for (int i=0; i<num_inputs; i++) {
    mpz_set(output2[i], input[i]);
  }
  
  mpz_t temp;
  alloc_init_scalar(temp);
  for (int j=0; j<num_inputs; j++)
  {
    int i_min = j;
    for (int i=j+1; i<num_inputs; i++)
    {
      if (output2[i] < output2[i_min])
      {
        i_min = i;
      }
    }

    if (i_min != j) {
      // swap
      mpz_set(temp, output2[j]);
      mpz_set(output2[j], output2[i_min]);
      mpz_set(output2[i_min], temp);
    }
  }


  bool listsEqual = true;
  for(int j = 0; j < num_outputs; j++){
    if (j > 0){
      //listsEqual &= mpq_equal(output_q[j], sorted_input[j].get_mpq_t());
      // gmp_printf("List sorted: %Qd\n", output_q[j]);
      if (mpz_cmp(output[j], output2[j]) == 0)
        continue;
      else
        listsEqual = false;
    }
  }

  return listsEqual;
#endif

  return true;
};

