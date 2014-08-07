#include <apps_sfdl_gen/fast_ram_test_v_inp_gen.h>
#include <apps_sfdl_hw/fast_ram_test_v_inp_gen_hw.h>
#include <apps_sfdl_gen/fast_ram_test_cons.h>

#include <math.h>
#include <time.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

fast_ram_testVerifierInpGenHw::fast_ram_testVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/fast_ram_test_cons.h for constants to use when generating input.
void fast_ram_testVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }

  //srand(time(NULL));

  //int i = rand() % 2;
  //int j = rand() % 4;
  //int k = rand() % 8;
  //mpq_set_ui(input_q[0], i, 1);
  //mpq_set_ui(input_q[1], j, 1);
  //mpq_set_ui(input_q[2], k, 1);

  //cout << "expected addr: " << 3 + i * 32 + j * 8 + k << endl;
}
