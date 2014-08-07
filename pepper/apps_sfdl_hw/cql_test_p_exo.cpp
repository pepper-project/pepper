#include <apps_sfdl_hw/cql_test_p_exo.h>
#include <apps_sfdl_gen/cql_test_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

cql_testProverExo::cql_testProverExo() { }

void cql_testProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  int SIZE = cql_test_cons::SIZE;
  int logSIZE = cql_test_cons::logSIZE;

  //There are SIZE + 1 inputs, put at the end of the inputs
  //We ignore the merkle tree root hash, which comes at the start.
  int input_start = num_inputs - SIZE - 1;

  int ram[96];
  uint32_t i, min, max, avg;
  uint32_t lowerBound, upperBound, index;
  uint32_t got;

  ram[0] = 11;
  ram[1] = 10;
  ram[2] = 0;
  ram[3] = 10;
  ram[4] = 5;
  ram[5] = 3;
  ram[6] = 2;
  ram[7] = 2;
  ram[8] = 6;
  ram[9] = 9;
  ram[10] = 13;
  ram[11] = 5;
  ram[12] = 3;
  ram[13] = 7;
  ram[14] = 2;
  ram[15] = 14;
  ram[16] = 3;
  ram[17] = 8;
  ram[18] = 0;
  ram[19] = 3;
  ram[20] = 4;
  ram[21] = 3;
  ram[22] = 14;
  ram[23] = 13;
  ram[24] = 7;
  ram[25] = 12;
  ram[26] = 14;
  ram[27] = 11;
  ram[28] = 5;
  ram[29] = 8;
  ram[30] = 15;
  ram[31] = 5;
  ram[32] = 0;
  ram[33] = 2;
  ram[34] = 2;
  ram[35] = 2;
  ram[36] = 3;
  ram[37] = 3;
  ram[38] = 5;
  ram[39] = 5;
  ram[40] = 6;
  ram[41] = 7;
  ram[42] = 9;
  ram[43] = 10;
  ram[44] = 10;
  ram[45] = 11;
  ram[46] = 13;
  ram[47] = 14;
  ram[48] = 2;
  ram[49] = 6;
  ram[50] = 7;
  ram[51] = 14;
  ram[52] = 5;
  ram[53] = 12;
  ram[54] = 4;
  ram[55] = 11;
  ram[56] = 8;
  ram[57] = 13;
  ram[58] = 9;
  ram[59] = 1;
  ram[60] = 3;
  ram[61] = 0;
  ram[62] = 10;
  ram[63] = 15;
  ram[64] = 0;
  ram[65] = 3;
  ram[66] = 3;
  ram[67] = 3;
  ram[68] = 4;
  ram[69] = 5;
  ram[70] = 5;
  ram[71] = 7;
  ram[72] = 8;
  ram[73] = 8;
  ram[74] = 11;
  ram[75] = 12;
  ram[76] = 13;
  ram[77] = 14;
  ram[78] = 14;
  ram[79] = 15;
  ram[80] = 2;
  ram[81] = 0;
  ram[82] = 3;
  ram[83] = 5;
  ram[84] = 4;
  ram[85] = 12;
  ram[86] = 15;
  ram[87] = 8;
  ram[88] = 1;
  ram[89] = 13;
  ram[90] = 11;
  ram[91] = 9;
  ram[92] = 7;
  ram[93] = 6;
  ram[94] = 10;
  ram[95] = 14;
  lowerBound = 64;
  min=64;
  max=79;
  for(i = 0; i < logSIZE; i++) {
    avg = (min + max) >> 1;
    got = ram[avg];
    if (got < 8) {
      min = avg+1;
    } else {
      max = avg-1;
    }
  }
  upperBound = max;
  for (i = 0; i < 3; i++) {
    if (i+lowerBound < upperBound) {
      index = ram[i+lowerBound+16];
      mpq_set_si(output_recomputed[i*2+0+1], ram[index+0], 1);
      mpq_set_si(output_recomputed[i*2+1+1], ram[index+16], 1);
    } else {
      mpq_set_si(output_recomputed[i*2+0+1], 0, 1);
      mpq_set_si(output_recomputed[i*2+1+1], 0, 1);
    }
  }

  //Return value
  mpq_set_si(output_recomputed[0], 0, 1);
}

//Refer to apps_sfdl_gen/cql_test_cons.h for constants to use in this exogenous
//check.
bool cql_testProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  //Fill me out!
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      passed_test = false;
      //gmp_printf("Failure: %Qd %Qd %d\n", output_recomputed[i], output_q[i], i);
      break;
    }
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

