#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/db_test_v_inp_gen.h>
#include <storage/merkle_ram.h>
#include <storage/hasher.h>

/*
* Provides the ability for user-defined input creation
*/
class db_testVerifierInpGenHw : public InputCreator {
  public:
    db_testVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    db_testVerifierInpGen compiler_implementation;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
