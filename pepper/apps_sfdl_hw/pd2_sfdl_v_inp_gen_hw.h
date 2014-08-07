#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/pd2_sfdl_v_inp_gen.h>

/*
* Provides the ability for user-defined input creation
*/
class pd2_sfdlVerifierInpGenHw : public InputCreator {
  public:
    pd2_sfdlVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    pd2_sfdlVerifierInpGen compiler_implementation;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
