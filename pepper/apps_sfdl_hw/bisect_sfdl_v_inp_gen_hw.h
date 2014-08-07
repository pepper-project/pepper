#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/bisect_sfdl_v_inp_gen.h>

/*
* Provides the ability for user-defined input creation
*/
class bisect_sfdlVerifierInpGenHw : public InputCreator {
  public:
    bisect_sfdlVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    bisect_sfdlVerifierInpGen compiler_implementation;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
