#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/divide_int_int_v_inp_gen.h>
#include <apps_sfdl_gen/divide_int_int_cons.h>
#pragma pack(push)
#pragma pack(1)

using namespace divide_int_int_cons;

/*
* Provides the ability for user-defined input creation
*/
class divide_int_intVerifierInpGenHw : public InputCreator {
  public:
    divide_int_intVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    divide_int_intVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
