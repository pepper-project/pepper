#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/tolling_v_inp_gen.h>
#include <apps_sfdl_gen/tolling_cons.h>
#pragma pack(push)
#pragma pack(1)

/*
* Provides the ability for user-defined input creation
*/
class tollingVerifierInpGenHw : public InputCreator {
  public:
    tollingVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    tollingVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
