#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/base_2_log_v_inp_gen.h>
#include <apps_sfdl_gen/base_2_log_cons.h>
#pragma pack(push)
#pragma pack(1)

using namespace base_2_log_cons;

/*
* Provides the ability for user-defined input creation
*/
class base_2_logVerifierInpGenHw : public InputCreator {
  public:
    base_2_logVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    base_2_logVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
