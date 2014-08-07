#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/email_history_v_inp_gen.h>
#include <apps_sfdl_gen/email_history_cons.h>
#pragma pack(push)
#pragma pack(1)

using namespace email_history_cons;

/*
* Provides the ability for user-defined input creation
*/
class email_historyVerifierInpGenHw : public InputCreator {
  public:
    email_historyVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    email_historyVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
