#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mr_cov_red_v_inp_gen.h>
#include <apps_sfdl_gen/mr_cov_red_cons.h>
#pragma pack(push)
#pragma pack(1)

using namespace mr_cov_red_cons;

/*
* Provides the ability for user-defined input creation
*/
class mr_cov_redVerifierInpGenHw : public InputCreator {
  public:
    mr_cov_redVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    mr_cov_redVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
