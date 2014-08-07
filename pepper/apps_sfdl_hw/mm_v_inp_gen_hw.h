#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mm_v_inp_gen.h>
#include <apps_sfdl_gen/mm_cons.h>

using namespace mm_cons;

/*
* Provides the ability for user-defined input creation
*/
class mmVerifierInpGenHw : public InputCreator {
  public:
    mmVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    mmVerifierInpGen compiler_implementation;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
