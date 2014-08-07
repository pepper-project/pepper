#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/ramput_fast_micro_v_inp_gen.h>
#include <apps_sfdl_gen/ramput_fast_micro_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace ramput_fast_micro_cons;

/*
* Provides the ability for user-defined input creation
*/
class ramput_fast_microVerifierInpGenHw : public InputCreator {
  public:
    ramput_fast_microVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    ramput_fast_microVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
