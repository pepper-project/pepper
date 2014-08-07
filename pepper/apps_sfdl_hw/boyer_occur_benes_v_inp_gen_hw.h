#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/boyer_occur_benes_v_inp_gen.h>
#include <apps_sfdl_gen/boyer_occur_benes_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace boyer_occur_benes_cons;

/*
* Provides the ability for user-defined input creation
*/
class boyer_occur_benesVerifierInpGenHw : public InputCreator {
  public:
    boyer_occur_benesVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    boyer_occur_benesVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
