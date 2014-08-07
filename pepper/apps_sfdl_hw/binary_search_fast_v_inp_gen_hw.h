#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/binary_search_fast_v_inp_gen.h>
#include <apps_sfdl_gen/binary_search_fast_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace binary_search_fast_cons;

/*
* Provides the ability for user-defined input creation
*/
class binary_search_fastVerifierInpGenHw : public InputCreator {
  public:
    binary_search_fastVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    binary_search_fastVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
