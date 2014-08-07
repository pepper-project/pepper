#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/merge_sort_v_inp_gen.h>
#include <apps_sfdl_gen/merge_sort_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace merge_sort_cons;

/*
* Provides the ability for user-defined input creation
*/
class merge_sortVerifierInpGenHw : public InputCreator {
  public:
    merge_sortVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    merge_sortVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
