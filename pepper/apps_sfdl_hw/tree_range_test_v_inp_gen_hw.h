#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/tree_range_test_v_inp_gen.h>
#include <apps_sfdl_gen/tree_range_test_cons.h>
#pragma pack(push)
#pragma pack(1)

using namespace tree_range_test_cons;

/*
* Provides the ability for user-defined input creation
*/
class tree_range_testVerifierInpGenHw : public InputCreator {
  public:
    tree_range_testVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    tree_range_testVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
