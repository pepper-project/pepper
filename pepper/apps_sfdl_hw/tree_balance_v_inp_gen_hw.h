#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/tree_balance_v_inp_gen.h>
#include <apps_sfdl_gen/tree_balance_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace tree_balance_cons;

/*
* Provides the ability for user-defined input creation
*/
class tree_balanceVerifierInpGenHw : public InputCreator {
  public:
    tree_balanceVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    tree_balanceVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
