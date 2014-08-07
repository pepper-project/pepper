#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/cql_ro_threshold_v_inp_gen.h>
#include <apps_sfdl_gen/cql_ro_threshold_cons.h>
#include <include/avl_tree.h>

#pragma pack(push)
#pragma pack(1)

/*
* Provides the ability for user-defined input creation
*/
class cql_ro_thresholdVerifierInpGenHw : public InputCreator {
  public:
    cql_ro_thresholdVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    cql_ro_thresholdVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
