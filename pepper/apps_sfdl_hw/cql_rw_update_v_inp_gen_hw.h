#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/cql_rw_update_v_inp_gen.h>
#include <apps_sfdl_gen/cql_rw_update_cons.h>
#pragma pack(push)
#pragma pack(1)

using namespace cql_rw_update_cons;

/*
* Provides the ability for user-defined input creation
*/
class cql_rw_updateVerifierInpGenHw : public InputCreator {
  public:
    cql_rw_updateVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    cql_rw_updateVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
