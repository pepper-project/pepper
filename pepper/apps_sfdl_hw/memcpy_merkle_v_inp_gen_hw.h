#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/memcpy_merkle_v_inp_gen.h>
#include <apps_sfdl_gen/memcpy_merkle_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace memcpy_merkle_cons;

/*
* Provides the ability for user-defined input creation
*/
class memcpy_merkleVerifierInpGenHw : public InputCreator {
  public:
    memcpy_merkleVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    memcpy_merkleVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
