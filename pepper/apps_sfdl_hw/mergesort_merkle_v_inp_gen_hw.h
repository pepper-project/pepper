#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mergesort_merkle_v_inp_gen.h>
#include <apps_sfdl_gen/mergesort_merkle_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace mergesort_merkle_cons;

/*
* Provides the ability for user-defined input creation
*/
class mergesort_merkleVerifierInpGenHw : public InputCreator {
  public:
    mergesort_merkleVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    mergesort_merkleVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
