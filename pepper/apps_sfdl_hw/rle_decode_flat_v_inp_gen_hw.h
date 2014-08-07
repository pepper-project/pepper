#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/rle_decode_flat_v_inp_gen.h>
#include <apps_sfdl_gen/rle_decode_flat_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace rle_decode_flat_cons;

/*
* Provides the ability for user-defined input creation
*/
class rle_decode_flatVerifierInpGenHw : public InputCreator {
  public:
    rle_decode_flatVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    rle_decode_flatVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
