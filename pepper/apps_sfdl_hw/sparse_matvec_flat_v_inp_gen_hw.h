#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/sparse_matvec_flat_v_inp_gen.h>
#include <apps_sfdl_gen/sparse_matvec_flat_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace sparse_matvec_flat_cons;

/*
* Provides the ability for user-defined input creation
*/
class sparse_matvec_flatVerifierInpGenHw : public InputCreator {
  public:
    sparse_matvec_flatVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    sparse_matvec_flatVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
