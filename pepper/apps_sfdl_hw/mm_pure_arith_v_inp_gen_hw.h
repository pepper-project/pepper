#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mm_pure_arith_v_inp_gen.h>
#include <apps_sfdl_gen/mm_pure_arith_cons.h>
#pragma pack(push)
#pragma pack(1)

//using namespace mm_pure_arith_cons;

/*
* Provides the ability for user-defined input creation
*/
class mm_pure_arithVerifierInpGenHw : public InputCreator {
  public:
    mm_pure_arithVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    mm_pure_arithVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
