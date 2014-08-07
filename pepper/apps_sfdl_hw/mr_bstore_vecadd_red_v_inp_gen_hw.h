#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mr_bstore_vecadd_red_v_inp_gen.h>
#include <apps_sfdl_gen/mr_bstore_vecadd_red_cons.h>

using namespace mr_bstore_vecadd_red_cons;

typedef struct _ReducerChunkIn {
  uint32_t input;
} ReducerChunkIn;

typedef struct _ReducerIn {
  ReducerChunkIn input[NUM_MAPPERS];
} ReducerIn;

typedef struct _ReducerOut {
  uint32_t output;
} ReducerOut;


/*
* Provides the ability for user-defined input creation
*/
class mr_bstore_vecadd_redVerifierInpGenHw : public InputCreator {
  public:
    mr_bstore_vecadd_redVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    mr_bstore_vecadd_redVerifierInpGen compiler_implementation;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
