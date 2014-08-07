#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mr_matvec_red_v_inp_gen.h>
#include <apps_sfdl_gen/mr_matvec_red_cons.h>
#pragma pack(push)
#pragma pack(1)

using namespace mr_matvec_red_cons;

typedef uint32_t num_t;

typedef struct _ReducerChunkIn {
  num_t product_part [NUM_ROWS_PER_MAPPER];
} ReducerChunkIn;

typedef struct _ReducerIn {
  ReducerChunkIn input[NUM_MAPPERS];
} ReducerIn;

typedef struct _ReducerOut {
  num_t product [NUM_ROWS_PER_MAPPER * NUM_MAPPERS];
} ReducerOut;

/*
* Provides the ability for user-defined input creation
*/
class mr_matvec_redVerifierInpGenHw : public InputCreator {
  public:
    mr_matvec_redVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    mr_matvec_redVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
