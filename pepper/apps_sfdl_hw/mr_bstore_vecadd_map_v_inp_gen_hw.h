#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mr_bstore_vecadd_map_v_inp_gen.h>
#include <apps_sfdl_gen/mr_bstore_vecadd_map_cons.h>
#include <storage/exo.h>

using namespace mr_bstore_vecadd_map_cons;

// actual input to the mapper
typedef struct _MapperIn {
  uint32_t input[SIZE_INPUT];
} MapperIn;

// actual output of the mapper
typedef struct _MapperChunkOut {
  uint32_t output[SIZE_OUTPUT];
} MapperChunkOut;

typedef struct _MapperOut {
  MapperChunkOut output[NUM_REDUCERS];
} MapperOut;

/*
* Provides the ability for user-defined input creation
*/
class mr_bstore_vecadd_mapVerifierInpGenHw : public InputCreator {
  public:
    mr_bstore_vecadd_mapVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    mr_bstore_vecadd_mapVerifierInpGen compiler_implementation;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
