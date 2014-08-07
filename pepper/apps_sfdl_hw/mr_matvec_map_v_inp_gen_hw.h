#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mr_matvec_map_v_inp_gen.h>
#include <apps_sfdl_gen/mr_matvec_map_cons.h>
#pragma pack(push)
#pragma pack(1)

using namespace mr_matvec_map_cons;

typedef uint32_t num_t;

// actual input to the mapper
typedef struct _MapperIn {
  num_t matrix [NUM_ROWS_PER_MAPPER][NUM_VARS];
  num_t vector [NUM_VARS];
} MapperIn;

// actual output of the mapper
typedef struct _MapperChunkOut {
  num_t product [NUM_ROWS_PER_MAPPER];
} MapperChunkOut;

typedef struct _MapperOut {
  MapperChunkOut output[NUM_REDUCERS];
} MapperOut;

/*
* Provides the ability for user-defined input creation
*/
class mr_matvec_mapVerifierInpGenHw : public InputCreator {
  public:
    mr_matvec_mapVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    mr_matvec_mapVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
