#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/shared_store_v_inp_gen.h>
#include <apps_sfdl_gen/shared_store_cons.h>

using namespace shared_store_cons;
struct In {uint32_t test;};
/*struct Out {uint8_t data[SIZE]; };*/
struct Out {uint32_t data;uint32_t data1;};
/*struct Out {hash_t hash; struct In data;};*/

/*
* Provides the ability for user-defined input creation
*/
class shared_storeVerifierInpGenHw : public InputCreator {
  public:
    shared_storeVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    shared_storeVerifierInpGen compiler_implementation;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
