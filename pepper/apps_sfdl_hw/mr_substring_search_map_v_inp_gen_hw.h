#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mr_substring_search_map_v_inp_gen.h>
#include <apps_sfdl_gen/mr_substring_search_map_cons.h>
#include <stdint.h>
#include <storage/exo.h>

using namespace mr_substring_search_map_cons;

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/mr_substring_search.h>

#pragma pack(pop)

/*
* Provides the ability for user-defined input creation
*/
class mr_substring_search_mapVerifierInpGenHw : public InputCreator {
  public:
    mr_substring_search_mapVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    mr_substring_search_mapVerifierInpGen compiler_implementation;
    uint32_t num_inputs_created;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
