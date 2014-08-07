#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/mr_substring_search_red_v_inp_gen.h>
#include <apps_sfdl_gen/mr_substring_search_red_cons.h>

using namespace mr_substring_search_red_cons;

#include <apps_sfdl/mr_substring_search.h>

/*
* Provides the ability for user-defined input creation
*/
class mr_substring_search_redVerifierInpGenHw : public InputCreator {
  public:
    mr_substring_search_redVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    mr_substring_search_redVerifierInpGen compiler_implementation;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
