#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/db_insertion_v_inp_gen.h>
#include <apps_sfdl_gen/db_insertion_cons.h>
#include <storage/hasher.h>

/*
* Provides the ability for user-defined input creation
*/
class db_insertionVerifierInpGenHw : public InputCreator {
  public:
    db_insertionVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    db_insertionVerifierInpGen compiler_implementation;

};
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
