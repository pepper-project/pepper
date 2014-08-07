#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/shared_tree_v_inp_gen.h>
#include <apps_sfdl_gen/shared_tree_cons.h>
#include <include/db.h>

#pragma pack(push)
#pragma pack(1)

using namespace shared_tree_cons;

struct In {hash_t root;};
struct Out {uint32_t rows; uint32_t values;};

/*
* Provides the ability for user-defined input creation
*/
class shared_treeVerifierInpGenHw : public InputCreator {
  public:
    shared_treeVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    shared_treeVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
