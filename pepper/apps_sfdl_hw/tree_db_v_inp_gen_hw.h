#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/tree_db_v_inp_gen.h>
#include <apps_sfdl_gen/tree_db_cons.h>

#pragma pack(push)
#pragma pack(1)

using namespace tree_db_cons;

typedef uint32_t KEY_t;
typedef uint64_t FName_t;
typedef uint64_t LName_t;
typedef uint32_t Age_t;
typedef uint32_t Major_t;
typedef uint32_t State_t;
typedef uint32_t PhoneNum_t;
typedef uint32_t Class_t;
typedef uint32_t Credits_t;
typedef uint32_t Average_t;

typedef struct Student {
    KEY_t KEY;
    FName_t FName;
    LName_t LName;
    Age_t Age;
    Major_t Major;
    State_t State;
    PhoneNum_t PhoneNum;
    Class_t Class;
    Credits_t Credits;
    Average_t Average;
} Student_t;

typedef struct StudentResult {
    KEY_t KEY;
    FName_t FName;
    LName_t LName;
    Age_t Age;
    Major_t Major;
    State_t State;
    PhoneNum_t PhoneNum;
    Class_t Class;
    Credits_t Credits;
    Average_t Average;
} StudentResult_t;

struct In {hash_t hash;};
struct Out {uint32_t rows; StudentResult_t result[3];};

/*
* Provides the ability for user-defined input creation
*/
class tree_dbVerifierInpGenHw : public InputCreator {
  public:
    tree_dbVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);

  private:
    Venezia* v;
    tree_dbVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
