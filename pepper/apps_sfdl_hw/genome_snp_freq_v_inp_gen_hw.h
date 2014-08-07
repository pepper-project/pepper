#ifndef APPS_SFDL_HW_V_INP_GEN_HW_H_
#define APPS_SFDL_HW_V_INP_GEN_HW_H_

#include <libv/libv.h>
#include <common/utility.h>
#include <apps_sfdl_gen/genome_snp_freq_v_inp_gen.h>
#include <apps_sfdl_gen/genome_snp_freq_cons.h>
#pragma pack(push)
#pragma pack(1)

using namespace genome_snp_freq_cons;

/*
* Provides the ability for user-defined input creation
*/
class genome_snp_freqVerifierInpGenHw : public InputCreator {
  public:
    genome_snp_freqVerifierInpGenHw(Venezia* v);

    void create_input(mpq_t* input_q, int num_inputs);
  private:
    Venezia* v;
    genome_snp_freqVerifierInpGen compiler_implementation;

};
#pragma pack(pop)
#endif  // APPS_SFDL_HW_V_INP_GEN_HW_H_
