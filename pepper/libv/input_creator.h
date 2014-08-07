#ifndef CODE_PEPPER_LIBV_INPUT_CREATOR_H_
#define CODE_PEPPER_LIBV_INPUT_CREATOR_H_

#include<common/utility.h>

class InputCreator {
  public:
    InputCreator();
    virtual ~InputCreator() {}
    void set_generate_states(int gen_states) {generate_states = gen_states;}
    void set_block_store_name(const char* shared_block_store_name);
    virtual void create_input(mpq_t* input_q, int num_inputs) = 0;
  protected:
    int generate_states;
    string shared_bstore_file_name;
};

#endif  // CODE_PEPPER_LIBV_INPUT_CREATOR_H_
