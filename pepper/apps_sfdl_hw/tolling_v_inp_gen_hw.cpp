#include <apps_sfdl_gen/tolling_v_inp_gen.h>
#include <apps_sfdl_hw/tolling_v_inp_gen_hw.h>
#include <apps_sfdl_gen/tolling_cons.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

tollingVerifierInpGenHw::tollingVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

#include <apps_sfdl/tolling.h>

int set_tuple(mpq_t* input_q, int inp, tuple_t* tuple){
  mpq_set_si(input_q[inp++], tuple->time, 1);
  mpq_set_ui(input_q[inp++], tuple->toll_booth_id, 1);
  mpq_set_ui(input_q[inp++], tuple->toll, 1);
  //Return the new offset in the variables list
  return inp;
}

//Refer to apps_sfdl_gen/tolling_cons.h for constants to use when generating input.
void tollingVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  char db_file_path[BUFLEN];
  snprintf(db_file_path, BUFLEN - 1, "%s/block_stores/prover_1_%s", FOLDER_STATE,
  shared_bstore_file_name.c_str());
  ConfigurableBlockStore bs(db_file_path);
  int i;

  hash_t hash;
  commitment_t commitment;
  struct pathdb db;
  for(i = 0; i < tolling_cons::MAX_TUPLES; i++){
    db.path[i].time = i;
    db.path[i].toll_booth_id = (i % 5);
    db.path[i].toll = 50; //50 cents per toll always
  }

  //CK bit string acting like a salt
  commitmentCK_t CK = {{
    //Randomly generated.
   0x8a, 0xf7, 0x24, 0xa1, 0x58, 
   0xc9, 0x8b, 0x89, 0x29, 0x85, 
   0xce, 0xa1, 0xae, 0xc3, 0x42, 
   0x6e, 0xbb, 0x86, 0x56, 0x37
  }};
  setcommitmentCK(&CK);
  hashput2(&bs, &hash, &db);
  commitmentput2(&bs, &commitment, &hash);

  //Position in input variables list
  int inp = 0;


  for(i = 0; i < NUM_CK_BITS/8; i++){
    mpq_set_ui(input_q[inp++], CK.bit[i], 1);
  }

  //for(i = 0; i < NUM_HASH_CHUNKS; i++){
  for(i = 0; i < NUM_COMMITMENT_CHUNKS; i++){
    mpq_set_ui(input_q[inp++], commitment.bit[i], 1);
  }
  //Fill in the rest of the input

  for(i = 0; i < tolling_cons::MAX_SPOTCHECKS; i++){
    //Choose the ith tuple in the db
    inp = set_tuple(input_q, inp, &(db.path[i]));
  }

  //Set the time threshold
  mpq_set_si(input_q[inp++], 2, 1);

  if (inp > num_inputs){
    std::cerr << "ERROR: Wrong num_inputs" << std::endl;
  }

  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }
}

#pragma pack(pop)
