#include <apps_sfdl_gen/face_detect_hamming_v_inp_gen.h>
#include <apps_sfdl_hw/face_detect_hamming_v_inp_gen_hw.h>
#include <apps_sfdl_gen/face_detect_hamming_cons.h>

#include <storage/configurable_block_store.h>
#include <storage/exo.h>
#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/face_detect_hamming.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

face_detect_hammingVerifierInpGenHw::face_detect_hammingVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/face_detect_hamming_cons.h for constants to use when generating input.
void face_detect_hammingVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  char db_file_path[BUFLEN];
  snprintf(db_file_path, BUFLEN - 1, "%s/block_stores/prover_1_%s", FOLDER_STATE, shared_bstore_file_name.c_str());
  ConfigurableBlockStore bs(db_file_path);

  //hash_t hash;
  commitment_t commitment;

    struct FaceDB db;

    //Read in FaceDB from static_state/imageList.txt
    ifstream fp;
    open_file_read(fp, "imageList.txt", "static_state");

    for(int i = 0; i < NUM_FACES; i++){
      std::string name;
      std::string bitstring;
      num_t threshold;
      fp >> name >> bitstring >> threshold;
      for(int word = 0; word < LENGTH_FACE/32; word ++){
        db.faces[i].data[word] = 0;
        for(int32_t c = 0; c < 32; c++){
          uint32_t index = word*32 + c;
          if (index < bitstring.length()){
            if (bitstring[index] == '1'){
              db.faces[i].data[word] |= 1 << c;
            }
          }
        }
        //cout << db.faces[i].data[word] << endl;
      }
      db.faces[i].threshold = threshold;
    }
    fp.close();

/*
    //Generate the database completely at random
    uint8_t* db_typepun = (uint8_t*)&db;
    for(uint32_t i = 0; i < sizeof(struct FaceDB); i++){
      db_typepun[i] = (uint8_t) rand();
    }
*/

  //CK bit string acting like a salt
  commitmentCK_t CK = {{
    //Randomly generated.
   0x8a, 0xf7, 0x24, 0xa1, 0x58, 
   0xc9, 0x8b, 0x89, 0x29, 0x85, 
   0xce, 0xa1, 0xae, 0xc3, 0x42, 
   0x6e, 0xbb, 0x86, 0x56, 0x37
  }};
  setcommitmentCK(&CK);

  hash_t hash;
  hashput2(&bs, &hash, &db);
  commitmentput2(&bs, &commitment, &hash);

  int i;
  int inp = 0;
  for(i = 0; i < NUM_CK_BITS/8; i++){
    mpq_set_ui(input_q[inp++], CK.bit[i], 1);
  }


  //NOTE - not NUM_HASH_CHUNKS
  for(i = 0; i < NUM_COMMITMENT_CHUNKS; i++){
    mpz_set_ui(mpq_numref(input_q[inp++]), commitment.bit[i]);
  }

  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }
}
#pragma pack(pop)
