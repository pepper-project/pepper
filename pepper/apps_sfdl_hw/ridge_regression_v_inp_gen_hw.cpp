#include <apps_sfdl_gen/ridge_regression_v_inp_gen.h>
#include <apps_sfdl_hw/ridge_regression_v_inp_gen_hw.h>
#include <apps_sfdl_gen/ridge_regression_cons.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

ridge_regressionVerifierInpGenHw::ridge_regressionVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

#include <apps_sfdl/ridge_regression.h>

//Refer to apps_sfdl_gen/ridge_regression_cons.h for constants to use when generating input.
void ridge_regressionVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  #if IS_REDUCER == 0
  //Default implementation is provided by compiler
  compiler_implementation.create_input(input_q, num_inputs);
  #endif

  char db_file_path[BUFLEN];
  snprintf(db_file_path, BUFLEN - 1, "%s/block_stores/prover_1_%s", FOLDER_STATE,
      shared_bstore_file_name.c_str());
  ConfigurableBlockStore bs(db_file_path);

  int d = NUM_FEATURES;

  commitment_t commitment;

  ifstream fp;
  open_file_read(fp, "ridge_regression_data.txt", "static_state");

  std::string ignore;
  int total_records, num_features;
  fp >> ignore >> total_records >> ignore >> num_features;

  if (num_features != NUM_FEATURES){
    cout << "ERROR: NUM_FEATURES should be " << num_features <<
      " to match data in datafile" << endl;
    exit(1);
  }
  if (total_records < NUM_RECORDS){
    cout << "ERROR: NUM_RECORDS should be at most " << total_records <<
      " to match data in datafile" << endl;
    exit(1);
  }
  int total_read = 0; //Number of records read from file thus far

  fp >> ignore;
  int norm_factors [d];
  for(int i = 0; i < d; i++){
    fp >> norm_factors[i];
  }

  patientDB db;
  //Add some number of records to this client
  for(int rec = 0; rec < NUM_RECORDS && rec < total_records; rec++){
    for(int j = 0; j < d; j++){
      fix_t feature;
      fp >> feature;
      feature = ((int64_t)feature)*FIX_SCALE / norm_factors[j]; //Convert integer to fix_t.
      db.records[rec].features[j] = feature;
    }
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
  hash_t hash;
  hashput2(&bs, &hash, &db);
  commitmentput2(&bs, &commitment, &hash);

  fp.close();

  fix_t k = FIX_SCALE / 16;


  int inp = 0;
  for(int j = 0; j < NUM_CK_BITS/8; j++){
    mpz_set_ui(mpq_numref(input_q[inp++]), CK.bit[j]);
  }

  //cout << "Wrote input: ";
  for(int j = 0; j < NUM_COMMITMENT_CHUNKS; j++){
    mpz_set_ui(mpq_numref(input_q[inp++]), commitment.bit[j]);
    //cout << (int)commitment.bit[j] << " ";
  }
  //cout << endl;
  mpz_set_si(mpq_numref(input_q[inp++]), k);

  // states that should be persisted and may not be generated everytime should be created here.
  if (generate_states) {
  }
}

#pragma pack(pop)
