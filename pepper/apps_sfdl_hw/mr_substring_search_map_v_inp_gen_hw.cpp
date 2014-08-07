#include <apps_sfdl_gen/mr_substring_search_map_v_inp_gen.h>
#include <apps_sfdl_hw/mr_substring_search_map_v_inp_gen_hw.h>
#include <apps_sfdl_gen/mr_substring_search_map_cons.h>
#include <apps_sfdl_hw/smallpox_seq.h>
#include <string.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_substring_search_mapVerifierInpGenHw::mr_substring_search_mapVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
  num_inputs_created = 0;
}

//Refer to apps_sfdl_gen/mr_substring_search_map_cons.h for constants to use when generating input.
void mr_substring_search_mapVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  //Default implementation is provided by compiler
  //compiler_implementation.create_input(input_q, num_inputs);
  MapperIn mapper_in;
  {
    int seq_len = SMALLPOX_SEQ_LEN;
    int overlap = SIZE_NEEDLE - 1;
    int distance_between_startpts = SIZE_HAYSTACK - overlap;
    int seq_startpt = distance_between_startpts * num_inputs_created;

    //haystack
#if CURRENT_IMPL == BOOLEAN_IMPL
    for(int i = 0; i < SIZE_HAYSTACK; i++){
      if (seq_startpt + i < seq_len){
	mapper_in.serverside_in.haystack[i*2] = ((uint64_t)rand()) * rand(); //SMALLPOX_SEQ[seq_startpt + i];
	mapper_in.serverside_in.haystack[i*2 + 1] = ((uint64_t)rand()) * rand(); //SMALLPOX_SEQ[seq_startpt + i];
      } else {
	mapper_in.serverside_in.haystack[i*2] = 0; //Pad ending with 0.
	mapper_in.serverside_in.haystack[i*2 + 1] = 0; //Pad ending with 0.
      }
    }
    //needles
    for(int i = 0; i < NUM_NEEDLES; i++){
      for(int j = 0; j < SIZE_NEEDLE; j++) {
	mapper_in.clientside_in.needle[i][j*2] = ((uint64_t)rand()) * rand(); //to create a 64 bit number //SMALLPOX_NEEDLES[i][j];
	mapper_in.clientside_in.needle[i][j*2 + 1] = ((uint64_t)rand()) * rand(); //to create a 64 bit number //SMALLPOX_NEEDLES[i][j];
      }
    }
#else
    for(int i = 0; i < SIZE_HAYSTACK_INTS; i++){
      if (seq_startpt + i < seq_len){
	mapper_in.serverside_in.haystack[i] = ((uint64_t)rand()) * rand(); //SMALLPOX_SEQ[seq_startpt + i];
      } else {
	mapper_in.serverside_in.haystack[i] = 0; //Pad ending with 0.
      }
    }

    //needles
    for(int i = 0; i < NUM_NEEDLES; i++){
      for(int j = 0; j < SIZE_NEEDLE_INTS; j++) {
	mapper_in.clientside_in.needle[i][j] = ((uint64_t)rand()) * rand(); //to create a 64 bit number //SMALLPOX_NEEDLES[i][j];
      }
    }
#endif

  }


  gmp_printf("Size of clientside in: %d, serverside in: %d\n",
  sizeof(ClientsideIn), sizeof(ServersideIn));

  hash_t digest_clientside;
  export_exo_inputs(&mapper_in.clientside_in, sizeof(ClientsideIn), &digest_clientside);

  hash_t digest_serverside;
  export_exo_inputs(&mapper_in.serverside_in, sizeof(ServersideIn), &digest_serverside);

  //dump_block(sizeof(mapper_in), (void *)&mapper_in, (const char *)hexstring); 
  export_digests_to_input(input_q, &digest_serverside, &digest_clientside);

  //cout<<"hash="<<input_name<<endl;
  //cout<<"Num inputs are "<<num_inputs<<endl;
  //cout<<"Num hash chunks "<<NUM_HASH_CHUNKS<<endl;
  //cout<<"Size of input is "<<SIZE_INPUT<<endl;

  num_inputs_created++;
}
