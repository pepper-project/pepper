#include <apps_sfdl_hw/mr_substring_search_map_p_exo.h>
#include <apps_sfdl_gen/mr_substring_search_map_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>
//This include should be automatically generated 
#include <apps_sfdl_hw/mr_substring_search_map_v_inp_gen_hw.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/mr_substring_search_map.c>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_substring_search_mapProverExo::mr_substring_search_mapProverExo() {

  baseline_minimal_input_size = sizeof(MapperIn);
  baseline_minimal_output_size = sizeof(MapperOut);

}

using namespace mr_substring_search_map_cons;

void mr_substring_search_mapProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs, char *folder_path, HashBlockStore *bs) {

  _bs = bs;

  MapperIn mapper_in;
  hash_t clientside_in_d;
  hash_t serverside_in_d;

  //Write input_q to the digests
  import_digests_from_input(input_q, &serverside_in_d, &clientside_in_d);

  //Read in the digests in preparation for running.
  import_exo_inputs_to_bs(bs, &mapper_in.clientside_in, sizeof(ClientsideIn), &clientside_in_d);
  import_exo_inputs_to_bs(bs, &mapper_in.serverside_in, sizeof(ServersideIn), &serverside_in_d);
}

void mr_substring_search_mapProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {

  MapperChunkOut mapper_chunk_out;
  hash_t digest;

  //IMPORTANT: 0th element in output_q is return status!! 
  int k=1;
  for (int i=0; i<NUM_REDUCERS; i++) {
    for (int j=0; j<NUM_HASH_CHUNKS; j++) {
      digest.bit[j] = mpz_get_ui(mpq_numref(output_q[k]));
      k++;
    }
    export_exo_inputs_from_bs(bs, &mapper_chunk_out,
sizeof(MapperChunkOut), &digest);
  }

}

void mr_substring_search_mapProverExo::run_shuffle_phase(char *folder_path) {

}

void mr_substring_search_mapProverExo::baseline_minimal(void* input,
void* output){
  MapperIn* mapper_input = (MapperIn*)input;
  MapperOut* mapper_output = (MapperOut*)output;

  ::map(mapper_input, mapper_output);
}

void mr_substring_search_mapProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  MapperIn mapper_in_;
  MapperOut mapper_out_;

  hash_t clientside_in;
  hash_t serverside_in;
  hash_t digest;

  import_digests_from_input(input_q, &serverside_in, &clientside_in);

  MapperIn *mapper_input = &mapper_in_;
  MapperOut *mapper_output = &mapper_out_;

  hashget2(_bs, &mapper_input->clientside_in, &clientside_in);
  hashget2(_bs, &mapper_input->serverside_in, &serverside_in);

  // Do the computation

  baseline_minimal(mapper_input, mapper_output);

  int i,j,k;

  // Fill code here to dump output to output_recomputed.
  //IMPORTANT: 0th element in output_q is return status!! 
  mpq_set_ui(output_recomputed[0], 0, 1);

  k=1;
  for (i=0; i<NUM_REDUCERS; i++) {
    hashput(&digest, &(mapper_output->output[i]));

    for (j=0; j<NUM_HASH_CHUNKS; j++) {
      mpq_set_ui(output_recomputed[k], digest.bit[j], 1);
      k++;
    }
  }
}

//Refer to apps_sfdl_gen/mr_substring_search_map_cons.h for constants to use in this exogenous
//check.
bool mr_substring_search_mapProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (i > 0){
      //gmp_printf("Found needle %d @ %Qd\n", i-1, output_recomputed[i]);
    }
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      gmp_printf("Failure: %Qd %Qd\n", output_recomputed[i], output_q[i]);
      passed_test = false;
      //break;
    }
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
