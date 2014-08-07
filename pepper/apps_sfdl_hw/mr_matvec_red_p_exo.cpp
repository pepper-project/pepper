#include <apps_sfdl_hw/mr_matvec_red_p_exo.h>
#include <apps_sfdl_gen/mr_matvec_red_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_matvec_redProverExo::mr_matvec_redProverExo() {

  baseline_minimal_input_size = sizeof(ReducerIn);
  baseline_minimal_output_size = sizeof(ReducerOut);

}

using namespace mr_matvec_red_cons;

void mr_matvec_redProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  _bs = bs;
  
  ReducerChunkIn reducer_chunk_in;
  hash_t digest;
  
  int k=0;
  for (int i=0; i<NUM_MAPPERS; i++) {
    for (int j=0; j<NUM_HASH_CHUNKS; j++) {
      digest.bit[j] = mpz_get_ui(mpq_numref(input_q[k]));
      k++;
    }
  
    import_exo_inputs_to_bs(bs, &reducer_chunk_in, sizeof(ReducerChunkIn), &digest); 
  }
  
}

void mr_matvec_redProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {
  ReducerOut reducer_out;
  hash_t digest;

  //IMPORTANT: 0th element in output_q is return status!! 
  int k=1;
  for (int j=0; j<NUM_HASH_CHUNKS; j++) {
    digest.bit[j] = mpz_get_ui(mpq_numref(output_q[k]));
    k++;
  }
  export_exo_inputs_from_bs(bs, &reducer_out, sizeof(reducer_out), &digest);
}

void mr_matvec_redProverExo::run_shuffle_phase(char *folder_path) {
  run_mapred_shuffle_phase(NUM_MAPPERS, NUM_REDUCERS, NUM_HASH_CHUNKS*NUM_MAPPERS, folder_path);
}

void mr_matvec_redProverExo::baseline_minimal(void* input, void*
output){
  ReducerIn* reducer_input = (ReducerIn*)input;
  ReducerOut* reducer_output = (ReducerOut*)output;

  int i,j,k;

  j = 0;
  for(i = 0; i < NUM_MAPPERS; i++) {
    for(k = 0; k < NUM_ROWS_PER_MAPPER; k++){
      reducer_output->product[j] = (reducer_input->input[i]).product_part[k];
      j++;
    }
  }
}

void mr_matvec_redProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  // Fill code here to prepare input from input_q.
  ReducerIn reducer_in;
  ReducerOut reducer_out;
  int i,j;

  hash_t digest; int k=0;
  // read input exogeneously using the client-provided digest
  for (i=0; i<NUM_MAPPERS; i++) {
    mpq_vec_to_digest(&digest, input_q + k);
    k += NUM_HASH_CHUNKS;
    
    hashget2(_bs, &reducer_in.input[i], &digest);
  }

  ReducerIn* reducer_input = &reducer_in;
  ReducerOut* reducer_output = &reducer_out;
  // Do the computation
  baseline_minimal(reducer_input, reducer_output);

  // store the output of the reducer
  hashput(&digest, reducer_output);

  // Fill code here to dump output to output_recomputed.
  k = 0;
  mpq_set_ui(output_recomputed[k++], 0, 1);
  digest_to_mpq_vec(output_recomputed + k, &digest);
}

//Refer to apps_sfdl_gen/mr_matvec_red_cons.h for constants to use in this exogenous
//check.
bool mr_matvec_redProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      passed_test = false;
      break;
    }
  }
  clear_vec(num_outputs, output_recomputed);
#else
  gmp_printf("<Exogenous check disabled>\n");
#endif
  return passed_test;
};

#pragma pack(pop)
