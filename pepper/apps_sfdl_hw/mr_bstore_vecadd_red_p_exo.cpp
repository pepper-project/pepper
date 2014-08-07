#include <apps_sfdl_hw/mr_bstore_vecadd_red_p_exo.h>
#include <apps_sfdl_gen/mr_bstore_vecadd_red_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>
#include <apps_sfdl_hw/mr_bstore_vecadd_red_v_inp_gen_hw.h>
#include <storage/exo.h>

#pragma pack(push)
#pragma pack(1)

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_bstore_vecadd_redProverExo::mr_bstore_vecadd_redProverExo() { }

using namespace mr_bstore_vecadd_red_cons;

void mr_bstore_vecadd_redProverExo::init_exo_inputs(
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

void mr_bstore_vecadd_redProverExo::export_exo_inputs(
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

void mr_bstore_vecadd_redProverExo::run_shuffle_phase(char *folder_path) {
  run_mapred_shuffle_phase(NUM_MAPPERS, NUM_REDUCERS, NUM_HASH_CHUNKS*NUM_MAPPERS, folder_path);
}

void mr_bstore_vecadd_redProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  // Fill code here to prepare input from input_q.
  ReducerIn reducer_input;
  ReducerOut reducer_output;
  int i;

  hash_t digest; int k=0;
  // read input exogeneously using the client-provided digest
  for (i=0; i<NUM_MAPPERS; i++) {
    for (int j=0; j<NUM_HASH_CHUNKS; j++) {
      digest.bit[j] = mpz_get_ui(mpq_numref(input_q[k++]));
    }
    
    hashget2(_bs, &reducer_input.input[i], &digest);
  }

  // Do the computation
  reducer_output.output = 0;
  for(i = 0; i < NUM_MAPPERS; i++) {
    reducer_output.output = reducer_output.output + (reducer_input.input[i]).input;
  }

  // store the output of the reducer
  hashput(&digest, &reducer_output);
 

  // Fill code here to dump output to output_recomputed.
  k = 0;
  mpq_set_ui(output_recomputed[k++], 0, 1);
  for (int j=0; j<NUM_HASH_CHUNKS; j++) {
    mpq_set_ui(output_recomputed[k], digest.bit[j], 1); 
    k++;
  }
}

//Refer to apps_sfdl_gen/mr_bstore_vecadd_red_cons.h for constants to use in this exogenous
//check.
bool mr_bstore_vecadd_redProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
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
