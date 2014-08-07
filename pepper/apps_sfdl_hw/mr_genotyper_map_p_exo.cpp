#include <apps_sfdl_hw/mr_genotyper_map_p_exo.h>
#include <apps_sfdl_gen/mr_genotyper_map_cons.h>
#include <common/sha1.h>
#include <storage/configurable_block_store.h>

#pragma pack(push)
#pragma pack(1)

#include <apps_sfdl/mr_genotyper_map.c>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_genotyper_mapProverExo::mr_genotyper_mapProverExo() {

  //Uncomment and fix to specify the sizes of the input and output types
  //to baseline_minimal:
  baseline_minimal_input_size = sizeof(MapperIn);
  baseline_minimal_output_size = sizeof(MapperOut);

}

//using namespace mr_genotyper_map_cons;

void mr_genotyper_mapProverExo::init_exo_inputs(
  const mpq_t* input_q, int num_inputs,
  char *folder_path, HashBlockStore *bs) {
  _bs = bs;

  MapperIn mapper_in;
  hash_t digest;

  mpq_vec_to_digest(&digest, input_q);

  import_exo_inputs_to_bs(bs, &mapper_in, sizeof(MapperIn), &digest);
}

void mr_genotyper_mapProverExo::export_exo_inputs(
  const mpq_t* output_q, int num_outputs,
  char* folder_path, HashBlockStore *bs) {
  MapperChunkOut mapper_chunk_out;
  hash_t digest;

  //IMPORTANT: 0th element in output_q is return status!! 
  int k=1;
  for (int i=0; i<NUM_REDUCERS; i++) {
    mpq_vec_to_digest(&digest, output_q + k);
    k += NUM_HASH_CHUNKS;
    export_exo_inputs_from_bs(bs, &mapper_chunk_out,
sizeof(MapperChunkOut), &digest);
  }

}

void mr_genotyper_mapProverExo::run_shuffle_phase(char *folder_path) {

}

void mr_genotyper_mapProverExo::baseline_minimal(void* input, void* output){
  //Run the computation
  MapperIn* mapper_input = (MapperIn*)input;
  MapperOut* mapper_output = (MapperOut*)output;

  //Note - at the moment, map reduce computations cannot use hashget or
  //hashput (because the block store isn't configured like the other
  //apps).

  //The two colons are necessary because std::map exists (C++ weirdness)
  ::map(mapper_input, mapper_output);
}

void mr_genotyper_mapProverExo::baseline(const mpq_t* input_q, int num_inputs, 
      mpq_t* output_recomputed, int num_outputs) {
  int i,j,k;
  MapperIn mapper_in;
  MapperOut mapper_out;

  hash_t digest;
  for(i = 0; i < num_inputs; i++){
    digest.bit[i] = mpz_get_ui(mpq_numref(input_q[i]));
  }

  hashget2(_bs, &mapper_in, &digest);

  baseline_minimal(&mapper_in, &mapper_out);

  //Code to dump output to output recomputed
  mpq_set_ui(output_recomputed[0], 0, 1);
  k = 1;
  for(i = 0; i < NUM_REDUCERS; i++){
    hashput(&digest, &(mapper_out.output[i]));
    for(int j = 0; j < NUM_HASH_CHUNKS; j++){
      mpq_set_ui(output_recomputed[k], digest.bit[j], 1);
      k++;
    }
  }
}

//Refer to apps_sfdl_gen/mr_genotyper_map_cons.h for constants to use in this exogenous
//check.
bool mr_genotyper_mapProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
      int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {

  bool passed_test = true;
#ifdef ENABLE_EXOGENOUS_CHECKING
  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for(int i = 0; i < num_outputs; i++){
    if (mpq_equal(output_recomputed[i], output_q[i]) == 0){
      gmp_printf("Index %d Should be %Qd was %Qd", i, output_recomputed[i],
      output_q[i]);
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
