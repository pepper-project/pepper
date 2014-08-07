#include <apps_sfdl_gen/mr_genotyper_map_v_inp_gen.h>
#include <apps_sfdl_hw/mr_genotyper_map_v_inp_gen_hw.h>
#include <apps_sfdl_gen/mr_genotyper_map_cons.h>
#include <apps_sfdl/mr_genotyper.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

mr_genotyper_mapVerifierInpGenHw::mr_genotyper_mapVerifierInpGenHw(Venezia* v_)
{
  v = v_;
  compiler_implementation.v = v_;
}

//Refer to apps_sfdl_gen/mr_genotyper_map_cons.h for constants to use when generating input.
void mr_genotyper_mapVerifierInpGenHw::create_input(mpq_t* input_q, int num_inputs)
{
  MapperIn mapper_in;
  for(int i = 0; i < NUM_LOCI_PER_MAPPER; i++){
    sequence_pileup *pileup = &(mapper_in.data[i]);
    for(int j = 0; j < MAX_SIZE_PILEUP; j++){
      nucleotide_read *read = &(pileup->pileup[j]);
      read->base = rand() & 3;
      read->quality = ((unsigned int)rand()) % 50;
    }
    mapper_in.refseq[i] = rand() & 3;
  }

  hash_t digest;
  export_exo_inputs(&mapper_in, sizeof(MapperIn), &digest);

  digest_to_mpq_vec(input_q, &digest);
}
