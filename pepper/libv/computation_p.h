#ifndef CODE_PEPPER_LIBV_COMPUTATION_P_H_
#define CODE_PEPPER_LIBV_COMPUTATION_P_H_

#include <libv/prover.h>
#include <iterator>
#include <sstream>
#include <storage/hasher.h>
#include <storage/hash_block_store.h>
#include <boost/dynamic_bitset.hpp>



#include <NTL/ZZ.h>
#include <NTL/ZZ_p.h>
#include <NTL/vec_ZZ_p.h>
#include <NTL/ZZ_pX.h>
NTL_CLIENT

class MerkleRAM;

class ComputationProver : public Prover {
  protected:
    string shared_bstore_file_name;

    uint32_t *F1_index;
    int temp_stack_size;
    int num_vars, num_cons;
    mpz_t temp, temp2;
    mpz_t *input, *output, *input_output;
    mpq_t *input_q, *output_q, *input_output_q, *temp_qs, *F1_q;
    mpq_t temp_q, temp_q2, temp_q3;
    int size_input, size_output, size_constants, size_f1_vec, size_f2_vec;
    
    // folder path where blockstore is created/stored
    char bstore_file_path[BUFLEN];

    mpz_t *F1, *F2;
    mpz_t *f1_commitment, *f2_commitment;
    mpz_t *f1_consistency, *f2_consistency;
    mpz_t *f1_q1, *f1_q2, *f1_q3, *f1_q4, *f1_q5;
    mpz_t *f2_q1, *f2_q2, *f2_q3, *f2_q4;
    
    MerkleRAM* _ram;
    HashBlockStore* _blockStore;

    void init_block_store();

    void compute_from_pws(const char* pws_filename);
    mpq_t& voc(const char*, mpq_t& use_if_constant);
    void compute_poly(FILE* pws_file, int);
    void compute_less_than_int(FILE* pws_file);
    void compute_less_than_float(FILE* pws_file);
    void compute_split_unsignedint(FILE* pws_file);
    void compute_split_int_le(FILE* pws_file);

    void compute_db_get_bits(FILE* pws_file);
    void compute_db_put_bits(FILE* pws_file);
    void compute_db_get_sibling_hash(FILE* pws_file);

    void compute_exo_compute(FILE *pws_file);
    void compute_exo_compute_getLL(std::vector< std::vector<std::string> > &inLL, FILE *pws_file, char *buf);
    void compute_exo_compute_getL (std::vector<std::string> &inL, FILE *pws_file, char *buf);

    void compute_fast_ramget(FILE* pws_file);
    void compute_fast_ramput(FILE* pws_file);

    void parse_hash(FILE* pws_file, HashBlockStore::Key& outKey, int numHashBits);
    void compute_matrix_vec_mul(FILE* pws_file);
    void compute_benes_network(FILE* pws_file);
    void compute_get_block_by_hash(FILE* pws_file);
    void compute_put_block_by_hash(FILE* pws_file);
    void compute_free_block_by_hash(FILE* pws_file);

    void compute_genericget(FILE* pws_file);
    void compute_printf(FILE* pws_file);

  public:
    ComputationProver(int ph, int b_size, int num_r, int size_input, const char *name_prover);
    ComputationProver(int ph, int b_size, int num_r, int size_input, const char *name_prover, const char *_shared_bstore_file_name);
    ~ComputationProver();
};

#endif  // CODE_PEPPER_LIBV_COMPUTATION_P_H_
