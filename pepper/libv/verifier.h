#ifndef CODE_PEPPER_LIBV_VERIFIER_H_
#define CODE_PEPPER_LIBV_VERIFIER_H_

#include <libv/libv.h>
#include <common/curl_util.h>
#include <common/measurement.h>
#include <common/utility.h>
#include <vector>

#ifdef INTERFACE_MPI
#include <libv/mpi_constants.h>
#include "mpi.h"
#endif

#define INPUT_FILE_NAME_SUBSTR "input"
#define OUTPUT_FILE_NAME_SUBSTR "output"

#define VERBOSE 0

class Verifier {
  protected:
    Venezia * v;
    Measurement m_plainq, m_runtests;

    int batch_size;
    int num_repetitions;
    int input_size;
    int optimize_answers;
    uint32_t num_bits_in_input;
    uint32_t num_bits_in_prime;
    char scratch_str[BUFLEN];
    char scratch_str2[BUFLEN];
    mpz_t prime;
    int expansion_factor;
    int crypto_in_use, png_in_use;
    mpz_t *cipher;
    mpz_t *c_values;
    int num_vars, num_cons;
    bool status_verifier_tests;
    mpz_t a, f_s;
    int num_verification_runs;

    mpz_t *input, *output;
    mpq_t *input_q, *output_q;
    int size_input, size_output, size_constants, size_f1_vec, size_f2_vec;

    long network_bytes_sent, network_bytes_input_sent;
    long network_bytes_rcvd, network_bytes_output_rcvd;
    double network_send_time_elapsed;
    double network_rcv_time_elapsed;

    // queries of different sizes.
    vector < uint32_t > commitment_query_sizes;
    vector < mpz_t * >f_commitment_ptrs;
    vector < mpz_t * >f_consistency_ptrs;
    vector < mpz_t * >con_coins_ptrs;
    vector < mpz_t * >temp_arr_ptrs;
    vector < mpz_t * >answers_ptrs;
    vector < mpz_t * >answers_rfetch_ptrs;
    vector < uint32_t > Q_list;
    uint32_t num_lin_pcp_queries;
    vector < uint32_t > pcp_queries;
    mpz_t *f_answers;
    mpz_t *f_con_coins;

    // curl variables
    CurlUtil *curl;
    char prover_query_url[BUFLEN];
    char prover_upload_url[BUFLEN];
    char prover_download_url[BUFLEN];
    char download_url[BUFLEN];
    char full_file_name[BUFLEN];
    char full_url[BUFLEN];

    virtual void create_input() = 0;
#if NONINTERACTIVE == 1
    virtual void create_noninteractive_query() {}
#endif
    virtual void create_plain_queries() = 0;
    virtual bool run_correction_and_circuit_tests(uint32_t beta) = 0;
    virtual void populate_answers(mpz_t *f_answers, int rho, int num_repetitions, int beta) = 0;
    virtual void prepare_noninteractive_answers(uint32_t) {}

    void init_state();
    void prepare_answers(int);
    bool run_tests(int);
    void init_server_variables(char *prover_url, const char *prover_name);
    void send_file(char *file_name);
    void recv_file(const char *file_name);
    void invoke_prover(int prover_phase);
    void create_commitment_query();
    void recv_comm_answers();
    void recv_outputs();
    void recv_plain_answers();
    //void begin_interactive_pepper();
    //void begin_noninteractive_pepper();

  public:
    Verifier(int batch, int reps, int ip_size, int optimize_answers,
             char *prover_host_url, const char *prover_name);
    ~Verifier(void);
    void begin_pepper();
};
#endif  // CODE_PEPPER_LIBV_VERIFIER_H_
