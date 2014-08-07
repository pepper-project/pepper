#ifndef CODE_PEPPER_LIBV_PROVER_H_
#define CODE_PEPPER_LIBV_PROVER_H_

#include <libv/libv.h>
#include <common/curl_util.h>
#include <common/utility.h>
#include <common/measurement.h>
#include <libv/exogenous_checker.h>
//#include <map>
#ifdef INTERFACE_MPI
#include <libv/mpi_constants.h>
#include "mpi.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#include <libconfig.h++>
#include <iostream>
#include <fcgi_stdio.h>

using namespace libconfig;

#ifndef INTERFACE_MPI
#define PROVER_CONFIG_FILE "/var/www/config/prover.cfg"
#else
#define PROVER_CONFIG_FILE "config/prover.cfg"
#endif

// Names used in the prover config file.
#define SERVER_CONFIG_NAME "servers"
#define FILES "files"
#define MASTER "master"
#define SLAVE "slave"
#define PHASE1 "phase1"
#define PHASE2 "phase2"

// the following flags are set via the compiler flags in GNUmakefile (so
// that it's easy to control them via python scripts)
// #define NUM_PROVERS_MAX 4
// #define USE_GPU 1

class Prover {
  protected:
    Venezia * v;
    Measurement m_computation, m_interpret_cons, m_answer_queries, m_proofv_creation, m_plainq;
    Measurement m_computation_minimal, m_computation_sha;
    uint32_t num_bits_in_input;
    uint32_t num_bits_in_prime;
    char scratch_str[BUFLEN];
    char scratch_str2[BUFLEN];
    int crypto_in_use;
    int png_in_use;
    int expansion_factor;
    mpz_t prime, con_answer, *dotp, *dotp2;
    int phase;
    int batch_size, batch_start, batch_end;
    int num_repetitions;
    int m;
    int optimize_answers;
    mpz_t answer;
    long network_bytes_sent;
    double network_send_time_elapsed;
    int num_lin_pcp_queries;
    int num_vars, num_cons;
    int num_local_runs;
    int num_interpret_runs;

    mpz_t *f_answers;
    vector < mpz_t * > F_ptrs;
    vector < mpz_t * > f_q_ptrs;
    vector < mpz_t * > f_q2_ptrs;
    vector < mpz_t * > f_q3_ptrs;
    vector < int > sizes;
    vector < int > qquery_sizes;
    vector < mpz_t * > qquery_f_ptrs;
    vector < mpz_t * > qquery_F_ptrs;
    vector < int > qquery_q_ptrs;

    CurlUtil *curl;
    char prover_name[BUFLEN];
    char prover_query_url[BUFLEN];
    char prover_upload_url[BUFLEN];
    char prover_download_url[BUFLEN];
    char tmp_buf[BUFLEN];

    vector<string> prover_nodes;
    int is_master;
    vector<string> master_files_p1;
    vector<string> master_files_p2;
    vector<string> slave_files_p1;
    vector<string> slave_files_p2;
    char FOLDER_WWW_DOWNLOAD[BUFLEN];
    ExogenousChecker* exogenous_checker;

#ifdef INTERFACE_MPI
    int num_procs, rank;
#endif

    void init_state();
    void init_prover_urls(const char *prover_url, const char *prover_name);
    void write_answer_to_file(mpz_t answer, char *a_name, FILE *fp);
    void prover_answer_query(uint32_t size, mpz_t * q, char *q_name, mpz_t
                             * assignment, mpz_t answer, mpz_t prime,
                             double *par_time);
    void prover_answer_queries(double *par_time);
    void send_file(char *file, char *url);

    virtual void prover_computation_commitment() = 0;
    virtual void find_cur_qlengths() = 0;
    virtual void deduce_queries() = 0;
  public:
    Prover(int ph, int b_size, int num_r, int i_size, const char *name);
    ~Prover();
    void handle_terminal_request();
#ifdef INTERFACE_MPI
    void handle_requests();
#else
    void handle_http_requests();
#endif
#ifdef INTERFACE_MPI
    void send_file_MPI(const char *filename, int rank, int tag, int send_name);
    int recv_file_MPI();
#endif

    void init_from_config();
    bool filename_in_list(const vector<string> &vec, const string &file);
    void upload_files_to_master();
    void distribute_and_invoke(int index, const char *server_url, int batch_start,
                               int batch_end);
};
#endif  // CODE_PEPPER_LIBV_PROVER_H_
