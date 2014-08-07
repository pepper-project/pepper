#include <libv/prover.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#include <libconfig.h++>
#include <storage/exo.h>
#include <common/utility.h>
#include <common/pairing_util.h>
#include <common/memory.h>

Prover::Prover(int ph, int b_size, int num_r, int i_size, const char *name) {
  phase = ph;
  batch_size = b_size;
  num_repetitions = num_r;
  m = i_size;
  prover_name[0] = '\0';
  strcpy(prover_name, name);

  network_bytes_sent = 0;
  network_send_time_elapsed = 0;

#ifdef INTERFACE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
  
  if ((num_procs-1) > NUM_PROVERS_MAX)
    num_procs = NUM_PROVERS_MAX + 1;
 
  if (rank == MPI_COORD_RANK) {
    cout<<"LOG: Maximum number of provers is "<<NUM_PROVERS_MAX<<endl;
    cout<<"LOG: Actual number of provers is "<<(num_procs-1)<<endl;
  }

  if (FOLDER_STATE_SHARED) {
    snprintf(FOLDER_WWW_DOWNLOAD, BUFLEN - 1, "%s", FOLDER_STATE);
  } else {
    snprintf(FOLDER_WWW_DOWNLOAD, BUFLEN - 1, "%s_%d", FOLDER_STATE, rank);
  }
#else
  snprintf(FOLDER_WWW_DOWNLOAD, BUFLEN - 1, "%s", FOLDER_STATE);
#endif
  mkdir(FOLDER_WWW_DOWNLOAD, S_IRWXU);

#if !(NONINTERACTIVE == 1)
  alloc_init_scalar(con_answer);
  alloc_init_vec(&dotp, 2);
  alloc_init_vec(&dotp2, 2);
#endif
}

Prover::~Prover() {
#if !(NONINTERACTIVE == 1)
  clear_scalar(con_answer);
  clear_vec(2, dotp);
  clear_vec(2, dotp2);
#endif
  clear_scalar(prime);

  delete v;
  delete curl;
}

void Prover::init_from_config() {
  Config cfg;

  try {
    cfg.readFile(PROVER_CONFIG_FILE);
  } catch(const FileIOException &fioex) {
    std::cerr << "I/O error while reading file:" << PROVER_CONFIG_FILE << std::endl;
    exit(EXIT_FAILURE);
  } catch(const ParseException &pex) {
    std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine()
              << " - " << pex.getError() << std::endl;
    exit(EXIT_FAILURE);
  }

  prover_nodes.clear();
  const Setting& root = cfg.getRoot();
  const Setting& list = root[SERVER_CONFIG_NAME];
  for (int i = 0; i < list.getLength(); i++) {
    const string server = root[SERVER_CONFIG_NAME][i];
    prover_nodes.push_back(server);
  }

  const Setting& master_p1 = root[FILES][MASTER][PHASE1];
  for (int i = 0; i < master_p1.getLength(); i++)
    master_files_p1.push_back(master_p1[i]);
  const Setting& master_p2 = root[FILES][MASTER][PHASE2];
  for (int i = 0; i < master_p2.getLength(); i++)
    master_files_p2.push_back(master_p2[i]);
  const Setting& slave_p1 = root[FILES][SLAVE][PHASE1];
  for (int i = 0; i < slave_p1.getLength(); i++)
    slave_files_p1.push_back(slave_p1[i]);
  const Setting& slave_p2 = root[FILES][SLAVE][PHASE2];
  for (int i = 0; i < slave_p2.getLength(); i++)
    slave_files_p2.push_back(slave_p2[i]);
}

void Prover::init_state() {
  mpz_init(prime);

  if (crypto_in_use == CRYPTO_ELGAMAL)
    expansion_factor = 2;
  else
    expansion_factor = 1;

#if NONINTERACTIVE == 1
#if PAIRING_LIB == PAIRING_PBC
  snprintf(scratch_str, BUFLEN - 1, "prime_f_%d.txt", num_bits_in_prime);
#else
  snprintf(scratch_str, BUFLEN - 1, "prime_libzm_%d.txt", num_bits_in_prime);
#endif
#else
  snprintf(scratch_str, BUFLEN - 1, "prime_%d.txt", num_bits_in_prime);
#endif

  load_txt_scalar(prime, scratch_str, const_cast<char *>("static_state"));

  v = new Venezia(ROLE_PROVER, crypto_in_use, png_in_use, num_bits_in_prime);
  curl = new CurlUtil();
  init_from_config();
}

void Prover::init_prover_urls(const char *prover_url, const char *name_prover) {
  int remaining = strlen(SERVICE_UPLOAD_NAME) + 1;
  strncpy(prover_upload_url, prover_url, BUFLEN-remaining);
  strcat(prover_upload_url, SERVICE_UPLOAD_NAME);

  remaining = strlen(SERVICE_DOWNLOAD_NAME) + 1;
  strncpy(prover_download_url, prover_url, BUFLEN-remaining);
  strcat(prover_download_url, SERVICE_DOWNLOAD_NAME);

  remaining = strlen(prover_name) + 1;
  strncpy(prover_query_url, prover_url, BUFLEN-remaining);
  strcat(prover_query_url, name_prover);
}

void Prover::write_answer_to_file(mpz_t answer, char *a_name, FILE *fp) {
  if (!optimize_answers) {
    dump_scalar(answer, a_name, FOLDER_WWW_DOWNLOAD);
  } else {
    size_t bytes = mpz_out_raw(FCGI_ToFILE(fp), answer);
    fseek(fp, bytes, SEEK_CUR);
  }
}

void Prover::prover_answer_query(uint32_t size, mpz_t * q, char *q_name,
                                 mpz_t * assignment, mpz_t answer,
                                 mpz_t prime, double *par_time) {
  load_vector(size, q, q_name, FOLDER_WWW_DOWNLOAD);
  v->dot_product_par(size, q, assignment, answer, prime, par_time);
}

void Prover::prover_answer_queries(double *par_time) {
  *par_time = 0;
  std::list<string> files = get_files_in_dir((char *)FOLDER_WWW_DOWNLOAD);
  mpz_t a1, a2;
  alloc_init_scalar(a1);
  alloc_init_scalar(a2);
  
  for (int i = batch_start; i <= batch_end; i++) {
    string assignment_vector = "fX_assignment_vector_b_%d";

    for (uint32_t j = 0; j < F_ptrs.size(); j++) {
      assignment_vector[1] = '0' + (j + 1);
      snprintf(scratch_str, BUFLEN - 1, assignment_vector.c_str(), i);
      load_vector(sizes[j], F_ptrs[j], scratch_str, FOLDER_WWW_DOWNLOAD);
    }

    FILE *fp = NULL;

    if (optimize_answers) {
      char f_name[BUFLEN];

      snprintf(f_name, BUFLEN - 1, "%s/answers_%d", FOLDER_WWW_DOWNLOAD,
               i + 1);
      fp = fopen(f_name, "wb");
      if (fp == NULL) {
        printf("Prover: could not create file %s.", f_name);
        exit(1);
      }
    }

    // consistency query
    string consistency_query = "fX_consistency_query";
    string consistency_answer = "f_consistency_answer_b_%d";
    mpz_set_ui(con_answer, 0);
    for (uint32_t j = 0; j < F_ptrs.size(); j++) {
      consistency_query[1] = '0' + (j + 1);
      //consistency_answer[1] = '0' + (j + 1);
      snprintf(scratch_str, BUFLEN - 1, consistency_query.c_str());
      prover_answer_query(sizes[j], f_q_ptrs[j], scratch_str, F_ptrs[j],
                          answer, prime, par_time);
      mpz_add(con_answer, con_answer, answer);
    }
    snprintf(scratch_str2, BUFLEN - 1, consistency_answer.c_str(), i);
    write_answer_to_file(con_answer, scratch_str2, fp);

    std::list<string> files = get_files_in_dir((char *)FOLDER_WWW_DOWNLOAD);
    for (int rho = 0; rho < num_repetitions; rho++) {
      string answer_str = "q%d_qquery_answer_b_%d_r_%d";
      string query_str = "q%d_qquery_r_%d";

      // hack to answer linearity queries efficiently 
      
      for (uint32_t j=0; j<qquery_sizes.size(); j++) {
        snprintf(scratch_str, BUFLEN - 1, query_str.c_str(), (qquery_q_ptrs[j]+1), rho);
        if (qquery_F_ptrs[j] == NULL) {
          // special signal to use the optimal path to answer this query
          mpz_add(answer, a1, a2);
        } else {
          prover_answer_query(qquery_sizes[j], qquery_f_ptrs[j], (char *)scratch_str,
                              qquery_F_ptrs[j], answer, prime, par_time);
        }
        
        //snprintf(scratch_str, BUFLEN - 1, answer_str.c_str(), qquery_q_ptrs[j]+1, i, rho);
       
        write_answer_to_file(answer, scratch_str, fp);
        // memorize the last two answers
        if (j==0) {
          mpz_set(a2, answer);
        } else {
          mpz_set(a1, a2);
          mpz_set(a2, answer);
        }
        
      }
    }

    if (fp != NULL)
      fclose(fp);
  }
  clear_scalar(a1);
  clear_scalar(a2);
}

// driver to to run the phases of the prover
void Prover::handle_terminal_request() {
  Measurement m_handle_req, m_dist, m_wait;
  m_handle_req.begin_with_init();
  m_dist.begin_with_init();

  if (phase == PHASE_PROVER_DEDUCE_QUERIES && ((batch_start == -1 && batch_end == -1) || is_master == 1)) {
    v->init_prng_decommit_queries();
    deduce_queries();
    return;
  }

  vector<int> batch_start_nums;
#ifdef INTERFACE_MPI
  int num_slaves = num_procs - 1;
  if (is_master) {
#else
  int num_slaves = prover_nodes.size();
  if (batch_start == -1 && batch_end == -1) {
    is_master = 1;
#endif
    // run the shuffle phase in case of MapRed
    // TODO: fix the next statement for Ginger's tailored apps
    if (phase == PHASE_PROVER_COMMITMENT)
      exogenous_checker->run_shuffle_phase((char *)FOLDER_STATE);

    // compute batch size for each server.
    int batches_per_server = batch_size / num_slaves;
    cout<<"LOG: Running with #instances per server "<<batches_per_server<<endl;
    batch_start = 0;
    batch_end = batch_start + batches_per_server - 1;
    if (batch_size % num_slaves != 0)
      batch_end += (batch_size % num_slaves);

    int remote_batch_start = batch_end + 1;
#ifdef INTERFACE_MPI
    for (int i = 2; i <= num_slaves; i++) {
#else
    for (int i = 1; i < prover_nodes.size(); i++) {
#endif
      batch_start_nums.push_back(remote_batch_start);
#ifdef INTERFACE_MPI
      distribute_and_invoke(i, "", remote_batch_start,
                            remote_batch_start + batches_per_server - 1);
#else
      distribute_and_invoke(i, prover_nodes[i].c_str(), remote_batch_start,
                            remote_batch_start + batches_per_server - 1);
#endif
      remote_batch_start += batches_per_server;
    }
  }
  m_dist.end();

  double p_answer_plainq_par = 0;

  switch (phase) {
  case PHASE_PROVER_COMMITMENT:
    // merge this into prover_computation_commitment with compiler flag. prover_noninteractive();
    prover_computation_commitment();
    break;

  case PHASE_PROVER_DEDUCE_QUERIES:
    if (is_master) {
      v->init_prng_decommit_queries();
      deduce_queries();
    }
    break;

  case PHASE_PROVER_PCP:
    if (FOLDER_STATE_SHARED) {
      int created;
      if (rank == MPI_COORD_RANK) {
        for (int i = 2; i <= num_slaves; i++)
          MPI_Send(&created, 1, MPI_INT, i, MPI_QUERY_CREATED, MPI_COMM_WORLD);
      } else {
        MPI_Recv(&created, 1, MPI_INT, MPI_COORD_RANK, MPI_QUERY_CREATED,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
    } else {
    }

    m_answer_queries.begin_with_init();
    prover_answer_queries(&p_answer_plainq_par);
    m_answer_queries.end();
    break;

  default:
    printf("Undefined prover phase %d", phase);
  }

  long network_bytes_sent_slaves = 0;
  double network_send_time_elapsed_slaves = 0;
  long bytes;
  double time;
  m_wait.begin_with_init();
  if (!is_master) {
    sprintf(tmp_buf, "%s/%s_%d_%d", FOLDER_WWW_DOWNLOAD, "DONE", phase,
            batch_start);
    FILE *done_fp = fopen(tmp_buf, "w");
    fprintf(done_fp, "%lu %f", network_bytes_sent,
            network_send_time_elapsed);
    fclose(done_fp);
    upload_files_to_master();
  } else {
#ifdef INTERFACE_MPI
    for (int num = 2; num <= num_slaves;) {
      if (recv_file_MPI() == MPI_PROVER_FINISHED)
        num++;
    }
#endif
    for (uint32_t num = 0; num < batch_start_nums.size();) {
      sprintf(tmp_buf, "%s/%s_%d_%d", FOLDER_WWW_DOWNLOAD, "DONE", phase,
              batch_start_nums[num]);
      if (access(tmp_buf, F_OK) == -1) // file does not exist
        continue;
      FILE *done_fp = fopen(tmp_buf, "r");
      int ret = fscanf(FCGI_ToFILE(done_fp), "%lu", &bytes);
      ret = fscanf(FCGI_ToFILE(done_fp), "%f", &time);
      fclose(done_fp);
      network_bytes_sent_slaves += bytes;
      network_send_time_elapsed_slaves += time;
      num++;
    }
  }
  m_wait.end();
  m_handle_req.end();

  if(is_master) {
#ifdef INTERFACE_MPI
    cout << "phase:" << phase << " p_handle_req " <<
         m_handle_req.get_papi_elapsed_time() << endl;
    cout << "phase:" << phase << " p_distribution " <<
         m_dist.get_papi_elapsed_time() << endl;
    cout << "phase:" << phase << " p_waiting " <<
         m_wait.get_papi_elapsed_time() << endl;
    cout << "phase:" << phase << " p_master_net_bytes_sent "
         << network_bytes_sent << endl;
    cout << "phase:" << phase << " p_master_send_time_elapsed " <<
         network_send_time_elapsed << endl;
    cout << "phase:" << phase << " p_slaves_net_bytes_sent " <<
         network_bytes_sent_slaves << endl;
    cout << "phase:" << phase << " p_slaves_send_time_elapsed " <<
         network_send_time_elapsed_slaves << endl;
#else
    printf("phase:%d p_handle_req %f\n", phase, m_handle_req.get_papi_elapsed_time());
    printf("phase:%d p_distribution %f\n", phase, m_dist.get_papi_elapsed_time());
    printf("phase:%d p_waiting %f\n", phase, m_wait.get_papi_elapsed_time());
    printf("phase:%d p_master_net_bytes_sent %lu\n", phase, network_bytes_sent);
    printf("phase:%d p_master_send_time_elapsed %f\n", phase,
           network_send_time_elapsed);
    printf("phase:%d p_slaves_net_bytes_sent %lu\n", phase,
           network_bytes_sent_slaves);
    printf("phase:%d p_slaves_send_time_elapsed %f\n", phase,
           network_send_time_elapsed_slaves);
#endif
  }
}

bool Prover::filename_in_list(const vector<string> &vec, const string &file) {
  vector<string>::const_iterator it = vec.begin();
  for (; it != vec.end(); it++) {
    if (file.find(*it) != string::npos)
      return true;
  }
  return false;
}

void Prover::distribute_and_invoke(int index, const char *server_url,
                                   int batch_start_remote, int batch_end_remote) {
#ifdef INTERFACE_MPI
  if (phase == PHASE_PROVER_COMMITMENT) {
    MPI_Send(&batch_size, 1, MPI_INT, index, MPI_PARAMS, MPI_COMM_WORLD);
    MPI_Send(&num_repetitions, 1, MPI_INT, index, MPI_PARAMS,
             MPI_COMM_WORLD);
    MPI_Send(&m, 1, MPI_INT, index, MPI_PARAMS, MPI_COMM_WORLD);
    MPI_Send(&optimize_answers, 1, MPI_INT, index, MPI_PARAMS,
             MPI_COMM_WORLD);
    MPI_Send(&batch_start_remote, 1, MPI_INT, index, MPI_PARAMS,
             MPI_COMM_WORLD);
    MPI_Send(&batch_end_remote, 1, MPI_INT, index, MPI_PARAMS,
             MPI_COMM_WORLD);
  }
#else
  init_prover_urls(server_url, prover_name);
#endif

  std::list<string> files = get_files_in_dir((char *)FOLDER_WWW_DOWNLOAD);
  std::list<string>::const_iterator it = files.begin();
  for (; it != files.end(); it++) {
    string file_name = *it;
    if (phase == PHASE_PROVER_COMMITMENT) {
      if (!filename_in_list(master_files_p1, file_name))
        continue;
    } else if (phase == PHASE_PROVER_PCP) {
      if (!filename_in_list(master_files_p2, file_name))
        continue;
    }

    size_t b_index = file_name.find("_b_");
    if (b_index != string::npos) {
      int b_num = atoi(file_name.substr(b_index + strlen("_b_"),
                                        file_name.length() - 1).c_str());
      if (b_num >= batch_start_remote && b_num <= batch_end_remote) {
#ifdef INTERFACE_MPI
        send_file_MPI(file_name.c_str(), index, MPI_FILE_SEND, 1);
#else
        sprintf(tmp_buf, "%s/%s", FOLDER_WWW_DOWNLOAD, file_name.c_str());
        send_file(tmp_buf, prover_upload_url);
#endif
      }
    } else {
#ifdef INTERFACE_MPI
      send_file_MPI(file_name.c_str(), index, MPI_FILE_SEND, 1);
#else
      sprintf(tmp_buf, "%s/%s", FOLDER_WWW_DOWNLOAD, file_name.c_str());
      send_file(tmp_buf, prover_upload_url);
#endif
    }

  }

#ifdef INTERFACE_MPI
  char *tmp_str = (char *)"start";
  MPI_Send(tmp_str, strlen(tmp_str) + 1, MPI_CHAR, index, MPI_INVOKE_PROVER,
           MPI_COMM_WORLD);
  MPI_Send(&phase, 1, MPI_INT, index, MPI_PARAMS, MPI_COMM_WORLD);

#else

  sprintf(tmp_buf,
          "%s?phase=%d&batch_size=%d&batch_start=%d&batch_end=%d&reps=%d&m=%d&opt=%d",
          prover_query_url, phase, batch_size, batch_start_remote, batch_end_remote,
          num_repetitions, m, optimize_answers);
  curl->get_nonblocking(tmp_buf);

#endif
}

void Prover::upload_files_to_master() {
#ifndef INTERFACE_MPI
  string master = prover_nodes[0];
  init_prover_urls(master.c_str(), prover_name);
#endif
  std::list<string> files = get_files_in_dir((char *)FOLDER_WWW_DOWNLOAD);

  char name_tmp[BUFLEN];
  std::list<string>::const_iterator it = files.begin();
  for (; it != files.end(); it++) {
    string file_name = *it;
    if (phase == PHASE_PROVER_COMMITMENT) {
      if (!filename_in_list(slave_files_p1, file_name))
        continue;
    } else if (phase == PHASE_PROVER_PCP) {
      if (!filename_in_list(slave_files_p2, file_name))
        continue;
    }
#ifdef INTERFACE_MPI
    send_file_MPI(file_name.c_str(), MPI_COORD_RANK, MPI_FILE_SEND + rank, 1);
#else
    sprintf(name_tmp, "%s/%s", FOLDER_WWW_DOWNLOAD, file_name.c_str());
    send_file(name_tmp, prover_upload_url);
#endif
  }

#ifdef INTERFACE_MPI
  int status = 0;
  MPI_Send(&status, 1, MPI_INT, MPI_COORD_RANK, MPI_PROVER_FINISHED,
           MPI_COMM_WORLD);
#endif
}

void Prover::send_file(char *file, char *url) {
  int size;
  double time;
  curl->send_file(file, url, &size, &time);
  network_bytes_sent += size;
  network_send_time_elapsed += time;
}

#ifdef INTERFACE_MPI
void Prover::send_file_MPI(const char *file_name, int rcv_rank, int tag,
                           int send_name) {
  if (!FOLDER_STATE_SHARED) {
    if (send_name) {
      MPI_Send(const_cast<char*> (file_name), strlen(file_name)+1, MPI_CHAR,
               rcv_rank, tag, MPI_COMM_WORLD);
    }
    snprintf(tmp_buf, BUFLEN - 1, "%s/%s", FOLDER_WWW_DOWNLOAD, file_name);
    off_t file_size = get_file_size(tmp_buf);

    // this condition checks if the file does not exist
    if (file_size == -1)
      file_size = 0;
    char *buf = new char[file_size];
    if (file_size > 0) {
      FILE *fp = fopen(tmp_buf, "r");
      int count = fread(buf, 1, file_size, fp);
      fclose(fp);
    }
    MPI_Send(buf, file_size, MPI_BYTE, rcv_rank, tag, MPI_COMM_WORLD);
    delete[] buf;
  }
}

int Prover::recv_file_MPI() {
  char file_name[1024];
  MPI_Status stat;
  MPI_Recv(file_name, 1024, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG,
           MPI_COMM_WORLD, &stat);
  int recv_tag = stat.MPI_TAG;

  if (recv_tag == MPI_FILE_RECV) {
    send_file_MPI(file_name, 0, MPI_FILE_RECV, 0);
    return 0;
  } else if (recv_tag >= MPI_FILE_SEND) {
    MPI_Probe(MPI_ANY_SOURCE, recv_tag, MPI_COMM_WORLD, &stat);
    int file_size;
    MPI_Get_count(&stat, MPI_BYTE, &file_size);
    char *buf = new char[file_size];
    MPI_Recv(buf, file_size, MPI_BYTE, MPI_ANY_SOURCE, recv_tag,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    snprintf(scratch_str, BUFLEN - 1, "%s/%s", FOLDER_WWW_DOWNLOAD, file_name);
    FILE *fp = fopen(scratch_str, "w");
    fwrite(buf, file_size, 1, fp);
    fclose(fp);
    delete[] buf;
  }

  return stat.MPI_TAG;
}
#endif

#ifdef INTERFACE_MPI
void Prover::handle_requests() {
  batch_start = -1;
  batch_end = -1;
  is_master = 0;

  double prover_total_cpu = 0;

  MPI_Recv(&batch_size, 1, MPI_INT, MPI_ANY_SOURCE, MPI_PARAMS,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&num_repetitions, 1, MPI_INT, MPI_ANY_SOURCE, MPI_PARAMS,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(&m, 1, MPI_INT, MPI_ANY_SOURCE, MPI_PARAMS, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
  MPI_Recv(&optimize_answers, 1, MPI_INT, MPI_ANY_SOURCE, MPI_PARAMS,
           MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  if (rank != MPI_COORD_RANK) {
    MPI_Recv(&batch_start, 1, MPI_INT, MPI_COORD_RANK, MPI_PARAMS,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&batch_end, 1, MPI_INT, MPI_COORD_RANK, MPI_PARAMS,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    is_master = 0;
  } else {
    is_master = 1;
  }

  while(1) {
    int tag = -1;
    do {
      tag = recv_file_MPI();
      if (tag == MPI_TERMINATE) {
        char *tmp_str = (char *)"terminate";  // content of string doesnt matter.
        for (int i = 2; i <= (num_procs - 1); i++)
          MPI_Send(tmp_str, strlen(tmp_str) + 1, MPI_CHAR, i,
                   MPI_TERMINATE, MPI_COMM_WORLD);
        MPI_Finalize();
        return;
      }
    } while(tag != MPI_INVOKE_PROVER);

    MPI_Recv(&phase, 1, MPI_INT, MPI_ANY_SOURCE, MPI_PARAMS, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    find_cur_qlengths();
    cout << "p_current_memory_usage " << getCurrentRSS() << endl;
    cout << "p_peak_memory_usage " << getPeakRSS() << endl;
    handle_terminal_request();
    cout << "p_current_memory_usage " << getCurrentRSS() << endl;
    cout << "p_peak_memory_usage " << getPeakRSS() << endl;

    if (phase == PHASE_PROVER_COMMITMENT) {
#ifdef INTERFACE_MPI
      cout << "computation "<<m_computation.get_ru_elapsed_time()/num_local_runs<<endl;
      cout << "computation_latency "<<m_computation.get_papi_elapsed_time()/num_local_runs<<endl;

      cout << "computation_minimal "<<m_computation_minimal.get_ru_elapsed_time()/num_local_runs<<endl;
      cout << "computation_minimal_latency "<<m_computation_minimal.get_papi_elapsed_time()/num_local_runs<<endl;
      
      cout << "computation_sha "<<m_computation_sha.get_ru_elapsed_time()/num_local_runs<<endl;
      cout << "computation_sha_latency "<<m_computation_sha.get_papi_elapsed_time()/num_local_runs<<endl;
       
      cout << "p_interpret_constraints " <<   m_interpret_cons.get_ru_elapsed_time() << endl;
      cout << "p_interpret_constraints_latency " << m_interpret_cons.get_papi_elapsed_time() << endl;

      cout << "p_create_proofvector " << m_proofv_creation.get_ru_elapsed_time() <<endl;
      cout << "p_create_proofvector_latency " << m_proofv_creation.get_papi_elapsed_time() <<endl;

      cout << "p_commitment_answer " << m_answer_queries.get_ru_elapsed_time() << endl;
      cout << "p_commitment_answer_par " << m_answer_queries.get_papi_elapsed_time() << endl;
#else
      printf("computation %f\n", m_computation.get_ru_elapsed_time()/num_local_runs);
      printf("computation_latency %f\n", m_computation.get_papi_elapsed_time()/num_local_runs);

      printf("computation_minimal %f\n", m_computation_minimal.get_ru_elapsed_time()/num_local_runs);
      printf("computation_minimal_latency %f\n", m_computation_minimal.get_papi_elapsed_time()/num_local_runs);
      
      printf("p_interpret_constraints %f\n", m_interpret_cons.get_ru_elapsed_time());
      printf("p_interpret_constraints_latency %f\n", m_interpret_cons.get_papi_elapsed_time());

      printf("p_create_proofvector %f\n", m_proofv_creation.get_ru_elapsed_time());
      printf("p_create_proofvector_latency %f\n", m_proofv_creation.get_papi_elapsed_time());
      
      printf("p_commitment_answer %f\n", m_answer_queries.get_ru_elapsed_time());
      printf("p_commitment_answer_par %f\n", m_answer_queries.get_papi_elapsed_time());
#endif
    
      prover_total_cpu = m_interpret_cons.get_ru_elapsed_time() + m_proofv_creation.get_ru_elapsed_time() + m_answer_queries.get_ru_elapsed_time();
    
    } else if (phase == PHASE_PROVER_PCP) {
      prover_total_cpu += m_answer_queries.get_ru_elapsed_time(); 
#ifdef INTERFACE_MPI
      cout << "p_answer_plainq " <<  m_answer_queries.get_ru_elapsed_time() << endl;
      cout << "p_answer_plainq_latency " <<  m_answer_queries.get_papi_elapsed_time() << endl;
      cout << "p_total " <<  prover_total_cpu << endl;
#else
      printf("p_answer_plainq %f\n", m_answer_queries.get_ru_elapsed_time());
      printf("p_answer_plainq_latency %f\n", m_answer_queries.get_papi_elapsed_time());
      printf("p_total %f\n", prover_total_cpu);
#endif

    }

    if (phase != PHASE_PROVER_DEDUCE_QUERIES) {
      int status = 0;
      MPI_Send(&status, 1, MPI_INT, 0, MPI_PROVER_FINISHED, MPI_COMM_WORLD);
    }
  }
}

#else

void Prover::handle_http_requests() {
  while (FCGI_Accept() >= 0) {
    printf("Content-type: text/html\r\n" "\r\n");
    batch_start = -1;
    batch_end = -1;
    is_master = 0;
    parse_http_args(getenv("QUERY_STRING"), &phase, &batch_size,
                    &batch_start, &batch_end, &num_repetitions,
                    &m, &optimize_answers);
    find_cur_qlengths();
    handle_terminal_request();

    // print out the measurement stuff here.
    if (phase == PHASE_PROVER_COMMITMENT) {
      printf("computation %f\n", m_interpret_cons.get_ru_elapsed_time());
      printf("computation_latency %f\n", m_interpret_cons.get_papi_elapsed_time());
      printf("p_commitment_answer %f\n",
             m_answer_queries.get_ru_elapsed_time());
      printf("p_commitment_answer_par %f\n",
             m_answer_queries.get_papi_elapsed_time());
    } else {
      printf("p_answer_plainq %f\n", m_answer_queries.get_ru_elapsed_time());
      printf("p_answer_plainq_latency %f\n", m_answer_queries.get_papi_elapsed_time());
    }
  }
}

#endif
