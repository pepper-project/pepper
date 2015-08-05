#include <libv/verifier.h>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>
#include <storage/exo.h>
#include <common/utility.h>
#include <common/pairing_util.h>
#include <common/memory.h>

#ifdef USE_LIBSNARK
using namespace libsnark;
#endif

Verifier::Verifier(int batch, int reps, int ip_size, int opt_answers,
                   char *prover_url, const char *prover_name) {
  batch_size = batch;
  num_repetitions = reps;
  input_size = ip_size;
  optimize_answers = opt_answers;
#if NONINTERACTIVE == 0
  alloc_init_vec(&c_values, batch_size * num_repetitions);
#endif

  network_bytes_sent = 0;
  network_bytes_rcvd = 0;
  network_bytes_input_sent = 0;
  network_bytes_output_rcvd = 0;
  network_send_time_elapsed = 0;
  network_rcv_time_elapsed = 0;

  if (prover_url[0] != '\0')
    init_server_variables(prover_url, prover_name);

  alloc_init_scalar(a);
  alloc_init_scalar(f_s);
#ifdef INTERFACE_MPI
  mkdir(FOLDER_STATE, S_IRWXU);
  mkdir(FOLDER_PERSIST_STATE, S_IRWXU);
#endif
}

Verifier::~Verifier(void) {
#if NONINTERACTIVE == 0
  clear_vec(batch_size * num_repetitions, c_values);
  clear_vec(num_repetitions * num_lin_pcp_queries, f_con_coins);
#endif
  clear_scalar(a);
  clear_scalar(f_s);

  mpz_clear(prime);

  delete v;
  delete curl;
}

void Verifier::init_state() {
  mpz_init(prime);
#if NONINTERACTIVE == 1
#if PAIRING_LIB == PAIRING_PBC
  snprintf(scratch_str, BUFLEN - 1, "prime_f_%d.txt", num_bits_in_prime);
#else
  snprintf(scratch_str, BUFLEN - 1, "prime_libzm_%d.txt", num_bits_in_prime);
   #ifdef USE_LIBSNARK
  snprintf(scratch_str, BUFLEN - 1, "prime_libsnark_%d.txt", num_bits_in_prime);
    #endif
#endif
#else
  snprintf(scratch_str, BUFLEN - 1, "prime_%d.txt", num_bits_in_prime);
#endif
  load_txt_scalar(prime, scratch_str, const_cast<char *>("static_state"));

  if (crypto_in_use == CRYPTO_ELGAMAL)
    expansion_factor = 2;
  else
    expansion_factor = 1;

  v = new Venezia(ROLE_VERIFIER, crypto_in_use, png_in_use, num_bits_in_prime);
  curl = new CurlUtil();

#if NONINTERACTIVE == 0
  alloc_init_vec(&f_con_coins, num_repetitions * num_lin_pcp_queries);
#endif
}

void Verifier::init_server_variables(char *prover_url,
                                     const char *prover_name) {
  int remaining = strlen(SERVICE_UPLOAD_NAME) + 1;

  snprintf(prover_upload_url, BUFLEN-1, "%s%s", prover_url,
           SERVICE_UPLOAD_NAME);
  snprintf(prover_download_url, BUFLEN-1, "%s%s", prover_url,
           SERVICE_DOWNLOAD_NAME);
  snprintf(prover_query_url, BUFLEN-1, "%s%s", prover_url, prover_name);
}

void Verifier::send_file(char *file_name) {
  long long int file_size = 0;
  double time = 0;
  snprintf(full_file_name, BUFLEN - 1, "%s/%s", FOLDER_STATE, file_name);
  file_size = get_file_size(full_file_name);

#ifdef INTERFACE_MPI
  if (!FOLDER_STATE_SHARED) {
    Measurement file_transfer;

    char *buf = new char[file_size];
    FILE *fp = fopen(full_file_name, "r");
    if ((int)fread(buf, 1, file_size, fp) != file_size) {
      cerr <<"Error reading "<<full_file_name<<endl;
    }
    if (fp != NULL)
      fclose(fp);

    file_transfer.begin_with_init();
    MPI_Send(file_name, strlen(file_name)+1, MPI_CHAR, MPI_COORD_RANK,
             MPI_FILE_SEND, MPI_COMM_WORLD);
    MPI_Send(buf, file_size, MPI_BYTE, MPI_COORD_RANK, MPI_FILE_SEND,
             MPI_COMM_WORLD);
    file_transfer.end();
    time = file_transfer.get_papi_elapsed_time();

    delete[] buf;
  }
#else
  snprintf(full_file_name, BUFLEN - 1, "%s/%s", FOLDER_STATE, file_name);
  curl->send_file(full_file_name, prover_upload_url, &file_size, &time);
#endif

  network_bytes_sent += file_size;
  network_send_time_elapsed += time;
  if (strstr(file_name, INPUT_FILE_NAME_SUBSTR "_b") != NULL)
    network_bytes_input_sent += file_size;
}

void Verifier::recv_file(const char *file_name) {
  int size = 0;
  double time = 0;

#ifdef INTERFACE_MPI
  snprintf(full_file_name, BUFLEN - 1, "%s/%s", FOLDER_STATE, file_name);
  if (!FOLDER_STATE_SHARED) {
    Measurement get_time;
    MPI_Send(const_cast<char *>(file_name), strlen(file_name)+1, MPI_CHAR, 1,
             MPI_FILE_RECV, MPI_COMM_WORLD);

    MPI_Status stat;
    MPI_Probe(MPI_COORD_RANK, MPI_FILE_RECV, MPI_COMM_WORLD, &stat);
    MPI_Get_count(&stat, MPI_BYTE, &size);
    char *buf = new char[size];
    get_time.begin_with_init();
    MPI_Recv(buf, size, MPI_BYTE, MPI_COORD_RANK, MPI_FILE_RECV,
             MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    get_time.end();
    FILE *fp = fopen(full_file_name, "w");
    fwrite(buf, size, 1, fp);
    fclose(fp);
    time = get_time.get_papi_elapsed_time();
    delete[] buf;
  }
#else
  snprintf(full_file_name, BUFLEN - 1, "%s/%s", FOLDER_STATE, file_name);
  snprintf(download_url, BUFLEN - 1, "%s?file=%s", prover_download_url,
           file_name);
  curl->recv_file(full_file_name, download_url, &size, &time);

#endif

  size = get_file_size(full_file_name);
  network_bytes_rcvd += size;
  network_rcv_time_elapsed += time;
  if (strstr(file_name, OUTPUT_FILE_NAME_SUBSTR) != NULL)
    network_bytes_output_rcvd += size;
}

void Verifier::invoke_prover(int prover_phase) {
#ifdef INTERFACE_MPI

  char *tmp_str = (char *) "start";  // content of string doesnt matter.
  MPI_Send(tmp_str, strlen(tmp_str) + 1, MPI_CHAR, MPI_COORD_RANK,
           MPI_INVOKE_PROVER, MPI_COMM_WORLD);
  MPI_Send(&prover_phase, 1, MPI_INT, MPI_COORD_RANK, MPI_PARAMS,
           MPI_COMM_WORLD);

#else

  // construct the GET url with the parameters
  // phase=1&reps=1&batch_size=1&m=2
  snprintf(full_url, BUFLEN - 1,
           "%s?phase=%d&batch_size=%d&reps=%d&m=%d&opt=%d",
           prover_query_url, prover_phase, batch_size, num_repetitions,
           input_size, optimize_answers);
  // wait till the prover finishes its work
  curl->get(full_url);

#endif
}

void Verifier::create_commitment_query() {
  cout << "LOG: Verifier is creating commitment queries." << endl;

#if NONINTERACTIVE == 1
  create_noninteractive_query();
#else
  string commitment_str = "fX_commitment_query";
  vector < uint32_t >::const_iterator it = commitment_query_sizes.begin();
  for (uint32_t i = 0; i < commitment_query_sizes.size(); i++) {
    v->create_commitment_query(commitment_query_sizes[i],
                               f_commitment_ptrs[i], f_consistency_ptrs[i],
                               prime);

    commitment_str[1] = '0' + (i + 1);

    bool prover_using_gpu = true;
#if USE_GPU == 0
    prover_using_gpu = false;
#endif

    if (prover_using_gpu == false) {
      dump_vector_interleaved(expansion_factor * (commitment_query_sizes[i]),
                              f_commitment_ptrs[i],
                              const_cast<char *>(commitment_str.c_str()));
    } else {
      dump_vector(expansion_factor * (commitment_query_sizes[i]),
                  f_commitment_ptrs[i],
                  const_cast<char *>(commitment_str.c_str()));
    }
    send_file(const_cast<char *>(commitment_str.c_str()));
  }
#endif
}

void Verifier::recv_comm_answers() {
#if NONINTERACTIVE == 1
  for (int beta = 0; beta < batch_size; beta++) {
    string answer_str = "f_answers_G1_b_%d";
    snprintf(scratch_str, BUFLEN - 1, answer_str.c_str(), beta);
    recv_file(scratch_str);
    answer_str = "f_answers_G2_b_%d";
    snprintf(scratch_str, BUFLEN - 1, answer_str.c_str(), beta);
    recv_file(scratch_str);
  }
#else
  for (int beta = 0; beta < batch_size; beta++) {
    string commitment_answer_str = "f_commitment_answer_b_%d";
    snprintf(scratch_str, BUFLEN - 1, commitment_answer_str.c_str(), beta);
    recv_file(scratch_str);
  }
#endif
}

void Verifier::recv_outputs() {
  for (int i=0; i<batch_size; i++) {
    snprintf(scratch_str, BUFLEN-1, "output_b_%d", i);
    recv_file(scratch_str);
  }
}

void Verifier::recv_plain_answers() {
  for (int beta = 0; beta < batch_size; beta++) {

    FILE *fp = NULL;

    // if optimized format is used, fetch everything at once
    if (optimize_answers) {
      char f_name[BUFLEN];
      snprintf(f_name, BUFLEN - 1, "answers_%d", beta + 1);
      recv_file(f_name);
      continue;
    }

    // else, fetch one by one
    string consistency_answer_str = "f_consistency_answer_b_%d";
    snprintf(scratch_str, BUFLEN - 1, consistency_answer_str.c_str(), beta);
    recv_file(scratch_str);
    for (int rho = 0; rho < num_repetitions; rho++) {
      // NOTE: assumes X is a single digit!
      string answer_str = "qX_qquery_answer_b_%d_r_%d";
      for (uint32_t i = 0; i < num_lin_pcp_queries; i++) {
        answer_str[1] = '0' + i;
        snprintf(scratch_str, BUFLEN - 1, answer_str.c_str(), beta,
                 rho);
        recv_file(scratch_str);
      }
    }

    if (fp != NULL)
      fclose(fp);
  }
}

void Verifier::prepare_answers(int beta) {
#if NONINTERACTIVE == 1
  prepare_noninteractive_answers(beta);
#else
#if VERBOSE == 1
  cout << endl << "LOG: Batch " << beta << endl;
#endif

  FILE *fp = NULL;
  if (optimize_answers) {
    char f_name[BUFLEN];
    snprintf(f_name, BUFLEN - 1, "answers_%d", beta + 1);
    snprintf(f_name, BUFLEN - 1, "%s/answers_%d",
             FOLDER_STATE, beta + 1);
    fp = fopen(f_name, "rb");
    if (fp == NULL) {
      printf("Failed to open %s file for reading.", f_name);
      exit(1);
    }
  }

  // only one commitment answer and one consistency answer
  string commitment_answer_str = "f_commitment_answer_b_%d";
  snprintf(scratch_str, BUFLEN - 1, commitment_answer_str.c_str(), beta);
  load_vector(expansion_factor, temp_arr_ptrs[0], scratch_str);

  if (!optimize_answers) {
    string consistency_answer_str = "f_consistency_answer_b_%d";
    snprintf(scratch_str, BUFLEN - 1, consistency_answer_str.c_str(), beta);
    load_scalar(a, scratch_str);
  } else {
    size_t bytes = mpz_inp_raw(a, fp);
    fseek(fp, bytes, SEEK_CUR);
  }

  for (uint32_t j=0; j<num_repetitions*num_lin_pcp_queries; j++)
    mpz_set_ui(f_answers[j], 0);

  std::list<string> files = get_files_in_dir((char *)FOLDER_STATE);
  for (int rho = 0; rho < num_repetitions; rho++) {
    if (!optimize_answers) {
      char *file_exp = (char *)"_qquery_answer_b_%d_r_%d";
      snprintf(scratch_str, BUFLEN-1, file_exp, beta, rho);
      std::list<string>::const_iterator it = files.begin();

      for (; it != files.end(); it++) {
        string file_name = *it;
        size_t b_index = file_name.find(scratch_str);
        if (b_index != string::npos) {
          int q_num = atoi(file_name.substr(file_name.find("q") + 1, b_index).c_str());
          load_scalar(f_answers[rho * num_lin_pcp_queries + Q_list[q_num - 1]],
                      (char*)file_name.c_str());
        }
      }
    } else {
      for (uint32_t q = 0; q < Q_list.size(); q++) {
        size_t bytes =
          mpz_inp_raw(f_answers[rho * num_lin_pcp_queries + Q_list[q]], fp);
        fseek(fp, bytes, SEEK_CUR);
      }
    }
    //populate_answers(f_answers, rho, num_repetitions, beta);
  }

  snprintf(scratch_str, BUFLEN-1, "input_b_%d", beta);
  load_vector(size_input, input, scratch_str);

  snprintf(scratch_str, BUFLEN-1, "output_b_%d", beta);
  load_vector(size_output, output, scratch_str);

  if (output_q != NULL) {
    snprintf(scratch_str, BUFLEN-1, "output_q_b_%d", beta);
    load_vector(size_output, output_q, scratch_str);
    if (verify_conversion_to_z(size_output, output, output_q, prime) == false) {
      cout<<"LOG: Prover presented two different versions of output"<<endl;
      exit(1);
    } 
  }

  if (fp != NULL)
    fclose(fp);
#endif
}

bool Verifier::run_tests(int beta) {
  bool result = true;
#if !(NONINTERACTIVE == 1)
  // consistency test
  if (crypto_in_use == CRYPTO_ELGAMAL) {
    v->elgamal_dec(f_s, temp_arr_ptrs[0][0], temp_arr_ptrs[0][1]);
  }

  result =  v-> consistency_test(num_repetitions * num_lin_pcp_queries, a, f_s,
      f_answers, f_con_coins, prime);

#if VERBOSE == 1
  if (false == result)
    cout << "LOG: F failed the consistency test" << endl;
  else
    cout << "LOG: F passed the consistency test" << endl;
#endif
#endif
  return result & run_correction_and_circuit_tests(beta) ;
}

void Verifier::begin_pepper() {
#ifdef INTERFACE_MPI
  MPI_Send(&batch_size, 1, MPI_INT, MPI_COORD_RANK, MPI_PARAMS,
           MPI_COMM_WORLD);
  MPI_Send(&num_repetitions, 1, MPI_INT, MPI_COORD_RANK, MPI_PARAMS,
           MPI_COMM_WORLD);
  MPI_Send(&input_size, 1, MPI_INT, MPI_COORD_RANK, MPI_PARAMS,
           MPI_COMM_WORLD);
  MPI_Send(&optimize_answers, 1, MPI_INT, MPI_COORD_RANK, MPI_PARAMS,
           MPI_COMM_WORLD);
#endif

  cout << "batch_size " << batch_size << endl;
  cout << "num_reps " << num_repetitions << endl;
  cout << "input_size " << input_size << endl;
  cout << "output_size " << size_output << endl;
  cout << "num_bits_in_input " << num_bits_in_input << endl;
  cout << "num_bits_in_prime " << num_bits_in_prime << endl;

  double v_setup_total = 0;
  Measurement m, m_prover;
  int status;

  m.begin_with_init();
  create_commitment_query();
  m.end();
  cout << "v_commitmentq_create " << m.get_ru_elapsed_time() << endl;
  cout << "v_commitmentq_create_latency " << m.get_papi_elapsed_time() << endl;
  v_setup_total += m.get_ru_elapsed_time();

  
  m.begin_with_init();
  create_input();
  m.end();
  cout << "v_input_create " << m.get_ru_elapsed_time() << endl;
  cout << "v_input_create_latency " << m.get_papi_elapsed_time() << endl;

  m_prover.begin_with_init();
  // send the computation and input
  invoke_prover(PHASE_PROVER_COMMITMENT);
#ifdef INTERFACE_MPI
  MPI_Recv(&status, 1, MPI_INT, 1, MPI_PROVER_FINISHED, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
#endif
  m_prover.end();

  // receive output
  recv_outputs();
  // receive proof
  recv_comm_answers();

#if !(NONINTERACTIVE == 1)
  v->dump_seed_decommit_queries();
  invoke_prover(PHASE_PROVER_DEDUCE_QUERIES);
  create_plain_queries();
#endif

  cout << "v_plainq_create " << m_plainq.get_ru_elapsed_time() << endl;
  cout << "v_plainq_create_latency " << m_plainq.get_papi_elapsed_time() << endl;
  v_setup_total += m_plainq.get_ru_elapsed_time();

#if !(NONINTERACTIVE == 1)
  m_prover.begin_with_history();
  invoke_prover(PHASE_PROVER_PCP);
#ifdef INTERFACE_MPI
  MPI_Recv(&status, 1, MPI_INT, 1, MPI_PROVER_FINISHED, MPI_COMM_WORLD,
           MPI_STATUS_IGNORE);
#endif
  m_prover.end();

  recv_plain_answers();
#endif

  if (num_verification_runs == 0)
    num_verification_runs = 1;

#ifndef USE_LIBSNARK
  // verify the computation is done correctly
  bool result = true;
  // prepare for the verification by loading VK.
  for (int beta=0; beta<batch_size; beta++) {
    prepare_answers(beta);
    m_runtests.begin_with_history(); // part of it is in the prev function call
    for (int i=0; i<num_verification_runs; i++) {
      result = run_tests(beta) & result;
    }
    m_runtests.end();
  }
#endif

#ifdef USE_LIBSNARK

   cout << "v_current_mem_before_ls_verifier " << getCurrentRSS() << endl;
   cout << "v_peak_mem_before_ls_verifier " << getPeakRSS() << endl;
   m_runtests.begin_with_init();
   typedef Fr<bn128_pp> FieldT;
   string filename = FOLDER_STATE;
   filename += "/libsnark_vk";
   ifstream vkey(filename);

   r1cs_ppzksnark_processed_verification_key<bn128_pp> pvk;
   vkey >> pvk;
   
   r1cs_variable_assignment<Fr<bn128_pp> > inputvec;
   r1cs_ppzksnark_proof<bn128_pp> proof;

   recv_file((char *) "libsnark_proof");
   filename = FOLDER_STATE;
   filename +="/libsnark_proof";
   ifstream proof_file(filename);

   proof_file >> proof;
   proof_file.close();

   ifstream inputs("./bin/inputs");
   inputs >> noskipws;
   char c;

   int numInputs;
   inputs >> numInputs >> c;

   for (int j = 1; j <= numInputs; j++)
   {
       FieldT currentVar;
       inputs >> currentVar;
       inputvec.push_back(currentVar);
       inputs >> c;
   }
   inputs.close();   

   ifstream outputs("./bin/outputs");
   outputs >> noskipws;
 
   int numOutputs;
   outputs >> numOutputs >> c;

   for (int j = 1; j <= numOutputs; j++)
   {
       FieldT currentVar;
       outputs >> currentVar;
       inputvec.push_back(currentVar);
       outputs >> c;
   }
   outputs.close();   
  
   bool ans = true;
   m_runtests.end();
   cout <<"v_load_proof " << m_runtests.get_papi_elapsed_time() << endl;

   m_runtests.begin_with_init();
   num_verification_runs = 1; //otherwise libsnark prints a bunch of output
   for (int i = 0; i < num_verification_runs; i++)
     ans = ans & r1cs_ppzksnark_online_verifier_strong_IC<bn128_pp>(pvk, inputvec, proof);

   bool  result = ans; 

   m_runtests.end();
   cout << "v_current_mem_before_ls_verifier " << getCurrentRSS() << endl;
   cout << "v_peak_mem_after_ls_verifier " << getPeakRSS() << endl;
#endif
     
  if (false == result)
    cout <<endl<<"LOG: The prover failed one of the tests; set VERBOSE to 1 to find the test that failed"<<endl<<endl;
  else
    cout <<endl<<"LOG: The prover passed the decommitment test and PCP tests"<<endl<<endl;
  
  // output measurements.
  cout << "num_verification_runs: " << num_verification_runs << endl;
  cout << "v_setup_total " << v_setup_total << endl;
  cout << "v_run_pcp_tests " << m_runtests.get_ru_elapsed_time()/num_verification_runs << endl;
  cout << "v_run_pcp_tests_latency " << m_runtests.get_papi_elapsed_time()/num_verification_runs << endl;
  cout << "v_run_pcp_tests_rclock " << m_runtests.get_rclock_elapsed_time()/num_verification_runs << endl;

  cout << "v_net_bytes_sent " << network_bytes_sent << endl;
  cout << "v_net_bytes_rcvd " << network_bytes_rcvd << endl;
  cout << "v_net_send_time_elapsed " << network_send_time_elapsed << endl;
  cout << "v_net_rcv_time_elapsed " << network_rcv_time_elapsed << endl;
  cout << "v_net_bytes_input_sent " << network_bytes_input_sent << endl;
  cout << "v_net_bytes_output_rcvd " << network_bytes_output_rcvd << endl;

  cout << "p_d_latency " << m_prover.get_papi_elapsed_time() << endl;

#ifdef INTERFACE_MPI
  char *tmp_str = (char *)"terminate";  // content of string doesnt matter.
  MPI_Send(tmp_str, strlen(tmp_str) + 1, MPI_CHAR, MPI_COORD_RANK,
           MPI_TERMINATE, MPI_COMM_WORLD);
  MPI_Finalize();
#endif
}

