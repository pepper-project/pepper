#include <apps_tailored/hdist_v.h>
#include <apps_tailored/hdist_p.h>

bool MICROBENCHMARKS = 0;

// driver to run the phases of the verifier
int main(int argc, char **argv) {
  int batch_size;
  int num_repetitions;
  int input_size;
  char prover_url[BUFLEN];
  char actor;
  int phase;

  parse_args(argc, argv, &actor, &phase, &batch_size, &num_repetitions, &input_size,
             prover_url);
  int optimize_answers = 1;

#ifdef INTERFACE_MPI

  MPI_Init(&argc, &argv);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {  // verifier
    HDistVerifier verifier(batch_size, num_repetitions, input_size,
                           optimize_answers, prover_url);
    verifier.begin_pepper();
  } else {  // prover
    HDistProver prover(0 /*phase*/, batch_size, num_repetitions, input_size);
    prover.handle_requests();
  }

#else
  if (actor == 'v') {
    HDistVerifier verifier(batch_size, num_repetitions, input_size,
                           optimize_answers, prover_url);
    verifier.begin_pepper();
  } else {
    if (argc > 2) {
      HDistProver prover(phase, batch_size, num_repetitions, input_size);
      prover.handle_terminal_request();
    } else {
      phase = 0;
      batch_size = 100;
      num_repetitions = 1;
      input_size = 100; // for now, the length of the two strings

      HDistProver prover(phase, batch_size, num_repetitions, input_size);
      prover.handle_http_requests();
    }
  }
#endif
  return 0;
}
