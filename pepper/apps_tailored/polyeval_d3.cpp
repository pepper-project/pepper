#include <apps_tailored/polyeval_d3_v.h>
#include <apps_tailored/polyeval_d3_p.h>
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
    PolyEvalD3Verifier verifier(batch_size, num_repetitions, input_size,
                                optimize_answers, prover_url);
    verifier.begin_pepper();
  } else {  // prover
    PolyEvalD3Prover prover(0 /*phase*/, batch_size, num_repetitions, input_size);
    prover.handle_requests();
  }

#else

  if (actor == 'v') {
    PolyEvalD3Verifier verifier(batch_size, num_repetitions, input_size,
                                optimize_answers, prover_url);

    verifier.begin_pepper();
  } else {
    if (argc > 2) {
      PolyEvalD3Prover prover(phase, batch_size, num_repetitions, input_size);
      prover.handle_terminal_request();
    } else {
      phase = 0;
      batch_size = 1;
      num_repetitions = 70;
      input_size = 200;

      PolyEvalD3Prover prover(phase, batch_size, num_repetitions, input_size);
      prover.handle_http_requests();
    }
  }
#endif

  return 0;
}
