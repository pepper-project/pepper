#include <mpi.h>
#include <argp.h>
#include <cassert>

#include <common/mpnclass.h>

#include <crypto/prng.h>
#include <net/mpi_client.h>

#include "../circuit/pws_circuit.h"
#include "../base/cmt_circuit_prover.h"
#include "../base/cmt_circuit_verifier.h"

class CmtPwsNetRunner
{
  static const argp_option options[];
  static const std::string argDocs;
  static const std::string doc;
  static const argp argpOptions;

  static const int SERIALIZATION_SIGNAL_TAG = 123;

  std::string pwsFile;

  size_t batchSize;
  size_t numBitsNum;
  size_t numBitsDen;
  size_t primeSize;

  MPZClass prime;

  int pid;
  int size;

public:
#if (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L)
  typedef std::function<void(MPQVector&, size_t, size_t)> InputCreator;
#else
  typedef void (*InputCreator)(MPQVector&, size_t, size_t);
#endif

  CmtPwsNetRunner(int argc, char *argv[])
    : pwsFile(), batchSize(1), numBitsNum(32), numBitsDen(1), primeSize(128)
  {
    argp_parse(&argpOptions, argc, argv, 0, 0, this);
    Circuit::loadPrime(prime, primeSize);

    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
  }

  bool run(InputCreator inpCreator)
  {
    const int verifierPid = 0;
    bool isVerifier = pid == verifierPid;

    if (size < 2)
    {
      std::cerr << "Error: CMT requires at least two processes." << std::endl;
      return false;
    }

    assert(verifierPid == 0);

    PWSCircuitParser pwsParser(prime);
    pwsParser.parse(pwsFile);

    PWSCircuitBuilder builder(pwsParser);

    if (isVerifier)
    {
      CMTCircuit *c = builder.buildCircuit();
      MPQVector in(c->getInputSize() * batchSize);
      MPQVector out(c->getOutputSize() * batchSize);

      std::vector<NetClient*> nets;
      for (int i = 0; i < size; i++)
      {
        if (i != verifierPid)
          nets.push_back(new MPIClient(i));
      }

      CMTCircuitVerifier verifier(nets, *c);
      inpCreator(in, numBitsNum, numBitsDen);
      bool success = verifier.compute(out, in, batchSize);

      builder.destroyCircuit(c);

      typedef std::vector<NetClient*>::iterator NetIt;
      for (NetIt it = nets.begin(); it != nets.end(); ++it)
        delete *it;

      waitMyTurn();
      verifier.reportStats();
      doneWithTurn();

      return success;
    }
    else
    {
      MPIClient net(verifierPid);
      CMTCircuitProver prover(net, builder);
      prover.run();

      waitMyTurn();
      prover.reportStats();
      doneWithTurn();

      return true;
    }
  }

private:
  void waitMyTurn() const
  {
    MPI_Barrier(MPI_COMM_WORLD);
    waitForSignal();
  }

  void waitForSignal() const
  {
    if (pid == 0)
      return;

    int recv;
    MPI_Recv(&recv, 1, MPI_INT, pid - 1, SERIALIZATION_SIGNAL_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    assert(recv == pid);
  }

  void doneWithTurn() const
  {
    if (pid + 1 < size)
    {
      int send = pid + 1;
      MPI_Send(&send, 1, MPI_INT, pid+1, SERIALIZATION_SIGNAL_TAG, MPI_COMM_WORLD);
    }
  }

  static error_t
  parseOpt(int key, char *arg, struct argp_state *state)
  {
    CmtPwsNetRunner *config = reinterpret_cast<CmtPwsNetRunner*>(state->input);

    // I HATE THIS. There should be a C++ version with less typing...
    switch (key)
    {
    case 'n':
      config->numBitsNum = atoi(arg);
      break;
    case 'd':
      config->numBitsDen = atoi(arg);
      break;
    case 'b':
      config->batchSize = atoi(arg);
      break;
    case 'p':
      config->primeSize = atoi(arg);
      break;
    case 'h':
      argp_usage(state);
      break;
    case ARGP_KEY_ARG:
      if (state->arg_num >= 1)
        argp_usage(state);
      config->pwsFile = arg;
      break;
    case ARGP_KEY_END:
      if (state->arg_num < 1)
        argp_usage(state);
      break;
    default:
      return ARGP_ERR_UNKNOWN;
    }

    return 0;
  }

};

// Fields in an argp_option:
// { "long name", 'short name', "arg name", flags, group }
const argp_option CmtPwsNetRunner::options[] = {
  {"na", 'n', "nbits", 0, "The number of bits in the numerator. Default: 32." },
  {"nb", 'd', "nbits", 0, "The number of bits in the denominator. Default: 1." },
  {"nbatch", 'b', "BATCH_SIZE", 0, "The number of parallel instances to run. Default: 1." },
  {"prime-size", 'p', "nbits", 0, "The size of prime to use. Default: 128"},
  { 0 }
};

const std::string CmtPwsNetRunner::argDocs = "PWS_FILE";
const std::string CmtPwsNetRunner::doc = "A Cmt app that constructs a circuit from a PWS file.";

const argp CmtPwsNetRunner::argpOptions = { options, CmtPwsNetRunner::parseOpt, argDocs.c_str(), doc.c_str() };

namespace {

void
genRandomInputs(MPQVector& in, size_t na, size_t nb)
{
  std::cout << "na: " << na << ", nb: " << nb << std::endl;
  Prng prng(PNG_CHACHA);
  for (size_t i = 0; i < in.size(); i++)
  {
    prng.get_randomb(mpq_numref(in[i]), na);

    if (rand() % 2 == 0)
      mpz_neg(mpq_numref(in[i]), mpq_numref(in[i]));

    mpz_set_ui(mpq_denref(in[i]), 1);
    mpz_mul_2exp(mpq_denref(in[i]), mpq_denref(in[i]), rand() % (nb + 1));

    mpq_canonicalize(in[i]);
  }
}

}


int main(int argc, char** argv)
{
#ifndef INTERFACE_MPI
#error "The Mip netapp requires the use of MPI."
#endif

  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
  {
    std::cerr << "MPI_Init failed. Sorry." << std::endl;
    return 1;
  }

  CmtPwsNetRunner runner(argc, argv);
  bool success = runner.run(genRandomInputs);

  MPI_Finalize();

  return success ? 0 : 1;
}
