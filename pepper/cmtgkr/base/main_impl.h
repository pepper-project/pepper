#ifndef CODE_PEPPER_CMTGKR_BASE_MAIN_IMPL_H_
#define CODE_PEPPER_CMTGKR_BASE_MAIN_IMPL_H_

#include <iostream>
#include <mpi.h>
 
#include <net/net_client.h>
#include <net/mpi_client.h>

#include "../base/cmt_circuit_verifier.h"
#include "../base/cmt_circuit_prover.h"

#include "../circuit/pws_circuit.h"

template<class C> std::pair<C*, int>
construct_circuit_switch(int argc, char** argv, char* name)
{
  if(argc < 2)
  {
    std::cout << "argc < 2. Command line arg should be " << argv[0] << " log(universe size) " << "<protocol>." << endl;
    std::cout << "\tprotocol is: (0) for CMT-original and (1) for CMT++." << endl;
    return std::pair<C*, int>();
  }

  int d = atoi(argv[0]);
  bool protocol = atoi(argv[1]) != 0;

  C* circuit = new C(d, protocol);
  return std::pair<C*, int>(circuit, 2);
}

void
gen_random_inputs(MPQVector& in, int na, int nb)
{
  cout << "na: " << na << ", nb: " << nb << endl;
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

template<int NA, int NB> int
create_random_inputs_static(MPQVector& in, int argc, char** argv, char* name)
{
  gen_random_inputs(in, NA, NB);
  return 0;
}

int
create_random_inputs_dynamic(MPQVector& in, int argc, char** argv, char* name)
{
  int na = 32;
  int nb = 1;

  int pop = 0;
  if (argc > 0)
  {
    na = atoi(argv[0]);
    pop = 1;
  }

  if (argc > 1)
  {
    nb = atoi(argv[1]);
    pop = 2;
  }

  gen_random_inputs(in, na, nb);
  return pop;
}

static int
getBatchSize(int *batchSize, int argc, char** argv, char* name)
{
  if (argc > 0)
  {
    *batchSize = atoi(argv[0]);
    return 1;
  }
  else
  {
    *batchSize = 1;
    return 0;
  }
}

template<class C> int
run_gkr_netapp(int argc, char** argv,
               std::pair<C*, int> (*builder_factory)(int argc, char** argv, char* name),
               int (*input_creator)(MPQVector& in, int argc, char** argv, char* name) = create_random_inputs_dynamic)
{
  std::vector<NetClient*> nets;
  bool isVerifier;

  int begin = 1;
  std::pair<C*, int> builder = builder_factory(argc - begin, argv + begin, argv[0]);

  begin += builder.second;

  int batchSize;
  begin += getBatchSize(&batchSize, argc - begin, argv + begin, argv[0]);

  if (!builder.first)
    return 1;

#ifdef INTERFACE_MPI
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
  {
    std::cerr << "MPI_Init failed. Sorry." << std::endl;
    return 1;
  }

  const int verifierRank = 0;
  int rank;
  int size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  isVerifier = (rank == verifierRank);

  if (isVerifier)
  {
    for (int i = 0; i < size; i++)
    {
      if (i != verifierRank)
        nets.push_back(new MPIClient(i));
    }
  }
  else
  {
    if (batchSize >= rank)
      nets.push_back(new MPIClient(verifierRank));
    else
      return 0;
  }
#endif

  bool error = 0;
  /*
  if (nets.empty())
  {
    std::cerr << "Fatal: Could not initialize a network object." << std::endl;
    std::cerr << "Fatal: Maybe there's no work to be done?" << std::endl;
    error = 1;
  }
  else
  */
  {
    if (isVerifier)
    {
      CMTCircuit *c = builder.first->buildCircuit();

      CMTCircuitVerifier verifier(nets, *c);
      MPQVector in(verifier.getInputSize() * batchSize);
      MPQVector out(verifier.getOutputSize() * batchSize);

      if (input_creator)
        input_creator(in, argc - begin, argv + begin, argv[0]);
      else
        create_random_inputs_static<32, 32>(in, argc - begin, argv + begin, argv[0]);

      verifier.compute(out, in, batchSize);

      builder.first->destroyCircuit(c);
    }
    else
    {
      CMTCircuitProver prover(*nets[0], *builder.first);
      prover.run();
    }

    delete builder.first;

    for (std::vector<NetClient*>::iterator it = nets.begin(); it != nets.end(); ++it)
      delete *it;
  }

#ifdef INTERFACE_MPI
  MPI_Finalize();
#endif

  return error;
}

#endif
