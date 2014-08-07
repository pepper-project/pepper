
#include <cstdlib>
#include <iostream>
#include <mpi.h>
#include <gmp.h>

#include <net/mpi_client.h>

using namespace std;

static void
doA()
{
  cout << "[A] Initialized." << endl;

  NetClient* net = new MPIClient(0);
  mpz_t a;
  mpz_init_set_si(a, rand() + 213);

  cout << "[A] Send " << 1 << " mpz_t to A" << endl;
  gmp_printf("[A] Should expect: %Zd\n", a);
  net->sendZVector("Vec from A.", &a, 1);

  string header;
  header = net->waitForMessage();
  int valid = net->getDataAsZVector(&a, 1);
  cout << "[A] Recv " << valid << " mpz_t from B with the header: '" << header << "'" << endl;
  gmp_printf("[A] We got: %Zd\n", a);

  cout << "[A] Sending empty message." << endl;
  net->sendEmpty("Empty message from A.");

  header = "MSG 1";
  cout << "[A] Sending message: '" << header << "'." << endl;
  net->sendData(header, header.c_str(), header.length());

  header = "MSG 2";
  cout << "[A] Sending message: '" << header << "'." << endl;
  net->sendData(header, header.c_str(), header.length());

  cout << net->getGlobalStat() << endl;

  delete net;
}

static void
doB()
{
  cout << "[B] Initialized." << endl;

  NetClient* net = new MPIClient(1);
  mpz_t a;
  mpz_init_set_ui(a, 0);

  string header;
  header = net->waitForMessage();
  int valid = net->getDataAsZVector(&a, 1);
  cout << "[B] Recv " << valid << " mpz_t from A with the header: '" << header << "'" << endl;
  gmp_printf("[B] We got: %Zd\n", a);

  mpz_set_si(a, rand() + 2342);
  cout << "[B] Send " << 1 << " mpz_t to A." << endl;
  gmp_printf("[B] Should expect: %Zd\n", a);
  net->sendZVector("Vec from B.", &a, 1);

  header = net->waitForMessage();
  cout << "[B] Recv Empty from A." << endl;
  cout << "[B] H: " << header << endl;

  header = net->waitForMessage();
  cout << "[B] Recv MSG treated as empty from A." << endl;
  cout << "[B] H: " << header << endl;

  char* buf;
  header = net->waitForMessage();
  size_t len = net->getData(&buf);
  cout << "[B] Recv MSG from A." << endl;
  cout << "[B] H: " << header << endl;
  header.assign(buf, len);
  cout << "[B] len: " << len << " D: " << header << endl;

  cout << net->getGlobalStat() << endl;

  delete net;
}

int main(int argc, char** argv)
{
  int rank;
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS)
    cerr << "MPI_Init failed. Sorry." << endl;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank)
    doA();
  else
    doB();

  MPI_Finalize();
}

