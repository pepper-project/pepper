
#include "cmt_prover.h"

using namespace std;

CMTProver::
CMTProver(NetClient& netClient)
  : net(netClient), conf(0, 1)
{ }

CMTProver::
CMTProver(NetClient& netClient, CMTProverConfig cc)
  : net(netClient), conf(cc)
{ }

