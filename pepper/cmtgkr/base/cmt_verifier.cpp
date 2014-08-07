#include "cmt_comm_consts.h"
#include "cmt_verifier.h"

#include <iostream>

using namespace std;

CMTVerifier::
CMTVerifier(const ProverNets& provers)
  : nets(provers), conf(1, nets.size()), prng(PNG_CHACHA)
{ }

CMTVerifier::
CMTVerifier(const ProverNets& provers, CMTProtocolConfig cc)
  : nets(provers), conf(cc), prng(PNG_CHACHA)
{ }

void CMTVerifier::
waitForResponse(const std::string& valName, int prover)
{
  if (nets.empty())
    return;

  const string expectedRespHeader = CMTCommConsts::response(valName);
  string actualRespHeader = "";
  size_t numValid = 0;

  do
  {
    actualRespHeader = nets[prover]->waitForMessage();
  }
  while (expectedRespHeader != actualRespHeader);
}

void CMTVerifier::
broadcastEmpty(const std::string& header)
{
  latestStat.clear();
  for (NetIterator it = nets.begin(); it != nets.end(); ++it)
  {
    (*it)->sendEmpty(header);
    latestStat += (*it)->getMessageStat();
  }
}

void CMTVerifier::
broadcastData(const std::string& header, const char* data, size_t size)
{
  latestStat.clear();
  for (NetIterator it = nets.begin(); it != nets.end(); ++it)
  {
    (*it)->sendData(header, data, size);
    latestStat += (*it)->getMessageStat();
  }
}

void CMTVerifier::
broadcastZVector(const std::string& header, const MPZVector& vec)
{
  latestStat.clear();
  for (NetIterator it = nets.begin(); it != nets.end(); ++it)
  {
    (*it)->sendZVector(header, vec);
    latestStat += (*it)->getMessageStat();
  }
}

void CMTVerifier::
broadcastZVector(const std::string& header, const mpz_t* vec, size_t size)
{
  latestStat.clear();
  for (NetIterator it = nets.begin(); it != nets.end(); ++it)
  {
    (*it)->sendZVector(header, vec, size);
    latestStat += (*it)->getMessageStat();
  }
}

void CMTVerifier::
distribute(const std::string& header, const MPQVector& vec, size_t instanceSize)
{
  latestStat.clear();
  for (size_t pid = 0; pid < conf.numProvers(); pid++)
  {
    NetClient* net = nets[pid];
    CMTProverConfig pConf = conf.pConfs[pid];

    size_t distroStart = pConf.startInstance * instanceSize;
    size_t distroSize = pConf.batchSize() * instanceSize;
    net->sendQVector(header, &vec[distroStart], distroSize);
    latestStat += net->getMessageStat();
  } 
}

void CMTVerifier::
gather(MPQVector& vec, const std::string& header, size_t instanceSize)
{
  latestStat.clear();

  if (instanceSize == 0)
    return;

  for (size_t pid = 0; pid < conf.numProvers(); pid++)
  {
    NetClient* net = nets[pid];
    CMTProverConfig pConf = conf.pConfs[pid];

    size_t gatherStart = pConf.startInstance * instanceSize;
    size_t gatherSize = pConf.batchSize() * instanceSize;

    if (gatherSize > 0)
    {
      waitForResponse(header, pid);
      net->getDataAsQVector(&vec[gatherStart], gatherSize);
      latestStat += net->getMessageStat();
    }
  }
}

void CMTVerifier::
gather(MPZVector& vec, const std::string& header, size_t instanceSize)
{
  latestStat.clear();

  if (instanceSize == 0)
    return;

  for (size_t pid = 0; pid < conf.numProvers(); pid++)
  {
    NetClient* net = nets[pid];
    CMTProverConfig pConf = conf.pConfs[pid];

    size_t gatherStart = pConf.startInstance * instanceSize;
    size_t gatherSize = pConf.batchSize() * instanceSize;

    if (gatherSize > 0)
    {
      waitForResponse(header, pid);
      net->getDataAsZVector(&vec[gatherStart], gatherSize);
      latestStat += net->getMessageStat();
    }
  }
}

NetStat CMTVerifier::
getLatestStat() const
{
  return latestStat;
}

