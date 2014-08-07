#ifndef CODE_PEPPER_CMTGKR_BASE_CMT_PROTOCOL_CONFIG_H_
#define CODE_PEPPER_CMTGKR_BASE_CMT_PROTOCOL_CONFIG_H_

#include <vector>

#ifndef NDEBUG
  // Uncomment to disable all protocol checks. The prover will not attempt to
  // compute any sane responses. This is useful to quickly check for memory
  // usage.
  //#define CMTGKR_DISABLE_CHECKS 1
 
  // Uncomment to log the current layer that the verifier is on.
  // TODO: Implement proper logging levels.
  //#define LOG_LAYER 1
#endif

struct CMTProverConfig
{
  int startInstance;
  int endInstance;

  CMTProverConfig(int firstInstance, int lastInstance)
    : startInstance(firstInstance), endInstance(lastInstance)
  { }

  int batchSize() const { return endInstance - startInstance; }
};

struct CMTVerifierConfig
{
  int batchSize;

  explicit CMTVerifierConfig(int nBatch)
    : batchSize(nBatch)
  { }
};

struct CMTProtocolConfig
{
  CMTVerifierConfig vConf;
  std::vector<CMTProverConfig> pConfs;

  CMTProtocolConfig()
    : vConf(1), pConfs(1, CMTProverConfig(0, 1))
  { }

  CMTProtocolConfig(int nBatch, int nProvers)
    : vConf(nBatch), pConfs()
  {
    int numResponsible = nBatch / nProvers;
    int extra = nBatch - numResponsible * nProvers;

    for (int pid = 0; pid < nProvers; pid++)
    {
      int start = pid * numResponsible + std::min(pid, extra);
      int end = start + numResponsible + (pid < extra ? 1 : 0);

      pConfs.push_back(CMTProverConfig(start, end));
    }
  }

  int batchSize() const { return vConf.batchSize; }
  size_t numProvers() const { return pConfs.size(); }
};

#endif

