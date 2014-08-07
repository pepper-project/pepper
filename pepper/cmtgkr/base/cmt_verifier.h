#ifndef CODE_PEPPER_CMTGKR_BASE_CMT_VERIFIER_H_
#define CODE_PEPPER_CMTGKR_BASE_CMT_VERIFIER_H_

#include <string>

#include <crypto/prng.h>

#include <common/mpnvector.h>
#include <net/net_client.h>
#include "cmt_protocol_config.h"

class CMTVerifier
{
  protected:
  typedef std::vector<NetClient*> ProverNets;
  typedef ProverNets::const_iterator NetIterator;

  const ProverNets nets;
  CMTProtocolConfig conf;
  Prng prng;

  NetStat latestStat;

  public:
  explicit CMTVerifier(const ProverNets& provers);
  CMTVerifier(const ProverNets& provers, CMTProtocolConfig cc);

  protected:
  void waitForResponse(const std::string& valName, int prover);

  // FIXME: Use variadic templates if possible.
  void broadcastEmpty(const std::string& header);
  void broadcastData(const std::string& header, const char* data, size_t size);
  void broadcastZVector(const std::string& header, const MPZVector& vec);
  void broadcastZVector(const std::string& header, const mpz_t* vec, size_t size);

  void distribute(const std::string& header, const MPQVector& vec, size_t instanceSize);
  void gather(MPZVector& vec, const std::string& header, size_t instanceSize);
  void gather(MPQVector& vec, const std::string& header, size_t instanceSize);

  template<typename T> void distribute(const std::string& header, const MPNVector<T>& vec)
  {
    size_t instanceSize = vec.size() / conf.batchSize();
    distribute(header, vec, instanceSize);
  }

  template<typename T> void gather(MPNVector<T>& vec, const std::string& header)
  {
    size_t instanceSize = vec.size() / conf.batchSize();
    gather(vec, header, instanceSize);
  }

  /*
   * Returns the NetStat for the latest broadcast, distribute, or gather operation.
   */
  NetStat getLatestStat() const;
};

#endif

