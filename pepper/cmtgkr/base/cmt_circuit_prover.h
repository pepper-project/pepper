#ifndef CODE_PEPPER_CMTGKR_BASE_CMT_CIRCUIT_PROVER_H_
#define CODE_PEPPER_CMTGKR_BASE_CMT_CIRCUIT_PROVER_H_

#include <string>

#include <common/mpnvector.h>
#include <common/measurement.h>

#include "../circuit/cmt_circuit.h"

#include "cmt_prover.h"
#include "cmt_sumcheck_p.h"
#include "cmt_protocol_config.h"
#include "cmt_circuit_builder.h"

class CMTCircuitProver : public CMTProver
{
  protected:
  Measurement time;
  Measurement compTime;

  NetStat inputsStat;

  CMTCircuitBuilder& builder;

  // Protocol values
  MPZVector zi;
  MPZVector ri;

  std::vector<CMTCircuit*> batch;

  public:
  CMTCircuitProver(NetClient& netClient, CMTCircuitBuilder& bb);
  ~CMTCircuitProver();

  void run();
  void reportStats();

  private:
  void initProtocol(const mpz_t prime);
  void clearCircuits();

  CMTCircuit& circuit(int instance = 0);

  //void sendLayer(const std::string& header, const CircuitLayer& l, bool sendQs);
  //void recvLayer(CircuitLayer& l, bool recvQs);

  bool doRound(int level);
  bool doMiniIP(int level, MPZVector& rand);
};

#endif

