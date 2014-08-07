#ifndef CODE_PEPPER_CMTGKR_BASE_CMT_CIRCUIT_VERIFIER_H_
#define CODE_PEPPER_CMTGKR_BASE_CMT_CIRCUIT_VERIFIER_H_

#include <string>

#include "../circuit/cmt_circuit.h"

#include "cmt_verifier.h"
#include "cmt_sumcheck_v.h"
#include "cmt_protocol_config.h"

class CMTCircuitVerifier : public CMTVerifier
{
  protected:
  CMTCircuit& c;

  //Measurement lde1;
  //Measurement lde2;
  //Measurement innerCheckTime;
  Measurement setupTime;
  Measurement totalCheckTime;
  NetStat inputsStat;

  public:
  CMTCircuitVerifier(const ProverNets& provers, CMTCircuit& cc);

  const CMTCircuit& getCircuit() const;

  size_t getInputSize() const;
  size_t getOutputSize() const;

  bool compute(MPQVector& outputs, const MPQVector& inputs, int batchSize = 1);
  void reportStats();

  private:
  void requestOutputs(MPQVector& outputs, MPQVector& magics, const MPQVector& inputs);
  void initializeProtocol(MPZVector& z0, MPZVector& r0, const MPQVector& outputs);
  int getMaxSumCheckRounds();
  bool checkLevel(MPZVector& zi, MPZVector& ri, int level);
  bool checkFinal(const MPZVector& zi, const MPZVector& ri, const MPQVector& inputs, const MPQVector& magics);
};

#endif

