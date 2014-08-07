#ifndef CODE_PEPPER_CMTGKR_BASE_CMTSUMCHECK_V_H_
#define CODE_PEPPER_CMTGKR_BASE_CMTSUMCHECK_V_H_

#include <gmp.h>

#include <common/measurement.h>
#include <common/mpnvector.h>

#include <cmtgkr/circuit/circuit_layer.h>
#include <cmtgkr/circuit/circuit.h>

#include "cmt_verifier.h"
#include "cmt_protocol_config.h"

struct CMTSumCheckResults
{
  Measurement totalCheckTime;
  Measurement setupTime;

  bool success;

  MPZVector Vs;
  MPZVector rand;

  CMTSumCheckResults(size_t numRounds, int batchSize);
};

class CMTSumCheckVerifier : public CMTVerifier
{
  private:
  // Input state
  CircuitLayer& inLayer;
  CircuitLayer& outLayer;
  const MPZVector& z;
  const MPZVector& trueSums;
  mpz_t prime;

  // Temporary variables local to a sum-check instance.
  mpz_t tmp;

  public:
  CMTSumCheckVerifier(const ProverNets& provers, CMTProtocolConfig cc,
                      CircuitLayer& in, CircuitLayer& out,
                      const MPZVector& zi, const MPZVector& ri,
                      const mpz_t p);
  ~CMTSumCheckVerifier();

  const CMTSumCheckResults run();
  int numRounds() const;

  private:
  void requestPoly(MPZVector& poly, const mpz_t rand);
  void requestVs(MPZVector& Vs);

  bool doRound(MPZVector& newExpected,
               CMTSumCheckResults& results,
               int round, MPZVector& roundPoly,
               const MPZVector& expected);

  bool doFinalCheck(CMTSumCheckResults& results, const MPZVector& expected);
};

#endif

