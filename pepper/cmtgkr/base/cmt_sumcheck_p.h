#ifndef CODE_PEPPER_CMTGKR_BASE_CMTSUMCHECK_P_H_
#define CODE_PEPPER_CMTGKR_BASE_CMTSUMCHECK_P_H_

#include <common/mpnvector.h>
#include <common/measurement.h>

#include <cmtgkr/circuit/circuit.h>

#include "cmt_prover.h"
#include "cmt_protocol_config.h"

class CMTSumCheckProver : public CMTProver
{
  Measurement time;

  // Input state
  std::vector< pair<CircuitLayer*, CircuitLayer*> > ioLayers;

  const MPZVector& z;
  MPZVector trueSum;
  mpz_t prime;

  // Temporary variables local to a sum-check instance.
  mpz_t gateBetar, gateAddormultr, V2, partialBetar, tmp;
  MPZVector expected;
  MPZVector V1s;
  MPZVector* vals;
  MPZVector betar;
  MPZVector addormultr;

  public:
  CMTSumCheckProver(NetClient& net, CMTProverConfig cc,
                    std::vector< std::pair<CircuitLayer*, CircuitLayer*> >& layers,
                    const MPZVector& zi, const MPZVector& ri,
                    const mpz_t p);
  ~CMTSumCheckProver();

  private:
  CMTSumCheckProver(const CMTSumCheckProver& other);

  public:
  int numRounds();
  bool run(MPZVector& rand);

  const Measurement& getTime() const;

  private:
  void computeBetaZAll();

  CircuitLayer& inLayer(int batchNum = 0);
  CircuitLayer& outLayer(int batchNum = 0);

  void doRound(MPZVector& roundPoly, int round, const mpz_t r);
  void doRoundP(MPZVector& poly, int bitPos, const mpz_t z, const mpz_t r);
  void doRoundO(MPZVector& poly, int bitPos, const mpz_t z, const mpz_t r, const bool onO1);
};

#endif

