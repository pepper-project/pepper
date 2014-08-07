
#include <common/math.h>
#include <cmtgkr/cmtgkr_env.h>

#include "cmt_comm_consts.h"
#include "cmt_sumcheck_v.h"

static const int polySize = 3;

CMTSumCheckResults::
CMTSumCheckResults(size_t numRounds, int batchSize)
  : success(false), Vs(2 * batchSize), rand(numRounds)
{ }

CMTSumCheckVerifier::
CMTSumCheckVerifier(
    const ProverNets& provers, CMTProtocolConfig cc,
    CircuitLayer& in, CircuitLayer& out,
    const MPZVector& zi, const MPZVector& ri,
    const mpz_t p)
  : CMTVerifier(provers, cc), inLayer(in), outLayer(out), z(zi), trueSums(ri)
{
  mpz_init_set(prime,  p);

  mpz_init(tmp);
}

CMTSumCheckVerifier::
~CMTSumCheckVerifier()
{
  mpz_clear(prime);

  mpz_clear(tmp);
}

int CMTSumCheckVerifier::
numRounds() const
{
  return outLayer.logSize() + 2 * inLayer.logSize();
}

const CMTSumCheckResults CMTSumCheckVerifier::
run()
{
  MPZVector expected(trueSums);
  MPZVector poly(conf.batchSize() * polySize);

  CMTSumCheckResults results(numRounds(), conf.batchSize());
  results.success = true;

  for (int i = 0; i < numRounds(); i++)
  {
    results.success = results.success &&
                      doRound(expected,
                              results,
                              i, poly, expected);
  }

  results.success = results.success &&
                    doFinalCheck(results, expected);

  return results;
}

void CMTSumCheckVerifier::
requestPoly(MPZVector& poly, const mpz_t rand)
{
  MPZVector r(1);
  r.set(0, rand);
  broadcastZVector(CMTCommConsts::REQ_SC_POLY, r);
  gather(poly, CMTCommConsts::REQ_SC_POLY);
}

void CMTSumCheckVerifier::
requestVs(MPZVector& Vs)
{
  broadcastEmpty(CMTCommConsts::REQ_VS);
  gather(Vs, CMTCommConsts::REQ_VS);
}

bool CMTSumCheckVerifier::
doRound(MPZVector& newExpected,
        CMTSumCheckResults& results,
        int round, MPZVector& roundPoly,
        const MPZVector& expected)
{
  mpz_t& newRand = results.rand[round];

  prng.get_random(newRand, prime);

  requestPoly(roundPoly, newRand);

  results.totalCheckTime.begin_with_history();
  bool success = true;
#ifndef CMTGKR_DISABLE_CHECKS
  for (int b = 0; b < conf.batchSize(); b++)
  {
    modadd(tmp, roundPoly[b * polySize], roundPoly[b * polySize + 1], prime);
    success = mpz_cmp(expected[b], tmp) == 0;
    if (!success)
    {
      cout << "[Instance " << b << "] [SumCheck] Round " << round << " sumcheck failed!" << endl;
      gmp_printf("poly[0][0]+poly[0][1] != ri\n");
      gmp_printf("poly[0] is: %Zd\n", roundPoly[b * polySize]);
      gmp_printf("poly[1] is: %Zd\n", roundPoly[b * polySize + 1]);
      gmp_printf("actual   is: %Zd\n", tmp);
      gmp_printf("expected is: %Zd\n", expected[b]);
      cout << endl;
      break;
    }
  }
#endif
  results.totalCheckTime.end();

  if (success)
  {
    MPZVector weights(polySize);

    results.setupTime.begin_with_history();
    bary_precompute_weights3(weights, newRand, prime);
    results.setupTime.end();

    results.totalCheckTime.begin_with_history();
    bary_extrap(newExpected, roundPoly, weights, prime);
    results.totalCheckTime.end();
  }

  return success;
}

bool CMTSumCheckVerifier::
doFinalCheck(CMTSumCheckResults& results, const MPZVector& expected)
{
  mpz_t add_predr, mul_predr, fz, betar;

  mpz_init(add_predr);
  mpz_init(mul_predr);
  mpz_init(betar);
  mpz_init(fz);

  requestVs(results.Vs);

  int mi = outLayer.logSize();
  int ni = outLayer.size();
  int mip1 = inLayer.logSize();
  int nip1 = inLayer.size();

  results.setupTime.begin_with_history();
  outLayer.computeWirePredicates(add_predr, mul_predr, results.rand, nip1, prime);
  //outLayer.add_fn(predr, results.rand.raw_vec(), mi, mip1, ni, nip1, prime);
  //outLayer.mul_fn(predr, results.rand.raw_vec(), mi, mip1, ni, nip1, prime);

  evaluate_beta(betar, z.raw_vec(), results.rand.raw_vec(), mi, prime);
  results.setupTime.end();

  // DEBUG
  //gmp_printf("[V] betar: %Zd\n", tmp);

  results.totalCheckTime.begin_with_history();
  bool success = true;
#ifndef CMTGKR_DISABLE_CHECKS
  for (int b = 0; b < conf.batchSize(); b++)
  {
    mpz_set_ui(fz, 0);

    mpz_add(tmp, results.Vs[2 * b], results.Vs[2 * b + 1]);
    mpz_addmul(fz, tmp, add_predr);

    mpz_mul(tmp, results.Vs[2 * b], results.Vs[2 * b + 1]);
    mpz_addmul(fz, tmp, mul_predr);

    modmult(fz, fz, betar, prime);

    success = mpz_cmp(expected[b], fz) == 0;
    if (!success)
    {
      cout << "[Instance " << b << "] [SumCheck] Final check failed!" << endl;
      gmp_printf("fz is: %Zd\n", fz);
      gmp_printf("expected is: %Zd\n", expected[b]);
      cout << endl;
      break;
    }
  }
#endif
  results.totalCheckTime.end();

  mpz_clear(add_predr);
  mpz_clear(mul_predr);
  mpz_clear(fz);

  return success;
}

