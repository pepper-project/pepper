
#include <iostream>
#include <sstream>

#include <common/math.h>
#include <cmtgkr/cmtgkr_env.h>

#include "cmt_comm_consts.h"
#include "cmt_sumcheck_p.h"

using namespace std;

static const int polySize = 3;

static void
init_vals_array(MPZVector& vals, CircuitLayer& inLayer)
{
    int num_effective = 1 << inLayer.logSize();
    for(int k = 0; k < num_effective; k++)
    {
       if(k < inLayer.size())
         mpz_set(vals[k], inLayer.gate(k).zValue());
       else
         mpz_set_ui(vals[k], 0);
    }
}

static void
init_vec_si(MPZVector& vec, long val)
{
  for (size_t i = 0; i < vec.size(); i++)
    mpz_set_si(vec[i], val);
}

CMTSumCheckProver::
CMTSumCheckProver(
    NetClient& net, CMTProverConfig cc,
    vector< pair<CircuitLayer*, CircuitLayer*> >& layers,
    const MPZVector& zi, const MPZVector& ri,
    const mpz_t p)
  : CMTProver(net, cc), ioLayers(layers), z(zi), trueSum(ri)
{
  mpz_init_set(prime,  p);

  mpz_init_set_ui(gateBetar, 0);
  mpz_init_set_ui(gateAddormultr, 0);
  mpz_init_set_ui(V2, 0);
  mpz_init_set_ui(partialBetar, 1);
  mpz_init(tmp);

  expected.resize(conf.batchSize());
  V1s.resize(conf.batchSize());
  betar.resize(outLayer().size());
  addormultr.resize(outLayer().size());

  vals = new MPZVector[conf.batchSize()];
  for (int b = 0; b < conf.batchSize(); b++)
    vals[b].resize(1 << inLayer().logSize());
}

CMTSumCheckProver::
~CMTSumCheckProver()
{
  mpz_clear(prime);

  mpz_clear(gateBetar);
  mpz_clear(gateAddormultr);
  mpz_clear(V2);
  mpz_clear(partialBetar);
  mpz_clear(tmp);

  delete[] vals;
}

const Measurement& CMTSumCheckProver::
getTime() const
{
  return time;
}

// Computes betar_z(i) for 1 <= i <= n
void CMTSumCheckProver::
computeBetaZAll()
{
  assert(betar.size() == (size_t) outLayer().size());
  computeChiAll(betar, z, prime);
}

int CMTSumCheckProver::
numRounds()
{
  return outLayer().logSize() + 2 * inLayer().logSize();
}

CircuitLayer& CMTSumCheckProver::
inLayer(int batchNum)
{
  return *ioLayers[batchNum].first;
}

CircuitLayer& CMTSumCheckProver::
outLayer(int batchNum)
{
  return *ioLayers[batchNum].second;
}

bool CMTSumCheckProver::
run(MPZVector& rand)
{
  MPZVector poly(polySize * conf.batchSize());

  int round = 0;
  expected.copy(trueSum);
  init_vec_si(addormultr, 1);

  time.reset();
  while (true)
  {
    string request = net.waitForMessage();

    if (request == CMTCommConsts::END_LAYER)
    {
      break;
    }
    else if (request == CMTCommConsts::REQ_SC_POLY)
    {
      if (round >= numRounds())
      {
        cerr << "Warning. Unexpected sumcheck round. Doing nothing. requested: " <<
                round << ", max: " << numRounds() << endl;
      }
      else
      {
        time.begin_with_history();
        net.getDataAsZVector(&rand[round], 1);
        doRound(poly, round, rand[round]);

        // Update the expected value for the next round.
        // This is so we don't have to compute poly[0].
        for (int b = 0; b < conf.batchSize(); b++)
          extrap(expected[b], &poly[b * polySize], polySize, rand[round], prime);

        time.end();

        net.sendZVector(CMTCommConsts::response(request), poly);
      }

      round++;
    }
    else if (request == CMTCommConsts::REQ_VS)
    {
      if (round < numRounds())
      {
        cerr << "Warning. V1 and V2 requested too early. Current round: " <<
                round << " total rounds: " << numRounds() << endl;
      }

      // DEBUG: 
      //evaluate_beta(tmp, z.raw_vec(), rand.raw_vec(), outLayer().logSize(), prime);
      //gmp_printf("beta: %Zd\n", betar[0]);
      //gmp_printf("beta2: %Zd\n", tmp);

      MPZVector Vs(2 * conf.batchSize());
      for (int b = 0; b < conf.batchSize(); b++)
      {
        mpz_set(Vs[2 * b], V1s[b]);
        mpz_set(Vs[2 * b + 1], vals[b][0]);
      }

      net.sendZVector(CMTCommConsts::response(request), Vs);

      // Implicit end of SC.
      break;
    }
  }

  return round >= numRounds();
}

void CMTSumCheckProver::
doRound(MPZVector& roundPoly, int round, const mpz_t r)
{
#ifndef CMTGKR_DISABLE_CHECKS
  for (size_t i = 0; i < roundPoly.size(); i++)
    mpz_set_ui(roundPoly[i], 0);

  const int mi   = outLayer().logSize();
  const int mip1 = inLayer().logSize();

  //cout << round << " ! " << mi << " ! " << mip1 << endl;
  if (round > mi + 2 * mip1)
    return;

  if (round == 0)
  {
    computeBetaZAll();
    mpz_set_ui(partialBetar, 1);
  }

  if (round == mi + mip1)
  {
    for (int b = 0; b < conf.batchSize(); b++)
      mpz_set(V1s[b], vals[b][0]);
  }

  if (round < mi)
    doRoundP(roundPoly, round, z[round], r);
  else if (round < mi + mip1)
    doRoundO(roundPoly, round - mi, z[round], r, true);
  else
    doRoundO(roundPoly, round - mi - mip1, z[round], r, false);

  // Infer the value of roundPoly[0] from the expected value of the previous
  // round and roundPoly[1]. This all works because the sumcheck protocol has
  // perfect completeness.
  for (int b = 0; b < conf.batchSize(); b++)
    modsub(roundPoly[b * polySize], expected[b], roundPoly[b * polySize + 1], prime);
#endif
}

static void
mulmle2_si(mpz_t rop, const mpz_t op1, long op2, int bit, const mpz_t prime)
{
  modmult_si(rop, op1, bit ? op2 : 1 - op2, prime);
}

static void
mulmle2(mpz_t rop, const mpz_t op1, const mpz_t op2, int bit, const mpz_t prime)
{
  mpz_t tmp;
  mpz_init(tmp);

  if (bit)
    mpz_set(tmp, op2);
  else
    one_sub(tmp, op2);

  modmult(rop, op1, tmp, prime);

  mpz_clear(tmp);
}

void CMTSumCheckProver::
doRoundP(MPZVector& roundPoly, int bitPos, const mpz_t z, const mpz_t r)
{
  int logn = outLayer().logSize();

  // A gate contributes to either roundPoly[0] or roundPoly[1], never both. We
  // compute this contribution now. This is a simplification of the contribution
  // for roundPoly[2] below.
  for (int gateIdx = 0; gateIdx < outLayer().size(); gateIdx++)
  {
    const int betarIdx = gateIdx >> bitPos;
    const bool bit = (gateIdx >> bitPos) & 1;

    // We don't compute roundPoly[0]
    if (!bit) continue;

    mpz_mul(tmp, addormultr[gateIdx], betar[betarIdx]);
    for (int b = 0; b < conf.batchSize(); b++)
    {
      const Gate& out = outLayer(b).gate(gateIdx);
      if (mpz_sgn(out.zValue()) != 0)
        addmodmult(roundPoly[b * polySize + bit], out.zValue(), tmp, prime);
    }
  }

  // Divide out the (1 - z) or z term from the betar to prepare to compute
  // betar at a non-boolean location. Since
  //    betar[2*i] / (1 - z) == betar[2*i + 1] / z
  // we only really need to use one.
  int nEffectiveBetar = ((outLayer().size() - 1) >> (bitPos + 1)) + 1;
  one_sub(tmp, z);
  mpz_mod(tmp, tmp, prime);

  // (1 - z) might equal 0, in which case we just recompute betar.
  if (mpz_sgn(tmp) != 0)
  {
    mpz_invert(tmp, tmp, prime);
    for (int i = 0; i < nEffectiveBetar; i++)
    {
      modmult(betar[i], betar[2 * i], tmp, prime);
    }
  }
  else
  {
    computeChiAll(betar, nEffectiveBetar, this->z, bitPos + 1, prime);
    for (int i = 0;  i < nEffectiveBetar; i++)
    {
      modmult(betar[i], betar[i], partialBetar, prime);
    }
  }

  for (int gateIdx = 0; gateIdx < outLayer().size(); gateIdx++)
  {
    const int betarIdx = gateIdx >> (bitPos + 1);
    const bool bit = (gateIdx >> bitPos) & 1;

    for (int m = 2; m < polySize; m++)
    {
      one_sub(tmp, z);
      mle_si(tmp, m, tmp, z, prime);
      modmult(gateBetar, betar[betarIdx], tmp, prime);

      //compute addormult for this gate for this round
      mulmle2_si(gateAddormultr, addormultr[gateIdx], m, bit, prime);

      mpz_mul(tmp, gateBetar, gateAddormultr);
      for (int b = 0; b < conf.batchSize(); b++)
      {
        const Gate& out = outLayer(b).gate(gateIdx);

        // Compute betar * pred * output_val
        if (mpz_sgn(out.zValue()) != 0)
          addmodmult(roundPoly[b * polySize + m], tmp, out.zValue(), prime);
      }
    }

    // Update the addormultr array.
    mulmle2(addormultr[gateIdx], addormultr[gateIdx], r, bit, prime);
  }

  // Update the betar array.
  // Compute (1 - z)(1 - r) + z * r
  one_sub(tmp, z);
  mle(tmp, r, tmp, z, prime);
  for (int i = 0;  i < nEffectiveBetar; i++)
  {
    modmult(betar[i], betar[i], tmp, prime);
  }
  modmult(partialBetar, partialBetar, tmp, prime);
}

void CMTSumCheckProver::
doRoundO(MPZVector& roundPoly, int bitPos, const mpz_t z, const mpz_t r, const bool onO1)
{
  if (bitPos == 0)
  {
    for (int b = 0; b < conf.batchSize(); b++)
      init_vals_array(vals[b], inLayer(b));
  }

  for (int gateIdx = 0; gateIdx < outLayer().size(); gateIdx++)
  {
    const GateWiring& reprGate = outLayer()[gateIdx];

    const int inputGateIdx = onO1 ? reprGate.in1 : reprGate.in2;
    const int base = (inputGateIdx >> bitPos) & (~1);
    const int bit  = (inputGateIdx >> bitPos) & 1;

    // A gate contributes to either roundPoly[0] or roundPoly[1], never both.
    // We compute this contribution now (and only for roundPoly[1]). This is a
    // simplification of the contribution for roundPoly[2] below.
    if (bit)
    {
      for (int b = 0; b < conf.batchSize(); b++)
      {
        mpz_t& variableV = onO1 ? V1s[b] : V2;
        mpz_t& op2V = onO1 ? inLayer(b).gate(reprGate.in2).zValue() : V2;

        mpz_set(variableV, vals[b][base + bit]);
        reprGate.applyGateOperation(tmp, V1s[b], op2V, prime);
        mpz_addmul(roundPoly[b * polySize + bit], tmp, addormultr[gateIdx]);
      }
    }

    for (int m = 2; m < polySize; m++)
    {
      //compute addormult for this gate for this variable.
      mulmle2_si(gateAddormultr, addormultr[gateIdx], m, bit, prime);

      for (int b = 0; b < conf.batchSize(); b++)
      {
        mpz_t& variableV = onO1 ? V1s[b] : V2;
        mpz_t& op2V = onO1 ? inLayer(b).gate(reprGate.in2).zValue() : V2;

        mle_si(variableV, m, vals[b][base], vals[b][base+1], prime);

        reprGate.applyGateOperation(tmp, V1s[b], op2V, prime);

        // Compute the contribution of this gate to the polynomial evaluation. We
        // delay multiplying in the betar term because it is constant.
        mpz_addmul(roundPoly[b * polySize + m], tmp, gateAddormultr);
      }
    }

    // Update the addormultr array.
    mulmle2(addormultr[gateIdx], addormultr[gateIdx], r, bit, prime);
  }

  // Perform the multiplication of betar that we delayed from above.
  for (int b = 0; b < conf.batchSize(); b++)
    for (int m = 1; m < 3; m++)
      modmult(roundPoly[b * polySize + m], roundPoly[b * polySize + m], betar[0], prime);

  // Update the vals array.
  int num_effective = 1 << (inLayer().logSize() - bitPos);
  for(int i = 0; i < (num_effective >> 1); i++)
  {
    one_sub(tmp, r);
    for (int b = 0; b < conf.batchSize(); b++)
    {
      // Do vals[i] = vals[2i] * (1 - r) + vals[2i + 1] * r.
      modmult(vals[b][i], vals[b][i << 1], tmp, prime);
      addmodmult(vals[b][i], vals[b][(i << 1) + 1], r, prime);
    }
  }
}

