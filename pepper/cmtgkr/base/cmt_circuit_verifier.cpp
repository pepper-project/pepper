
#include <sstream>

#include "cmtgkr/cmtgkr_env.h"

#include "cmt_comm_consts.h"
#include "cmt_protocol_config.h"
#include "cmt_circuit_verifier.h"

using namespace std;

CMTCircuitVerifier::
CMTCircuitVerifier(const ProverNets& provers, CMTCircuit& cc)
  : CMTVerifier(provers), c(cc)
{ }

const CMTCircuit& CMTCircuitVerifier::
getCircuit() const
{
  return c;
}

size_t CMTCircuitVerifier::
getInputSize() const
{
  return c.getInputSize();
}

size_t CMTCircuitVerifier::
getOutputSize() const
{
  return c.getOutputSize();
}

bool CMTCircuitVerifier::
compute(MPQVector& outputs, const MPQVector& inputs, int batchSize)
{
  if (batchSize < 1)
    return true;

  inputsStat.clear();
  setupTime.reset();
  totalCheckTime.reset();
  //innerCheckTime.reset();
  conf = CMTProtocolConfig(batchSize, nets.size());

  MPQVector pOutputs(getOutputSize() * conf.batchSize());
  MPQVector pMagics(c.getMagicSize() * conf.batchSize());

  cout << "LOG: [V] Initializing Circuit(s)..." << flush;
  for (size_t pid = 0; pid < nets.size(); pid++)
  {
    nets[pid]->sendData(CMTCommConsts::CONFIGURE,
                        (char *) &conf.pConfs[pid],
                        sizeof(conf.pConfs[pid]));
  }
  broadcastZVector(CMTCommConsts::BEGIN_CMT, &c.prime, 1);
  requestOutputs(pOutputs, pMagics, inputs);
  cout << "DONE" << endl;

  cout << "LOG: [V] Using a " << c.get_prime_nbits() << "-bit prime." << endl;
  cout << "LOG: [V] Initializing Protocol with " << conf.batchSize() << " instances on ";
  cout << conf.numProvers() << " provers..." << flush;
  MPZVector zi(getMaxSumCheckRounds());
  MPZVector ri(conf.batchSize());

  initializeProtocol(zi, ri, pOutputs);
  broadcastZVector(CMTCommConsts::NOTIFY_Z0, zi.raw_vec(), c[0].logSize());
  cout << "DONE" << endl;

  cout << "LOG: [V] Running CMT Protocol..." << flush;
  bool success = true;

  for (int i = 0; i < c.depth() - 1; i++)
  {
#ifdef LOG_LAYER
    if (i == 0)
      cout << endl;
    cout << "LOG: [V] Verifying layer " << i << "..." << endl;
#endif
    if (!(success = checkLevel(zi, ri, i)))
    {
      cout << "Failed at layer " << i << endl;
      break;
    }
  }

  success = success && checkFinal(zi, ri, inputs, pMagics);
  broadcastEmpty(CMTCommConsts::END_CMT);
  cout << "DONE" << endl;

  if (success)
  {
    outputs.copy(pOutputs);
    cout << "Verification completed successfully.";
  }
  else
  {
    cout << "Verification failed.";
  }

  cout << endl << endl;

  return success;
}

int CMTCircuitVerifier::
getMaxSumCheckRounds()
{
  int maxSumCheckRounds = 0;
  for (int i = 0; i < c.depth() - 1; i++)
  {
    int nSCRounds = c[i].logSize() + 2 * c[i+1].logSize();
    maxSumCheckRounds = max(maxSumCheckRounds, nSCRounds);
  }
  return maxSumCheckRounds;
}

void CMTCircuitVerifier::
requestOutputs(MPQVector& outputs, MPQVector& magics, const MPQVector& inputs)
{
  distribute(CMTCommConsts::REQ_OUTPUTS, inputs);
  inputsStat += getLatestStat();

  gather(outputs, CMTCommConsts::REQ_OUTPUTS);
  inputsStat += getLatestStat();

  gather(magics, CMTCommConsts::REQ_OUTPUTS);
  inputsStat += getLatestStat();
}

void CMTCircuitVerifier::
initializeProtocol(MPZVector& z0, MPZVector& r0, const MPQVector& outputs)
{
  MPZVector chis(c[0].size());
  MPZVector z(c[0].logSize());

  setupTime.begin_with_history();
  for (size_t i = 0; i < z.size(); i++)
    prng.get_random(z[i], c.prime);

  computeChiAll(chis, z, c.prime);
  setupTime.end();

  MPQVector inner(getOutputSize());
  totalCheckTime.begin_with_history();
  //lde1.begin_with_history();
  for (int b = 0; b < conf.batchSize(); b++)
  {
    inner.copy(outputs, b * inner.size(), inner.size());
    c.initializeOutputs(inner);

    evaluate_V_chis(r0[b], c[0], chis, c.prime);
  }
  //lde1.end();
  totalCheckTime.end();

  z0.copy(z);
}

static bool
check(const string name, const mpz_t expected, const mpz_t actual)
{
#ifndef CMTGKR_DISABLE_CHECKS
  bool success = mpz_cmp(expected, actual) == 0;
  if(!success)
  {
    gmp_printf("%s value unexpected.\n", name.c_str());
    gmp_printf("expected is: %Zd\n", expected);
    gmp_printf("%s is: %Zd\n", name.c_str(), actual);
    cout << endl;
  }
#else
  bool success = true;
#endif
  return success;
}

bool CMTCircuitVerifier::
checkLevel(MPZVector& zi, MPZVector& ri, int level)
{
  broadcastData(CMTCommConsts::BEGIN_LAYER, (char*) &level, sizeof(int));

  CMTSumCheckVerifier scv(nets, conf, c[level + 1], c[level], zi, ri, c.prime);
  const CMTSumCheckResults results = scv.run();
  bool success = results.success;

  setupTime += results.setupTime;
  totalCheckTime += results.totalCheckTime;
  //innerCheckTime += results.totalCheckTime;

  if (success)
  {
    int mi   = c[level].logSize();
    int mip1 = c[level+1].logSize();
    const int lpolySize = mip1 + 1;

    MPZVector lpoly(lpolySize * conf.batchSize());
    MPZVector vec(2);
    MPZVector rand(1);

    broadcastEmpty(CMTCommConsts::REQ_LPOLY);
    gather(lpoly, CMTCommConsts::REQ_LPOLY, lpolySize);

    totalCheckTime.begin_with_history();
    //innerCheckTime.begin_with_history();
    for (int b = 0; b < conf.batchSize(); b++)
    {
      for (size_t i = 0; i < 2; i++)
      {
        stringstream ss;
        ss << "[Instance " << b << "] [Lvl " << level << "] lpoly" << i;
        success = success && check(ss.str(), results.Vs[2 * b + i], lpoly[lpolySize * b + i]);
      }
    }
    totalCheckTime.end();
    //innerCheckTime.end();

    if (success)
    {
      MPZVector weights(lpolySize);

      setupTime.begin_with_history();
      prng.get_random(rand[0], c.prime);

      for(int i = 0; i < mip1; i++)
      {
        mpz_set(vec[0], results.rand[mi + i]);
        mpz_set(vec[1], results.rand[mi + mip1 + i]);
        extrap(zi[i], vec.raw_vec(), vec.size(), rand[0], c.prime);
      }

      bary_precompute_weights(weights, rand[0], c.prime);
      setupTime.end();

      totalCheckTime.begin_with_history();
      //innerCheckTime.begin_with_history();
      bary_extrap(ri, lpoly, weights, c.prime);
      //innerCheckTime.end();
      totalCheckTime.end();

      // Make sure we tell the prover about the next zi and ri values.
      broadcastZVector(CMTCommConsts::NOTIFY_LPOLY_EXTRAP_PT, rand);
    }
  }

  if (!success)
    broadcastEmpty(CMTCommConsts::END_LAYER);

  return success;
}

bool CMTCircuitVerifier::
checkFinal(const MPZVector& zi, const MPZVector& ri, const MPQVector& inputs, const MPQVector& magics)
{
  MPZVector fr(1);
  MPQVector innerInputs(c.getInputSize());
  MPQVector innerMagics(c.getMagicSize());

  CircuitLayer& inputLayer = c.getInputLayer();

  bool success = true;

  MPZVector chis(inputLayer.size());
  MPZVector z(inputLayer.logSize());
  z.copy(zi, 0, z.size());

  setupTime.begin_with_history();
  computeChiAll(chis, z, c.prime);
  setupTime.end();

  totalCheckTime.begin_with_history();
  //lde2.begin_with_history();
  for (int b = 0; b < conf.batchSize(); b++)
  {
    innerInputs.copy(inputs, b * innerInputs.size(), innerInputs.size());
    innerMagics.copy(magics, b * innerMagics.size(), innerMagics.size());
    c.initializeInputs(innerInputs, innerMagics);

    evaluate_V_chis(fr[0], inputLayer, chis, c.prime);

    stringstream ss;
    ss << "[Instance " << b << "] fr";
    success = check(ss.str(), ri[b], fr[0]);

    if (!success)
      break;
  }
  //lde2.end();
  totalCheckTime.end();

  return success;
}

void CMTCircuitVerifier::
reportStats()
{
  cout << "v_setup_costs " << setupTime.get_ru_elapsed_time() << endl;
  cout << "v_setup_costs_latency " << setupTime.get_papi_elapsed_time() << endl << endl;

  cout << "v_run_tests " << (totalCheckTime.get_ru_elapsed_time() / conf.batchSize()) << endl;
  cout << "v_run_tests_latency " << (totalCheckTime.get_papi_elapsed_time() / conf.batchSize()) << endl << endl;

  //cout << "v_inner_run_tests " << (innerCheckTime.get_ru_elapsed_time() / conf.batchSize()) << endl;
  //cout << "v_inner_run_tests_latency " << (innerCheckTime.get_papi_elapsed_time() / conf.batchSize()) << endl << endl;

  //cout << "lde1: " << lde1.get_ru_elapsed_time() << endl;
  //cout << "lde2: " << lde2.get_ru_elapsed_time() << endl;

  NetStat protocolStats;
  for (size_t pid = 0; pid < nets.size(); pid++)
    protocolStats += nets[pid]->getGlobalStat();

  protocolStats.print("v_net_") << endl;
  inputsStat.print("v_net_input_");
}

