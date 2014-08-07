
#include <string>

#include <cmtgkr/cmtgkr_env.h>

#include "cmt_comm_consts.h"
#include "cmt_circuit_prover.h"

using namespace std;

typedef vector<CMTCircuit*>::iterator CircuitIt;

CMTCircuitProver::
CMTCircuitProver(NetClient& netClient, CMTCircuitBuilder& bb)
  : CMTProver(netClient), builder(bb), zi(0)
{ }

CMTCircuitProver::
~CMTCircuitProver()
{
  clearCircuits();
}

void CMTCircuitProver::
run()
{
  inputsStat.clear();
  compTime.reset();
  time.reset();
  while (true)
  {
    // Wait for starting command.
    string header = net.waitForMessage();
    bool shouldExit = false;

    if (header == CMTCommConsts::END_CMT)
    {
      shouldExit = true;
    }
    else if (header == CMTCommConsts::CONFIGURE)
    {
      net.getData((char*) &conf, sizeof(conf));
    }
    else if (header == CMTCommConsts::BEGIN_CMT)
    {
      MPZVector prime(1);

      net.getDataAsZVector(prime);
      initProtocol(prime[0]);
    }
    else if (header == CMTCommConsts::REQ_OUTPUTS)
    {
      MPQVector inner(circuit().getInputSize());
      MPQVector vec(inner.size() * conf.batchSize());
      net.getDataAsQVector(vec);
      inputsStat = net.getMessageStat();

      for (int b = 0; b < conf.batchSize(); b++)
      {
        inner.copy(vec, b * inner.size(), inner.size());
        circuit(b).initializeInputs(inner);
      }

      compTime.begin_with_init();
      for (int b = 0; b < conf.batchSize(); b++)
        circuit(b).evaluate();
      compTime.end();

      //c.print();

      inner.resize(circuit().getOutputSize());
      vec.resize(inner.size() * conf.batchSize());
      for (int b = 0; b < conf.batchSize(); b++)
      {
        circuit(b).getOutputs(inner);
        vec.copy(inner, 0, inner.size(), b * inner.size());
      }
      net.sendQVector(CMTCommConsts::response(CMTCommConsts::REQ_OUTPUTS), vec);
      inputsStat += net.getMessageStat();

      inner.resize(circuit().getMagicSize());
      vec.resize(inner.size() * conf.batchSize());
      for (int b = 0; b < conf.batchSize(); b++)
      {
        circuit(b).getMagics(inner);
        vec.copy(inner, 0, inner.size(), b * inner.size());
      }
      net.sendQVector(CMTCommConsts::response(CMTCommConsts::REQ_OUTPUTS), vec);
      inputsStat += net.getMessageStat();
    }
    else if (header == CMTCommConsts::NOTIFY_Z0)
    {
      net.getDataAsZVector(zi.raw_vec(), circuit()[0].logSize());

      time.begin_with_history();
      for (int b = 0; b < conf.batchSize(); b++)
        evaluate_V_i(ri[b], circuit(b)[0], zi.raw_vec(), circuit(b).prime);
      time.end();
    }
    else if (header == CMTCommConsts::BEGIN_LAYER)
    {
      int round;
      net.getData((char*) &round, sizeof(int));
      //cout << "Begin layer: " << round << endl;
      shouldExit = !doRound(round);
      //cout << "End layer: " << round << " : " << time.get_ru_elapsed_time() << endl;
    }

    if (shouldExit)
      break;
  }
}

void CMTCircuitProver::
clearCircuits()
{
  for (CircuitIt it = batch.begin(); it != batch.end(); ++it)
    builder.destroyCircuit(*it);
  batch.clear();
}

CMTCircuit& CMTCircuitProver::
circuit(int instance)
{
  return *batch[instance];
}

void CMTCircuitProver::
initProtocol(const mpz_t prime)
{
  clearCircuits();

  for (int i = 0; i < conf.batchSize(); i++)
  {
    batch.push_back(builder.buildCircuit());
    mpz_set(batch.back()->prime, prime);
  }

  int maxSumCheckRounds = 0;
  for (int i = 0; i < circuit().depth() - 1; i++)
  {
    int nSCRounds = circuit()[i].logSize() + 2 * circuit()[i+1].logSize();
    maxSumCheckRounds = max(maxSumCheckRounds, nSCRounds);
  }

  zi.resize(maxSumCheckRounds);
  ri.resize(conf.batchSize());
}

bool CMTCircuitProver::
doRound(int level)
{
  if (level >= circuit().depth() - 1)
    return true;

  vector< pair<CircuitLayer*, CircuitLayer*> >layers;
  for (int b = 0; b < conf.batchSize(); b++)
  {
    CMTCircuit& c = circuit(b);
    layers.push_back(pair<CircuitLayer*, CircuitLayer*>(&c[level+1], &c[level]));
  }

  CMTSumCheckProver scp(net, conf, layers, zi, ri, circuit().prime);
  MPZVector rand(scp.numRounds());

  if (!scp.run(rand))
    return false;

  time += scp.getTime();

  return doMiniIP(level, rand);
}

bool CMTCircuitProver::
doMiniIP(int level, MPZVector& rand)
{
  const int mi   = circuit()[level].logSize();
  const int mip1 = circuit()[level+1].logSize();
  const size_t lpolySize = mip1 + 1;

  MPZVector lpoly(lpolySize * conf.batchSize());
  MPZVector point(mip1);
  MPZVector vec(2);
  MPZVector r(1);

  bool returnVal = true;
  while (true)
  {
    string header = net.waitForMessage();
    if (header == CMTCommConsts::END_LAYER)
    {
      returnVal = false;
      break;
    }
    else if (header == CMTCommConsts::REQ_LPOLY)
    {
      mpz_t* pt1 = &rand[mi];
      mpz_t* pt2 = &rand[mi + mip1];

      time.begin_with_history();
#ifndef CMTGKR_DISABLE_CHECKS
      for (size_t i = 0; i < lpolySize; i++)
      {
        for (size_t j = 0; j < point.size(); j++)
        {
          mpz_set(vec[0], pt1[j]);
          mpz_set(vec[1], pt2[j]);
          extrap_ui(point[j], vec.raw_vec(), 2, i, circuit().prime);
        }

        for (int b = 0; b < conf.batchSize(); b++)
        {
          evaluate_V_i(lpoly[b * lpolySize + i], circuit(b)[level+1], point.raw_vec(), circuit(b).prime);
        }
      }
#endif
      time.end();

      net.sendZVector(CMTCommConsts::response(header), lpoly);
    }
    else if (header == CMTCommConsts::NOTIFY_LPOLY_EXTRAP_PT)
    {
      net.getDataAsZVector(r);

      time.begin_with_history();
      for(int i = 0; i < mip1; i++)
      {
          mpz_set(vec[0], rand[mi + i]);
          mpz_set(vec[1], rand[mi + mip1 + i]);
          extrap(zi[i], vec.raw_vec(), 2, r[0], circuit().prime);
      }

      for (int b = 0; b < conf.batchSize(); b++)
        extrap(ri[b], &lpoly[b * lpolySize], lpolySize, r[0], circuit().prime);

      time.end();

      // Implicit END_LAYER
      returnVal = true;
      break;
    }
  }

  return returnVal;
}

void CMTCircuitProver::
reportStats()
{
  cout << "computation " << (compTime.get_ru_elapsed_time() / conf.batchSize()) << endl;
  cout << "computation_latency " << (compTime.get_papi_elapsed_time() / conf.batchSize()) << endl << endl;

  cout << "p_answer_query " << time.get_ru_elapsed_time() << endl;
  cout << "p_answer_query_latency " << time.get_papi_elapsed_time() << endl << endl;

  net.getGlobalStat().print("p_net_") << endl;
  inputsStat.print("p_net_input_");

  cout << "p_d_latency " << time.get_papi_elapsed_time() << endl;
}

