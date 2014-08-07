#include <iostream>

#include "../cmtgkr_env.h"

#include "magic_var_operation.h"
#include "pws_circuit_parser.h"
#include "pws_circuit.h"

using namespace std;

typedef map<int, int>::const_iterator ConstGateMapIt;

static const bool DEBUG_MODE = true;
static void
error(const string& msg)
{
  if (DEBUG_MODE)
    cerr << "ERROR - " << msg << endl;
}

PWSCircuit::
PWSCircuit(PWSCircuitParser& pp)
  : parser(pp)
{
  // Inherit the parser's prime
  setPrime(parser.prime);
}

CircuitLayer& PWSCircuit::
getGatePosLayer(int gatePosLayer)
{
  return (*this)[depth() - 1 - gatePosLayer];
}

Gate PWSCircuit::
getGate(const GatePosition& pos)
{
  return getGatePosLayer(pos.layer).gate(pos.name);
}

void PWSCircuit::
evalGates(const vector<int>& start)
{
  // We don't deal with input gates.
  for (int lNum= 1; lNum < depth(); lNum++)
  {
    CircuitLayer& prevLayer = getGatePosLayer(lNum - 1);
    CircuitLayer& layer = getGatePosLayer(lNum);

    int gNumStart = start.size() > (size_t) lNum ? start[lNum] : 0;
    for (int gNum = gNumStart; gNum < layer.size(); gNum++)
    {
      Gate rop = layer.gate(gNum);
      rop.computeGateValue(prevLayer.gate(rop.wiring.in1), prevLayer.gate(rop.wiring.in2));
    }
  }
}

void PWSCircuit::
evalGates(const vector<int>& start, const vector<int>& end)
{
  // We don't deal with input gates.
  for (size_t lNum= 1; lNum < end.size(); lNum++)
  {
    CircuitLayer& prevLayer = getGatePosLayer(lNum - 1);
    CircuitLayer& layer = getGatePosLayer(lNum);

    int gNumStart = start.size() > lNum ? start[lNum] : 0;
    for (int gNum = gNumStart; gNum < end[lNum]; gNum++)
    {
      Gate rop = layer.gate(gNum);
      rop.computeGateValue(prevLayer.gate(rop.wiring.in1), prevLayer.gate(rop.wiring.in2));
    }
  }
}

void PWSCircuit::
evaluate()
{
  // Here, we assume all of the inputs have been filled in, but not the magic
  // variables. We proceed in order.
  vector<pair<vector<int>, MagicVarOperation*> >::const_iterator it = parser.magicOps.begin();
  int curOp = 0;

  vector<int> lastGuard;
  while (it != parser.magicOps.end())
  {
    // First, we evaluate all gate before the guard.
    evalGates(lastGuard, it->first);

    // Now we evaluate the magic variables. The guard guarantees that all
    // dependencies have been computed.
    it->second->computeMagicGates(*this);

    lastGuard = it->first;

    // Then, we move on to the next op.
    ++it;
  }

  // Do any remaining operations.
  evalGates(lastGuard);
}

void PWSCircuit::
makeGateMapping(vector<Gate*>& gates, CircuitLayer& layer, const vector<int>& mapping)
{
  gates.clear();

  if (mapping.empty())
    return;

  typedef vector<int>::const_iterator MapIt;
  for (MapIt it = mapping.begin(); it != mapping.end(); ++it)
  {
    gates.push_back(new Gate(layer.gate(*it)));
  }
}

void PWSCircuit::
makeGateMapping(vector<Gate*>& gates, CircuitLayer& layer, const map<int, int>& mapping, int offset)
{
  gates.clear();

  if (mapping.empty())
    return;

  int size = mapping.rbegin()->first + offset + 1;

  for (int i = 0; i < size; i++)
    gates.push_back(NULL);

  typedef map<int, int>::const_iterator MapIt;
  for (MapIt it = mapping.begin(); it != mapping.end(); ++it)
  {
    //cout << it->first << ", " << it->second << endl;
    gates[it->first + offset] = new Gate(layer.gate(it->second));
  }
}

void PWSCircuit::
constructCircuit()
{
  inGates.clear();
  outGates.clear();
  magicGates.clear();

  const CircuitDescription& desc = parser.circuitDesc;
  makeShell(desc.size());

  int numGates = 0;
  typedef CircuitDescription::const_iterator LayerIt;
  typedef LayerDescription::const_iterator GateIt;
  for (LayerIt lit = desc.begin(); lit != desc.end(); ++lit)
  {
    const LayerDescription& layer = *lit;
    makeLevel(layer.size(), NULL, NULL);
    for (size_t i = 0; i < layer.size(); i++)
    {
      numGates++;
      layer[i].makeGate(clayer()[i]);
    }
    //cout << numGates << endl;
    //cout << numGates * sizeof(Gate) << endl;
  }

  // Set up in/out mappings.
  makeGateMapping(inGates, getInputLayer(), parser.inGates, 0);
  makeGateMapping(outGates, (*this)[0], parser.outGates, -parser.outputGateBegin);
  makeGateMapping(magicGates, getInputLayer(), parser.magicGates);
}

void PWSCircuit::
initializeInputs(const MPQVector& inputs, const MPQVector& magic)
{
  CMTCircuit::initializeInputs(inputs, magic);

  mpz_t num;
  mpz_init(num);

  CircuitLayer& inLayer = getInputLayer();
  vector< pair<string, int> >::const_iterator it;
  for (it = parser.inConstants.begin(); it != parser.inConstants.end(); ++it)
  {
    mpz_set_str(num, it->first.c_str(), 10);
    inLayer.gate(it->second).setValue(num);
  }
  mpz_clear(num);
}

void PWSCircuit::
initializeOutputs(const MPQVector& op)
{
  CMTCircuit::initializeOutputs(op);

  mpz_t num;
  mpz_init(num);

  CircuitLayer& outLayer = (*this)[0];
  map<string, vector<int> >::const_iterator it;
  for (it = parser.outConstants.begin(); it != parser.outConstants.end(); ++it)
  {
    mpz_set_str(num, it->first.c_str(), 10);

    const vector<int>& gateNums = it->second;
    vector<int>::const_iterator gateNumIt;
    for (gateNumIt = gateNums.begin(); gateNumIt != gateNums.end(); ++gateNumIt)
    {
      outLayer.gate(*gateNumIt).setValue(num);
    }
  }

  mpz_clear(num);
}

PWSCircuitBuilder::
PWSCircuitBuilder(PWSCircuitParser& pp)
  : parser(pp)
{ }

PWSCircuit* PWSCircuitBuilder::
buildCircuit()
{
  PWSCircuit* c = new PWSCircuit(parser);
  c->construct();
  return c;
}

