#include <iostream>
#include <cassert>

#include <common/utility.h>
#include <common/debug_utils.h>

#include "../cmtgkr_env.h"
#include "cmt_circuit.h"

using namespace std;

size_t CMTCircuit::
getInputSize() const
{
  return inGates.size();
}

size_t CMTCircuit::
getOutputSize() const
{
  return outGates.size();
}

size_t CMTCircuit::
getMagicSize() const
{
  return magicGates.size();
}

static void
setGates(vector<Gate*>& gates, const MPQVector& vec)
{
  for (size_t i = 0; i < gates.size(); i++)
  {
    if (gates[i] && i < vec.size())
      gates[i]->setValue(vec[i]);
  }
}

void CMTCircuit::
initializeInputs(const MPQVector& inputs, const MPQVector& magic)
{
  setGates(inGates, inputs);
  setGates(magicGates, magic);
}

void CMTCircuit::
initializeOutputs(const MPQVector& outputs)
{
  setGates(outGates, outputs);
}

static void
getGates(MPQVector& vec, const vector<Gate*>& gates, const mpz_t prime)
{
  for (size_t i = 0; i < gates.size(); i++)
  {
    if (gates[i] && i < vec.size())
    {
      mpq_set(vec[i], gates[i]->qValue());
      modIfNeeded(vec[i], prime, 1);
    }
  }
}

void CMTCircuit::
getInputs(MPQVector& inputs) const
{
  getGates(inputs, inGates, prime);
}

void CMTCircuit::
getOutputs(MPQVector& outputs) const
{
  getGates(outputs, outGates, prime);
}

void CMTCircuit::
getMagics(MPQVector& magics) const
{
  getGates(magics, magicGates, prime);
}

void CMTCircuit::
makeShell(int nlevels)
{
  layers.clear();
  for (int i = 0; i < nlevels; i++)
  {
    layers.push_back(CircuitLayer(this, i, 0));
  }

  currentLevel = nlevels;
}

void CMTCircuit::
makeLevel(int size, mle_fn add_ifn, mle_fn mul_ifn)
{
  makeLevel(currentLevel - 1, size, add_ifn, mul_ifn);
}

void CMTCircuit::
makeLevel(int level, int size, mle_fn add_ifn, mle_fn mul_ifn)
{
  CircuitLayer& newLayer = layers[level];
  newLayer.resize(size);
  newLayer.add_fn = add_ifn;
  newLayer.mul_fn = mul_ifn;

  currentLevel = level;
}

CircuitLayer& CMTCircuit::
clayer(int shift)
{
  assert(inRange(currentLevel + shift, 0, depth()));
  return layers[currentLevel + shift];
}
