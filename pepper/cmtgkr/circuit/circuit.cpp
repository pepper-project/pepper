#include <gmp.h>
#include <cassert>

#include <common/utility.h>
#include <common/debug_utils.h>

#include "circuit.h"

typedef vector<CircuitLayer>::const_iterator CLayerIt;
typedef vector<CircuitLayer>::reverse_iterator ReverseLayerIt;

Circuit::
Circuit(size_t primeSize)
  : qData(CIRCUIT_DATA_BUDGET, vector<size_t>()),
    zData(CIRCUIT_DATA_BUDGET, vector<size_t>()),
    valid(true)
{
  mpz_init(prime);
  loadPrime(prime, primeSize);
}

Circuit::
~Circuit()
{
  mpz_clear(prime);
}

int Circuit::
depth() const
{
  return layers.size();
}

CircuitLayer& Circuit::
operator[] (int lvl)
{
  assert(inRange(lvl, 0, depth()));
  return layers[lvl];
}

const CircuitLayer& Circuit::
operator[] (int lvl) const
{
  assert(inRange(lvl, 0, depth()));
  return layers[lvl];
}

const CircuitLayer& Circuit::
getInputLayer() const
{
  assert(!layers.empty());
  return layers.back();
}

CircuitLayer& Circuit::
getInputLayer()
{
  assert(!layers.empty());
  return layers.back();
}

void Circuit::
construct()
{
  valid = false;
  constructCircuit();
  finalizeConstruct();
}

void Circuit::
finalizeConstruct()
{
  if (!valid)
  {
    vector<size_t> sizes;
    for (vector<CircuitLayer>::const_iterator it = layers.begin();
          it != layers.end();
          ++it)
    {
      sizes.push_back(it->size());
    }
    assert(sizes.size() == size_t(depth()));

    zData.setSizes(sizes);
    qData.setSizes(sizes);

    valid = true;
  }
}

void Circuit::
evaluate()
{
  if (depth() < 2)
    return;

  ReverseLayerIt it = layers.rbegin();

  // Evaluate from the top down.
  while (it != --layers.rend())
  {
    CircuitLayer& prevLayer = *it;
    CircuitLayer& layer = *(++it);
    for(int j = 0; j < layer.size(); j++)
    {
      Gate ropgate = layer.gate(j);
      Gate op1gate = prevLayer.gate(ropgate.wiring.in1);
      Gate op2gate = prevLayer.gate(ropgate.wiring.in2);

      ropgate.computeGateValue(op1gate, op2gate);

      if (false) { // DEBUG {
      gmp_printf("i: ?? | j: %d | ropgate.type: %d | ropgate.in1: %d | ropgate.in2: %d | ropgate: %Zd\n",
                 j, ropgate.wiring.type, ropgate.wiring.in1, ropgate.wiring.in2, ropgate.zValue());
      }
    }
    // DEBUG cout << endl;

  }
}

void Circuit::
setPrime(const mpz_t p)
{
  mpz_set(prime, p);
}

int Circuit::
get_prime_nbits() const
{
  return mpz_sizeinbase(prime, 2);
}

void Circuit::
print() const
{
  int lvl = 0;
  for (CLayerIt it = layers.begin(); it != layers.end(); ++it)
  {
    const CircuitLayer& l = *it;

    for (int i = 0; i < l.size(); i++)
    {
      const Gate gate = l.gate(i);
      gmp_printf("%02d | %02d | type: %d | in1: %02d | in2: %02d | val: %Zd\n",
                 lvl, i, gate.wiring.type, gate.wiring.in1,
                 gate.wiring.in2, gate.zValue());
    }
    cout << endl;
    lvl++;
  }
}

void Circuit::
loadPrime(mpz_t prime, size_t numBits)
{
  stringstream ss;
  ss << "prime_" << numBits << ".txt";
  load_txt_scalar(prime, ss.str().c_str(), "static_state");
  // DEBUG
  //mpz_set_ui(prime, 2305843009213693951);
}
