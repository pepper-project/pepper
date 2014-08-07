#include "basic_cmt_circuit.h"

using std::vector;

BasicCMTCircuit::
BasicCMTCircuit()
{
  mpz_init(psub1);
  mpz_sub_ui(psub1, prime, 1);
}

BasicCMTCircuit::
~BasicCMTCircuit()
{
  mpz_clear(psub1);
}

void BasicCMTCircuit::
addToLastGate(int start, int end, int shift)
{
  for (int i = start; i < end; i++)
    clayer()[i].setWiring(GateWiring::ADD, shift + i, clayer(1).size() - 1);
}

template<GateWiring::GateType type> static void
makeGates(CircuitLayer& layer, int start, int end, int shift, int stride)
{
  for (int i = start; i < end; i += stride)
    layer[i].setWiring(type, i, shift + i);
}

void BasicCMTCircuit::
makeMulGates(int start, int end, int shift, int stride)
{
  makeGates<GateWiring::MUL>(clayer(), start, end, shift, stride);
}

void BasicCMTCircuit::
makeAddGates(int start, int end, int shift, int stride)
{
  makeGates<GateWiring::ADD>(clayer(), start, end, shift, stride);
}

void BasicCMTCircuit::
makeReduceLayer(GateWiring::GateType type, int start, int end)
{
  for (int i = start; i < end; i++)
    clayer()[i].setWiring(type, 2*i, 2*i + 1);
}

void BasicCMTCircuit::
makeFLTLvl2(int start, int end)
{
  for (int i = start; i < end; i++)
  {
    if (i & 1)
      clayer()[i].setWiring(GateWiring::MUL, i >> 1, i >> 1);
    else
      clayer()[i].setWiring(GateWiring::ADD, i >> 1, clayer(1).size() - 1);
  }
}

void BasicCMTCircuit::
makeFLTGeneral(int start, int end, int bit_num)
{
  for (int i = start; i < end; i++)
  {
    if (i & 1)
    {
      clayer()[i].setWiring(GateWiring::MUL, i, i);
    }
    else
    {
      if (mpz_tstbit(psub1, bit_num))
        clayer()[i].setWiring(GateWiring::MUL, i, i + 1);
      else
        clayer()[i].setWiring(GateWiring::ADD, i, clayer(1).size() - 1);

    }
  }
}

void BasicCMTCircuit::
makeFLTLast(int start, int end)
{
  makeReduceLayer(GateWiring::MUL, start, end);
}

int BasicCMTCircuit::
getFLTDepth()
{
  return mpz_sizeinbase(psub1, 2);
}

void BasicCMTCircuit::
addGatesFromLevel(vector<Gate*>& gates, int level, int start, int end)
{
  CircuitLayer& layer = (*this)[level];
  for (int i = start; i < end; i++)
  {
    gates.push_back(new Gate(layer.gate(i)));
  }
}
