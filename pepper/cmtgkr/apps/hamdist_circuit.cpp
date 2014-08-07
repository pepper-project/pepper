
#include "../cmtgkr_env.h"
#include "hamdist_circuit.h"

using namespace std;

HamdistCircuit::
HamdistCircuit(int dd, bool cmtPP)
  : d(dd), useCmtPP(cmtPP)
{ }

void HamdistCircuit::
constructCircuit()
{
  int n = powi(2, d);
  int flt_depth = getFLTDepth();

  makeShell(flt_depth + d + 3);

  //set up input level
  makeLevel(2*n + 2, zero, zero);

  addGatesFromLevel(inGates, depth() - 1, 0, 2*n);
  addGatesFromLevel(outGates, depth() - 1, 2*n, 2*n + 1);

  // First, add the two vector
  makeLevel(n + 2,
             hamdist_add_wrap<hamdist_sum_vec>,
             zero);
  makeAddGates(0, n, n);
  addToLastGate(n, clayer().size(), n);

  // Now, FLT the summed vector

  // Square level
  makeLevel(n + 2,
             hamdist_add_last_two,
             hamdist_flt_mul_lvl1);
  makeMulGates(0, n, 0);
  addToLastGate(n, clayer().size());

  // Second level
  makeLevel(2*n + 2,
             hamdist_add_wrap<hamdist_flt_add_lvl2>,
             hamdist_flt_mul_lvl2);
  makeFLTLvl2(0, 2*n);
  addToLastGate(2*n, clayer().size(), -n);

  // All but last level.
  for (int i = 2; i < flt_depth - 1; i++)
  {
    if (mpz_tstbit(psub1, i))
      makeLevel(2*n + 2,
                 hamdist_add_wrap< hamdist_flt_add<1> >,
                 hamdist_flt_mul<1>);
    else
      makeLevel(2*n + 2,
                 hamdist_add_wrap< hamdist_flt_add<0> >,
                 hamdist_flt_mul<0>);

    makeFLTGeneral(0, 2*n, i);
    addToLastGate(2*n, clayer().size());
  }

  // Last level
  makeLevel(n + 2,
             hamdist_add_last_two,
             hamdist_reduce);
  makeFLTLast(0, n);
  addToLastGate(n, clayer().size(), n);

  // Add up all of the FLT-ed sums
  for (int i = d - 1; i > 0; i--)
  {
    makeLevel((1 << i) + 2,
               hamdist_add_wrap<hamdist_reduce>,
               zero);
    makeReduceLayer(GateWiring::ADD, 0, 1 << i);
    addToLastGate(1 << i, clayer().size(), 1 << i);
  }

  makeLevel(2, reduce, zero);
  makeReduceLayer(GateWiring::ADD, 0, 2);

  // Add the FLT-ed sum and the claimed result.
  makeLevel(1, reduce, zero);
  makeReduceLayer(GateWiring::ADD, 0, 1);
}

void HamdistCircuit::
evaluate()
{
  CircuitLayer& inLayer = getInputLayer();

  int n = getInputSize() / 2;
  int hamdist = 0;
  for (int i = 0; i < n; i++)
  {
    if (mpz_cmp(inLayer.gate(i).zValue(), inLayer.gate(n + i).zValue()) != 0)
      hamdist++;
  }
  inLayer.gate(2*n).setValue(-hamdist);

  CMTCircuit::evaluate();
}

void HamdistCircuit::
initializeInputs(const MPQVector& op, const MPQVector& magic)
{
  CMTCircuit::initializeInputs(op, magic);
  CircuitLayer& inLayer = getInputLayer();

  // Set the constant 0 gate.
  inLayer.gate(inLayer.size() - 1).setValue(0);
}

void HamdistCircuit::
initializeOutputs(const MPQVector& op)
{
  CMTCircuit::initializeOutputs(op);
  (*this)[0].gate(0).setValue(0);
}

HamdistCircuitBuilder::
HamdistCircuitBuilder(int dd, bool cmtPP)
  : d(dd), useCmtPP(cmtPP)
{ }

HamdistCircuit* HamdistCircuitBuilder::
buildCircuit()
{
  HamdistCircuit* c = new HamdistCircuit(d, useCmtPP);
  c->construct();
  return c;
}

