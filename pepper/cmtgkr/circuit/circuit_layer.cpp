#include <cmath>
#include <cassert>
#include <limits>

#include <common/utility.h>
#include <common/debug_utils.h>
#include <common/math.h>

#include "../cmtgkr_env.h"
#include "circuit_layer.h"

using namespace std;

GateWiring::
GateWiring()
  : type(ADD), in1(0), in2(0)
{ }

GateWiring::
GateWiring(GateType t, int i1, int i2)
  : type(t), in1(i1), in2(i2)
{ }

void GateWiring::
setWiring(GateType t, int i1, int i2)
{
  type = t;
  in1 = i1;
  in2 = i2;
}

bool GateWiring::
shouldBeTreatedAs(GateType gateType) const
{
  return gateType == type ||
           (gateType == MUL && type == DIV_INT) ||
           (gateType == DIV_INT && type == MUL);
}

void GateWiring::
applyGateOperation(mpz_t rop, const mpz_t op1, const mpz_t op2, const mpz_t prime) const
{
  if(type == ADD)
    {
      if (mpz_sgn(op1) == 0)
        mpz_set(rop, op2);
      else if (mpz_sgn(op2) == 0)
        mpz_set(rop, op1);
      else
        mpz_add(rop, op1, op2);
    }
    else if(type == MUL || type == DIV_INT)
    {
      if ((mpz_sgn(op1) == 0) || (mpz_sgn(op2) == 0))
        mpz_set_ui(rop, 0);
      else
        mpz_mul(rop, op1, op2);
    }
    else
    {
      assert(false);
    }

    modIfNeeded(rop, prime);
}

Gate::
Gate(CircuitLayer* layer, GateWiring wire, int label)
  : layer(layer), wiring(wire), idx(label)
{
  assert(inRange(label, 0, layer->size()));
}

void Gate::
getValue(mpq_t result) const { mpq_set(result, qValue()); }

void Gate::
getValue(mpz_t result) const { mpz_set(result, zValue()); }

mpq_t& Gate::
qValue() { return layer->qData()[idx]; }

const mpq_t& Gate::
qValue() const { return layer->qData()[idx]; }

mpz_t& Gate::
zValue() { return layer->zData()[idx]; }

const mpz_t& Gate::
zValue() const { return layer->zData()[idx]; }

void Gate::
canonicalize()
{
  modIfNeeded(qValue(), layer->circuit->prime);
  modIfNeeded(zValue(), layer->circuit->prime);
}

void Gate::
computeGateValue(const Gate& op1, const Gate& op2)
{
  MPQVector qOperand(2);
  op1.getValue(qOperand[0]);
  op2.getValue(qOperand[1]);

  MPZVector zOperand(2);
  op1.getValue(zOperand[0]);
  op2.getValue(zOperand[1]);

  MPQVector out(1);
  const mpz_t& prime = layer->circuit->prime;

  if (wiring.type == GateWiring::ADD)
   {
     if (mpq_sgn(qOperand[0]) == 0)
       mpq_set(out[0], qOperand[1]);
     else if (mpq_sgn(qOperand[1]) == 0)
       mpq_set(out[0], qOperand[0]);
     else
       mpq_add(out[0], qOperand[0], qOperand[1]);
   }
   else if (wiring.type == GateWiring::MUL)
   {
     if ((mpq_sgn(qOperand[0]) == 0) || (mpq_sgn(qOperand[1]) == 0))
     {
       mpq_set_ui(qValue(), 0, 1);
       mpz_set_ui(zValue(), 0);
       return;
     }
     else
     {
       mpq_mul(out[0], qOperand[0], qOperand[1]);
     }
   }
   else if (wiring.type == GateWiring::DIV_INT)
   {
     // This requires special handling.
     MPZVector tmp(2);
     mpz_tdiv_q_2exp(tmp[0], prime, 1);

     mpz_invert(tmp[1], zOperand[1], prime);
     toTrueNumber(tmp[1], tmp[0], prime);

     MPQVector divisor(1);
     mpq_set_z(divisor[0], tmp[1]);
     mpq_div(out[0], qOperand[0], divisor[0]);
   }
   else
   {
     assert(false);
   }

   // Mod down the number if needed.
   setValue(out[0]);
   wiring.applyGateOperation(zValue(), zOperand[0], zOperand[1], prime);
}

void Gate::
setValue(const mpq_t value)
{
  mpq_set(qValue(), value);
  convert_to_z(zValue(), value, layer->circuit->prime);
  canonicalize();
}

void Gate::
setValue(const mpz_t value)
{
  mpq_set_z(qValue(), value);
  mpz_set(zValue(), value);
  canonicalize();
}

void Gate::
setValue(int value)
{
  mpq_set_si(qValue(), value, 1);
  mpz_set_si(zValue(), value);
  canonicalize();
}

CircuitLayer::
CircuitLayer(Circuit* c, int layerIdx, int size)
  : circuit(c), layer(layerIdx),
    gates(size), add_fn(NULL), mul_fn(NULL)
{
  assert(inRange(size, 0, numeric_limits<int>::max()));
}

int CircuitLayer::
size() const
{
  return gates.size();
}

int CircuitLayer::
logSize() const
{
  return log2i(size());
}

Gate CircuitLayer::
gate(int idx)
{
  assert(inRange(idx, 0, size()));
  return Gate(this, this->operator[](idx), idx);
}

const Gate CircuitLayer::
gate(int idx) const
{
  assert(inRange(idx, 0, size()));
  // HACK, HACK, HACK
  return Gate(const_cast<CircuitLayer*>(this), this->operator[](idx), idx);
}

GateWiring& CircuitLayer::
operator[] (int idx)
{
  assert(inRange(idx, 0, size()));
  return gates[idx];
}

const GateWiring& CircuitLayer::
operator[] (int idx) const
{
  assert(inRange(idx, 0, size()));
  return gates[idx];
}

void CircuitLayer::
resize(int newSize)
{
  assert(inRange(newSize, 0, numeric_limits<int>::max()));
  gates.resize(newSize);
}

void CircuitLayer::
computeWirePredicates(mpz_t add_predr, mpz_t mul_predr, const MPZVector& rand, int inputLayerSize, const mpz_t prime) const
{
  const int mi = logSize();
  const int ni = size();
  const int mip1 = log2i(inputLayerSize);
  const int nip1 = inputLayerSize;

  bool addDone = false;
  bool mulDone = false;

  mpz_set_ui(add_predr, 0);
  mpz_set_ui(mul_predr, 0);

  if (add_fn)
  {
    add_fn(add_predr, rand.raw_vec(), mi, mip1, ni, nip1, prime);
    addDone = true;
  }

  if (mul_fn)
  {
    mul_fn(mul_predr, rand.raw_vec(), mi, mip1, ni, nip1, prime);
    mulDone = true;
  }

  if (addDone && mulDone)
    return;

  // Here, no function was provided, so we compute by brute force.
  MPZVector pChi(ni);
  MPZVector w1Chi(nip1);
  MPZVector w2Chi(nip1);

  MPZVector pR(mi);
  MPZVector wR(mip1);

  pR.copy(rand, 0, mi);
  computeChiAll(pChi,  pR,  prime);

  wR.copy(rand, mi, mip1);
  computeChiAll(w1Chi, wR, prime);

  wR.copy(rand, mi + mip1, mip1);
  computeChiAll(w2Chi, wR, prime);

  mpz_t tmp;
  mpz_init(tmp);
  for (int i = 0; i < size(); i++)
  {
    const GateWiring& wiring = gates[i];
    if (!addDone && wiring.shouldBeTreatedAs(GateWiring::ADD))
    {
      mpz_mul(tmp, pChi[i], w1Chi[wiring.in1]);
      modmult(tmp, tmp, w2Chi[wiring.in2], prime);
      mpz_add(add_predr, add_predr, tmp);
    }

    if (!mulDone && wiring.shouldBeTreatedAs(GateWiring::MUL))
    {
      mpz_mul(tmp, pChi[i], w1Chi[wiring.in1]);
      modmult(tmp, tmp, w2Chi[wiring.in2], prime);
      mpz_add(mul_predr, mul_predr, tmp);
    }
  }
  mpz_clear(tmp);

  if (!addDone)
    mpz_mod(add_predr, add_predr, prime);

  if (!mulDone)
    mpz_mod(mul_predr, mul_predr, prime);
}

LayerMPQData& CircuitLayer::
qData()
{
  assert(circuit->valid);
  return circuit->qData[layer];
}

const LayerMPQData& CircuitLayer::
qData() const
{
  assert(circuit->valid);
  return circuit->qData[layer];
}

LayerMPZData& CircuitLayer::
zData()
{
  assert(circuit->valid);
  return circuit->zData[layer];
}

const LayerMPZData& CircuitLayer::
zData() const
{
  assert(circuit->valid);
  return circuit->zData[layer];
}
