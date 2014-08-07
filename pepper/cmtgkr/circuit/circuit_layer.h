#ifndef CODE_PEPPER_CMTGKR_CIRCUIT_CIRCUIT_LAYER_H_
#define CODE_PEPPER_CMTGKR_CIRCUIT_CIRCUIT_LAYER_H_

#include <gmp.h>
#include <common/mpnvector.h>

#include "circuit_data.h"

typedef void (*mle_fn)(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime);

class CircuitLayer;
class Circuit;

class GateWiring
{
public:
  enum GateType {
    ADD, MUL, DIV_INT
  };

  GateType type;
  int in1;
  int in2;

public:
  GateWiring();
  GateWiring(GateType t, int i1, int i2);

  void setWiring(GateType t, int i1, int i2);

  bool shouldBeTreatedAs(GateType gateType) const;
  void applyGateOperation(mpz_t rop, const mpz_t op1, const mpz_t op2, const mpz_t prime) const;
};

class Gate
{
public:
  CircuitLayer* layer;
  GateWiring wiring;
  int idx;

public:
  Gate(CircuitLayer* layer, GateWiring wire, int label);

  void getValue(mpz_t result) const;
  void getValue(mpq_t result) const;

  mpq_t&       qValue();
  const mpq_t& qValue() const;
  mpz_t&       zValue();
  const mpz_t& zValue() const;

  void canonicalize();
  void computeGateValue(const Gate& op1, const Gate& op2);

  void setValue(const mpq_t value);
  void setValue(const mpz_t value);
  void setValue(int val);
};

class CircuitLayer
{
private:
  Circuit* circuit;
  int layer;

  std::vector<GateWiring> gates;

  friend class Gate;

public:
  mle_fn add_fn;
  mle_fn mul_fn;

public:
  CircuitLayer(Circuit* c, int layerIdx, int size = 0);

  int size() const;
  int logSize() const;

  Gate       gate(int idx);
  const Gate gate(int idx) const;

  GateWiring&       operator[](int idx);
  const GateWiring& operator[](int idx) const;

  void resize(int newSize);
  void computeWirePredicates(
            mpz_t add_predr, mpz_t mul_predr,
            const MPZVector& rand, int inputLayerSize,
            const mpz_t prime) const;

protected:
  LayerMPQData&       qData();
  const LayerMPQData& qData() const;

  LayerMPZData&       zData();
  const LayerMPZData& zData() const;
};

#endif

