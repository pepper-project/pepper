#ifndef CODE_PEPPER_CMTGKR_CIRCUIT_MAGIC_VAR_OPERATION_H_
#define CODE_PEPPER_CMTGKR_CIRCUIT_MAGIC_VAR_OPERATION_H_

#include <vector>

#include <gmp.h>

#include "circuit_layer.h"
#include "pws_circuit.h"

#include "pws_primitives.h"

class MagicVarOperation
{
  public:
  virtual ~MagicVarOperation() { }
  virtual void computeMagicGates(PWSCircuit& c) = 0;

  protected:
  Gate getGate(PWSCircuit& c, const GatePosition& pos);

  mpz_t& getZ(PWSCircuit& c, const GatePosition& pos);
  mpq_t& getQ(PWSCircuit& c, const GatePosition& pos);

  void setVal(PWSCircuit& c, const GatePosition& pos, const mpz_t val);
  void setVal(PWSCircuit& c, const GatePosition& pos, int val);
};

class NotEqualOperation : public MagicVarOperation
{
  protected:
  GatePosition M;
  GatePosition X1;
  GatePosition X2;

  public:
  NotEqualOperation(
      GatePosition m,
      GatePosition x1,
      GatePosition x2);

  void computeMagicGates(PWSCircuit& c);
};

class LessThanIntOperation : public MagicVarOperation
{
  //protected:
  public:
  std::vector<GatePosition> Ms;
  std::vector<GatePosition> Ns;
  GatePosition X1;
  GatePosition X2;

  public:
  LessThanIntOperation(
      std::vector<GatePosition>& ms,
      std::vector<GatePosition>& ns,
      GatePosition x1,
      GatePosition x2);

  void computeMagicGates(PWSCircuit& c);

  protected:
  void computeMs(PWSCircuit& c, int sgn);
  void computeBits(PWSCircuit& c, std::vector<GatePosition>& bits, const mpz_t sum, bool trueBits);
};

class LessThanFloatOperation : public LessThanIntOperation
{
  protected:
  std::vector<GatePosition> Ds;

  public:
  LessThanFloatOperation(
      std::vector<GatePosition>& ms,
      std::vector<GatePosition>& ns,
      std::vector<GatePosition>& ds,
      GatePosition x1,
      GatePosition x2);

  void computeMagicGates(PWSCircuit& c);
};

#endif

