#include <iostream>

#include "../cmtgkr_env.h"

#include "magic_var_operation.h"

using namespace std;

typedef vector<GatePosition> GatePosIt;

Gate MagicVarOperation::
getGate(PWSCircuit& c, const GatePosition& pos)
{
  return c.getGate(pos);
}

mpz_t& MagicVarOperation::
getZ(PWSCircuit& c, const GatePosition& pos)
{
  return getGate(c, pos).zValue();
}

mpq_t& MagicVarOperation::
getQ(PWSCircuit& c, const GatePosition& pos)
{
  return getGate(c, pos).qValue();
}

void MagicVarOperation::
setVal(PWSCircuit& c, const GatePosition& pos, const mpz_t val)
{
  Gate g = getGate(c, pos);
  mpz_set(g.zValue(), val);
  mpq_set_z(g.qValue(), val);
}

void MagicVarOperation::
setVal(PWSCircuit& c, const GatePosition& pos, int val)
{
  Gate g = getGate(c, pos);
  mpz_set_ui(g.zValue(), val);
  mpq_set_ui(g.qValue(), val, 1);
}

NotEqualOperation::
NotEqualOperation(GatePosition m,
                  GatePosition x1,
                  GatePosition x2)
  : M(m), X1(x1), X2(x2)
{ }

void NotEqualOperation::
computeMagicGates(PWSCircuit& c)
{
  mpz_t& x1Val = getZ(c, X1);
  mpz_t& x2Val = getZ(c, X2);

  if (mpz_cmp(x1Val, x2Val) == 0)
  {
    setVal(c, M, 0);
  }
  else
  {
    mpz_t tmp;
    mpz_init(tmp);

    mpz_sub(tmp, x1Val, x2Val);
    mpz_invert(tmp, tmp, c.prime);

    setVal(c, M, tmp);

    mpz_clear(tmp);
  }
}

LessThanIntOperation::
LessThanIntOperation(vector<GatePosition>& ms,
                     vector<GatePosition>& ns,
                     GatePosition x1,
                     GatePosition x2)
  : Ms(ms), Ns(ns), X1(x1), X2(x2)
{ }

void LessThanIntOperation::
computeMs(PWSCircuit& c, int sgn)
{
  if (sgn < 0)
  {
    setVal(c, Ms[0], 1);
    setVal(c, Ms[1], 0);
    setVal(c, Ms[2], 0);
  }
  else if (sgn == 0)
  {
    setVal(c, Ms[0], 0);
    setVal(c, Ms[1], 1);
    setVal(c, Ms[2], 0);
  }
  else
  {
    setVal(c, Ms[0], 0);
    setVal(c, Ms[1], 0);
    setVal(c, Ms[2], 1);
  }
}

void LessThanIntOperation::
computeBits(PWSCircuit& c, vector<GatePosition>& bits, const mpz_t sum, bool trueBits)
{
  mpz_t tmp;
  mpz_init_set_ui(tmp, 0);
  mpz_setbit(tmp, bits.size());

  if (trueBits)
  {
    mpz_set(tmp, sum);
  }
  else
  {
    /*
    mpz_t ff;
    mpz_init(ff);
    mpz_neg(ff, sum);
    if (mpz_cmp(tmp, ff) < 0)
      cout << "PANIC!!" << endl;
     */

    mpz_add(tmp, sum, tmp);
  }

  for (size_t i = 0; i < bits.size(); i++)
  {
    mpz_t& Bi = getZ(c, bits[i]);
    mpz_set_ui(Bi, 0);

    if (mpz_tstbit(tmp, i))
    {
      if (trueBits)
      {
        mpz_set_ui(Bi, 1);
      }
      else
      {
        mpz_setbit(Bi, i);
      }
    }

    setVal(c, bits[i], Bi);
  }
  mpz_clear(tmp);
}

void LessThanIntOperation::
computeMagicGates(PWSCircuit& c)
{
  mpz_t diff;
  mpz_init(diff);

  mpz_tdiv_q_2exp(diff, c.prime, 1);

  mpz_t& x1 = getZ(c, X1);
  mpz_t& x2 = getZ(c, X2);

  //gmp_printf("D1: %Zd\n", x1);
  //gmp_printf("D2: %Zd\n", x2);

  toTrueNumber(x1, diff, c.prime);
  toTrueNumber(x2, diff, c.prime);

  //gmp_printf("A1: %Zd\n", x1);
  //gmp_printf("A2: %Zd\n", x2);

  int sgn = mpz_cmp(x1, x2);
  computeMs(c, sgn);

  if (sgn < 0)
    mpz_sub(diff, x1, x2);
  else
    mpz_sub(diff, x2, x1);

  computeBits(c, Ns, diff, false);

  mpz_clear(diff);
}

LessThanFloatOperation::
LessThanFloatOperation(vector<GatePosition>& ms,
                       vector<GatePosition>& ns,
                       vector<GatePosition>& ds,
                       GatePosition x1,
                       GatePosition x2)
  : LessThanIntOperation(ms, ns, x1, x2), Ds(ds)
{ }

void LessThanFloatOperation::
computeMagicGates(PWSCircuit& c)
{
  mpq_t diff;
  mpq_init(diff);

  mpz_tdiv_q_2exp(mpq_numref(diff), c.prime, 1);

  mpq_t& x1 = getQ(c, X1);
  mpq_t& x2 = getQ(c, X2);

  //gmp_printf("P : %Zd\n", c.prime);
  //gmp_printf("PH: %Zd\n", mpq_numref(diff));
  //gmp_printf("D1: %Zd\n", mpq_denref(x1));
  //gmp_printf("D2: %Zd\n", mpq_denref(x2));

  //gmp_printf("N1: %Zd\n", mpq_numref(x1));
  //gmp_printf("N2: %Zd\n", mpq_numref(x2));

  toTrueNumber(x1, mpq_numref(diff), c.prime);
  toTrueNumber(x2, mpq_numref(diff), c.prime);

  //gmp_printf("D1: %Zd\n", mpq_denref(x1));
  //gmp_printf("D2: %Zd\n", mpq_denref(x2));

  //gmp_printf("N1: %Zd\n", mpq_numref(x1));
  //gmp_printf("N2: %Zd\n", mpq_numref(x2));

  int sgn = mpq_cmp(x1, x2);
  computeMs(c, sgn);

  if (sgn < 0)
    mpq_sub(diff, x1, x2);
  else
    mpq_sub(diff, x2, x1);

  computeBits(c, Ns, mpq_numref(diff), false);
  computeBits(c, Ds, mpq_denref(diff), true);

  mpq_clear(diff);
}

