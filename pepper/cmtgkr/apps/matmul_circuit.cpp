#include <cassert>

#include <common/math.h>
#include "../cmtgkr_env.h"
#include "matmul_circuit.h"

using namespace std;

static void
mat_sq(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t tmp;
  mpz_t tmp2;
  mpz_init_set_ui(tmp, 1);
  mpz_init(tmp2);

  const mpz_t *p = r;
  const mpz_t *w1 = &r[mi];
  const mpz_t *w2 = &w1[mip1];

  const int d = mi / 3;

  check_equal(tmp, p, w2, 0, d, prime);
  check_equal(tmp, &p[d], &w1[d], 0, d, prime);
  check_equal(tmp, &p[2*d], w1, &w2[d], 0, d, prime);

  mpz_mul(tmp, tmp, w2[2*d]);

  one_sub(tmp2, w1[2*d]);
  modmult(rop, tmp, tmp2, prime);

  mpz_clear(tmp);
  mpz_clear(tmp2);
}

static void
mat_reduce(mpz_t rop, const mpz_t* r, int mi, int mip1, int ni, int nip1, const mpz_t prime)
{
  mpz_t tmp, tmp2;
  mpz_init_set_ui(tmp, 1);
  mpz_init(tmp2);

  const mpz_t *p = r;
  const mpz_t *w1 = &r[mi];
  const mpz_t *w2 = &w1[mip1];

  check_equal(tmp, p, w1, w2, 0, mip1 - 1, prime);

  mpz_mul(tmp, tmp, w2[mip1 - 1]);

  one_sub(tmp2, w1[mip1 - 1]);
  modmult(rop, tmp, tmp2, prime);

  mpz_clear(tmp);
  mpz_clear(tmp2);
}

MatmulCircuit::
MatmulCircuit(int dd, bool cmtPP)
  : d(dd), useCmtPP(cmtPP)
{ }

void MatmulCircuit::
constructCircuit()
{
  if (useCmtPP)
  {
    int n = powi(2, d);
    int n_squared = n*n;
    int n_cubed = n*n*n;
    int flt_depth = getFLTDepth();
    int nconsts = 0;

    makeShell(d + 2);

    makeLevel(2*n_squared+nconsts, zero, zero);

    addGatesFromLevel(inGates, depth() - 1, 0, 2*n_squared);

    //set up input level
    makeLevel(n_cubed, zero, mat_sq);

    for (int idx = 0; idx < clayer().size(); idx++)
    {
      int j = idx & (n - 1);
      int i = (idx >> d) & (n - 1);
      int k = (idx >> (2 * d)) & (n - 1);

      int in1 = (i << d) + k;
      int in2 = n_squared + (k << d) + j;
      clayer()[idx].setWiring(GateWiring::MUL, in1, in2);

    }

    for (int i = 0; i < d; i++)
    {
      makeLevel(n_cubed >> (i + 1), mat_reduce, zero);

      for (int idx = 0; idx < clayer().size(); idx++)
      {
        int lower = idx & (clayer().size() - 1);
        int in1 = lower;
        int in2 = clayer().size() + in1;

        clayer()[idx].setWiring(GateWiring::ADD, in1, in2);
      }
    }

    addGatesFromLevel(outGates, 0, 0, (*this)[0].size());
  }
  else
  {
    int n = powi(2, d);
    int n_squared = n*n;
    int n_cubed = n*n*n;
    int flt_depth = getFLTDepth();
    int nconsts = 1;

    makeShell(flt_depth + 3 + 3*d);

    makeLevel(3*n_squared+nconsts, zero, zero);

    addGatesFromLevel(inGates, depth() - 1, 0, 2*n_squared);
    addGatesFromLevel(outGates, depth() - 1, 2*n_squared, 3*n_squared);

    //set up input level

    //mpz_set_ui(clayer().gate(3*n_squared).zValue(), 0);//need a constant 0 gate. So universe really has size 2^d-1, not 2^d
    //gates[63+3*d][3*n_squared].zValue()=0;//need a constant 0 gate. So universe really has size 2^d-1, not 2^d
    mpq_t r_q;
    mpz_t r_z;
    mpq_init(r_q);
    mpz_init(r_z);

    makeLevel(n_cubed + n_squared + nconsts,
                 mat_add_63_p3d,
                 mat_mult_63_p3d);
   
    for(int index = 0; index < clayer().size() - n_squared - nconsts; index++)
    {
          int k = (index & (n-1));
          int i = ((index >> (2*d)) & (n-1));
          int j = ((index >> d) & (n-1));

          clayer()[index].setWiring(GateWiring::MUL,
                                        (i << d) + k,
                                        (1 << (2*d)) + (k << d) + j);
    }

    for(int index = clayer().size() - n_squared - nconsts; index < clayer().size() ; index++)
    {
          clayer()[index].setWiring(GateWiring::ADD,
                                        clayer(1).size() - (clayer().size() - index),
                                        clayer(1).size() - 1);
    }

    //cout << "after 62+3*d\n";
    int step=1;

    for(int j = flt_depth+3*d, step = 1; j >=flt_depth + 2*d + 1; j--, step++)
    {
            //cout << "at level " << j << endl;
            makeLevel(n_cubed/powi(2, step) + n_squared + nconsts,
                         mat_add_below_63_p3d,
                         zero);

            if (j == flt_depth + 2*d + 1)
              clayer().add_fn = mat_add_61_p2dp1;

            makeReduceLayer(GateWiring::ADD, 0, clayer().size() - n_squared - nconsts);

            CircuitLayer& gates = clayer();
            for(int k = clayer().size() - n_squared-nconsts; k < clayer().size(); k++)
            {
                  gates[k].setWiring(
                              GateWiring::ADD,
                              clayer(1).size() - (clayer().size() - k),
                              clayer(1).size()-1);
            }
    }

    //cout << "starting 61+2*d\n";
    makeLevel(n_squared+nconsts,
                 mat_add_61_p2d,
                 zero);
    makeAddGates(0, n_squared, n_squared);
    addToLastGate(n_squared, clayer().size(), n_squared);
    
    //cout << "starting 60+2*d\n";

    /* Set up FLT */
    int flt_top = flt_depth + 2*d;
    int flt_bottom = 2*d + 1;
    mpz_t psub1;
    mpz_init_set(psub1, prime);
    mpz_sub_ui(psub1, psub1, 1);

    // Now, FLT the summed vector

    // Square level
    makeLevel(n_squared + 1,
                 zero,
                 FLT_mul_lvl1);
    makeMulGates(0, n_squared, 0);
    addToLastGate(n_squared, clayer().size());

    // Second level
    makeLevel(2*n_squared + 1,
                 FLT_add_lvl2,
                 FLT_mul_lvl2);
    makeFLTLvl2(0, 2*n_squared);
    clayer()[clayer().size() - 1].setWiring(GateWiring::ADD, clayer(1).size() - 1, clayer(1).size() - 1);

    // All but last level.
    for (int i = 2; i < flt_depth - 1; i++)
    {
      // The last gate is the zero gate.
      if (mpz_tstbit(psub1, flt_top - i - 1))
        makeLevel(2 * n_squared + 1,
                     FLT_add<1>,
                     FLT_mul<1>);
      else
        makeLevel(2 * n_squared + 1,
                     FLT_add<0>,
                     FLT_mul<0>);

      makeFLTGeneral(0, clayer().size() - 1, i);
      addToLastGate(clayer().size() - 1, clayer().size());
    }

    // Last level
    makeLevel(n_squared,
                 zero,
                 F0mult_d);
    makeFLTLast(0, n_squared);


    /* Add all of the flts together
     * There are n^2 of them, so 2*d levels.
     */
    //set up levels 0 to d-1

    int size = n_squared/2;
    for(int i = 0; i < 2 * d; i++)
    {
      makeLevel(size,
                   zero,
                   zero);
      makeReduceLayer(GateWiring::ADD, 0, clayer().size());

      size=size >> 1;
    }

    mpz_clear(psub1);
  }
}

void MatmulCircuit::
evaluate()
{
  if (!useCmtPP)
  {
    CircuitLayer& inLayer = getInputLayer();

    mpq_t tmp;
    mpq_init(tmp);

    int n = powi(2, d);
    int n_sq = n * n;
    for(int i = 0; i < n; i++)
    {
      for(int j = 0; j < n; j++)
      {
        Gate sum = inLayer.gate(2*n_sq + i*n + j);

        mpq_set_ui(sum.qValue(), 0, 1);
        for(int k = 0; k < n; k++)
        {
          mpq_mul(tmp, 
                  inLayer.gate(i*n + k).qValue(),
                  inLayer.gate(n_sq + k*n + j).qValue());
          mpq_add(sum.qValue(), sum.qValue(), tmp);
        }

        mpq_neg(sum.qValue(), sum.qValue());

        sum.setValue(sum.qValue());
      }
    }
  }

  CMTCircuit::evaluate();
}

void MatmulCircuit::
initializeInputs(const MPQVector& op, const MPQVector& magic)
{
  CMTCircuit::initializeInputs(op, magic);

  if (!useCmtPP)
  {
    CircuitLayer& inLayer = getInputLayer();
    // Set the constant 0 gate.
    inLayer.gate(inLayer.size() - 1).setValue(0);
  }
}

void MatmulCircuit::
initializeOutputs(const MPQVector& op)
{
  CMTCircuit::initializeOutputs(op);

  if (!useCmtPP)
    (*this)[0].gate(0).setValue(0);
}

MatmulCircuitBuilder::
MatmulCircuitBuilder(int dd, bool cmtPP)
  : d(dd), useCmtPP(cmtPP)
{ }

MatmulCircuit* MatmulCircuitBuilder::
buildCircuit()
{
  MatmulCircuit* c = new MatmulCircuit(d, useCmtPP);
  c->construct();
  return c;
}

