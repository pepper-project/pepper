
#include <cassert>
#include <cmath>

#include "poly_utils.h"

using namespace std;

void computeChiAll(MPZVector& rop, const MPZVector& r, const mpz_t prime)
{
  computeChiAll(rop, rop.size(), r, 0, prime);
}

void computeChiAll(MPZVector& rop, size_t n, const MPZVector& r, size_t startAt, const mpz_t prime)
{
  computeMLEAll(rop, n, r, startAt, prime, one_sub, mpz_set);
}

void
mul_chi(mpz_t rop, const uint64_t v, const mpz_t* r, int n, const mpz_t prime)
{
    mpz_t tmp;
    mpz_init(tmp);

    for(int i = 0; i < n; i++)
    {
        if( (v >> i) & 1)
        {
            modmult(rop, rop, r[i], prime);
        }
        else
        {
            one_sub(tmp, r[i]);
            modmult(rop, rop, tmp, prime);
        }
    }

    mpz_clear(tmp);
}

//computes chi_v(r), where chi is the Lagrange polynomial that takes
//boolean vector v to 1 and all other boolean vectors to 0. (we view v's bits as defining
//a boolean vector. n is dimension of this vector.
//all arithmetic done mod p
void
chi(mpz_t rop, const uint64_t v, const mpz_t* r, int n, const mpz_t prime)
{
    mpz_t chi;
    mpz_init_set_ui(chi, 1);

    mul_chi(chi, v, r, n, prime);

    mpz_set(rop, chi);
    mpz_clear(chi);
}

void
bary_precompute_weights3(MPZVector& weights, const mpz_t r, const mpz_t prime)
{
  MPZVector tmp(3);

  for (int i = 0; i < 3; i++)
    mpz_sub_ui(tmp[i], r, i);

  for (int i = 0; i < 3; i++)
  {
    int idx1 = (i + 1) % 3;
    int idx2 = (i + 2) % 3;

    mpz_mul(weights[i], tmp[idx1], tmp[idx2]);
  }

  // tmp[0] = invert(2)
  mpz_add_ui(tmp[0], prime, 1);
  mpz_tdiv_q_2exp(tmp[0], tmp[0], 1);

  modmult(weights[0], weights[0], tmp[0], prime);
  modmult_si(weights[1], weights[1], -1, prime);
  modmult(weights[2], weights[2], tmp[0], prime);
}


/*
 * The barycentric formula is as follows:
 * Define:
 *      l(r)  = (r - x0) (r - x1) ... (r - xn)
 *      wi    = 1 / prod(j=0 to n, j!=i, xi - xj)
 *      li(r) = l(r) * wi / (r - xi)
 *
 * Given evaluations y0 = P(x0), y1 = P(x1),..., yn = P(xn), from some
 * univariate polynomial with degree at most n-1, we can interpolate the value
 * of P at any point r by computing
 *      P(r) = sum(i=0 to n, li(r) * yi)
 * 
 * Note that this is not the best form for evaluation if multiple points of P
 * are required since it doesn't save the barycentric weights (wi).
 *
 * This function assumes that xi = i and returns li(r) for i=0 to n.
 */
void
bary_precompute_weights(MPZVector& weights, const mpz_t r, const mpz_t prime)
{
  mpz_t tmp, lr;
  mpz_init(tmp);
  mpz_init(lr);

  const int n = (int) weights.size();

  // First we compute weights[i] = prod(j=0 to n, j!=i, i - j)
  for (int i = 0; i < n; i++)
  {
    mpz_set_ui(weights[i], 1);
    for (int j = 0; j < n; j++)
    {
      if (i != j)
        mpz_mul_si(weights[i], weights[i], i - j);
    }
  }

  // Mod if needed.
  if ((mpz_cmp(r, prime) >= 0) || mpz_sgn(r) < 0)
    mpz_mod(tmp, r, prime);

  // The barycentric formula specified above only works
  // when r != i for i=0 to n. We fall back to standard lagrange weights
  // in this case.
  assert(mpz_sgn(tmp) >= 0);
  if (mpz_cmp_ui(tmp, n) < 0)
  {
    // Compute the l(r) / (r - i) term as lr = prod(j=0 to n, j!=i, r - j)
    for (int i = 0; i < n; i++)
    {
      mpz_set_ui(lr, 1);
      for (int j = 0; j < n; j++)
      {
        if (i != j)
        {
          mpz_sub_ui(tmp, r, j);
          modmult(lr, lr, tmp, prime);
        }
      }

      // Compute the weight: lr / weights[i].
      mpz_invert(weights[i], weights[i], prime);
      modmult(weights[i], weights[i], lr, prime);
    }
  }
  else
  {
    // We can slightly optimize the computation of li(r) by noticing that the
    // constant 1 polynomial is equal to
    //    sum(i=0 to n, li(r)).
    // If we divide li(r) by this sum, we get
    //    li(r) =          wi / (r - xi)
    //             ----------------------------
    //             sum(i=0 to n, wi / (r - xi))
    //
    // This new form allows us to replace the multiplies previously required in
    // computing l(r) with adds at the cost of one additional invert operation.
    mpz_set_ui(lr, 0);
    for (int i = 0; i < n; i++)
    {
      // tmp = r - i
      mpz_sub_ui(tmp, r, i);

      // weights[i] = (r - i) * prod(j=0 to n, j!=i, i - j)
      mpz_mul(weights[i], weights[i], tmp);

      // weights[i] = wi / (r - i)
      mpz_invert(weights[i], weights[i], prime);

      mpz_add(lr, lr, weights[i]);
    }

    mpz_invert(lr, lr, prime);
    for (int i = 0; i < n; i++)
      modmult(weights[i], weights[i], lr, prime);
  }

  mpz_clear(lr);
  mpz_clear(tmp);
}

void
bary_extrap(MPZVector& rop, const MPZVector& vec, const MPZVector& weights, const mpz_t prime)
{
  assert(vec.size() == rop.size() * weights.size());
  mpz_t tmp;
  mpz_init(tmp);

  for (size_t b = 0; b < rop.size(); b++)
  {
    mpz_set_ui(rop[b], 0);
    const mpz_t *y = &vec[b * weights.size()];

    for (size_t i = 0; i < weights.size(); i++)
    {
      mpz_mul(tmp, weights[i], y[i]);
      mpz_add(rop[b], rop[b], tmp);
    }

    mpz_mod(rop[b], rop[b], prime);
  }

  mpz_clear(tmp);
}

void
extrap3(mpz_t rop, const mpz_t* vec, const mpz_t r, const mpz_t prime)
{
  mpz_t tmp1, tmp2;
  mpz_init(tmp1);
  mpz_init(tmp2);

  // rop = vec[0] * (r - 1) * (r - 2)
  mpz_sub_ui(tmp1, r, 1);
  mpz_sub_ui(tmp2, r, 2);
  mpz_mul(rop, tmp1, tmp2);
  mpz_mul(rop, rop, vec[0]);

  // tmp2 = 2 * (r - 2) * vec[1]
  mpz_mul_2exp(tmp2, tmp2, 1);
  mpz_mul(tmp2, tmp2, vec[1]);

  // tmp1 = (r - 1) * vec[2]
  mpz_mul(tmp1, tmp1, vec[2]);

  mpz_sub(tmp1, tmp1, tmp2);
  mpz_mul(tmp1, tmp1, r);

  mpz_add(rop, rop, tmp1);

  // tmp1 = invert(2)
  mpz_add_ui(tmp1, prime, 1);
  mpz_tdiv_q_2exp(tmp1, tmp1, 1);

  mpz_mul(rop, rop, tmp1);
  mpz_mod(rop, rop, prime);

  mpz_clear(tmp1);
  mpz_clear(tmp2);
}

//extrapolate the polynomial implied by vector vec of length n to location r
void
extrap(mpz_t rop, const mpz_t* vec, const uint64_t n, const mpz_t r, const mpz_t prime)
{
    mpz_t mult, inv, out;
    mpz_init(mult);
    mpz_init(inv);
    mpz_init(out);

    mpz_set_ui(out, 0);

    for (uint64_t i = 0; i < n; i++)
    {
        mpz_set_ui(mult, 1);
        for (uint64_t j = 0; j < n; j++)
        {
            if (i != j)
            {
                mpz_set_si(inv, i - j);
                mpz_invert(inv, inv, prime);

                mpz_mul(mult, mult, inv);
                mpz_sub_ui(inv, r, j);
                modmult(mult, mult, inv, prime);
            }
        }
        addmodmult(out, mult, vec[i], prime);
    }

    mpz_set(rop, out);

    mpz_clear(mult);
    mpz_clear(inv);
    mpz_clear(out);
}

void
extrap_ui(mpz_t rop, const mpz_t* vec, const uint64_t n, const uint64_t r, const mpz_t prime)
{
    mpz_t mpzr;
    mpz_init_set_ui(mpzr, r);
    extrap(rop, vec, n, mpzr, prime);
    mpz_clear(mpzr);
}

