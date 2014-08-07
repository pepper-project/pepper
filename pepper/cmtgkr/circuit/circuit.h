#ifndef CODE_PEPPER_CMTGKR_CIRCUIT_CIRCUIT_H_
#define CODE_PEPPER_CMTGKR_CIRCUIT_CIRCUIT_H_

#include <gmp.h>
#include <vector>

#include "circuit_data.h"
#include "circuit_layer.h"

// TODO: Tuning parameter. Auto generate?
#define MEGABYTE (1l << 20)
#define GIGABYTE (1l << 30)
#define CIRCUIT_DATA_BUDGET (4 * GIGABYTE)

class Circuit
{
protected:
  MPQData qData;
  MPZData zData;
  std::vector<CircuitLayer> layers;

  bool valid;
  friend class CircuitLayer;

public:
  mpz_t prime;

public:
  Circuit(size_t primeSize = 128);
  Circuit(const Circuit& other); // Disabled
  virtual ~Circuit();

  Circuit& operator=(const Circuit& other); // Disabled

  int depth() const;

  CircuitLayer&       operator[] (int lvl);
  const CircuitLayer& operator[] (int lvl) const;

  CircuitLayer&       getInputLayer();
  const CircuitLayer& getInputLayer() const;

  void construct();

  virtual void evaluate();

  void setPrime(const mpz_t prime);
  int get_prime_nbits() const;
  void print() const;

protected:
  void finalizeConstruct();
  virtual void constructCircuit() = 0;

public:
  static void loadPrime(mpz_t prime, size_t numBits);
};
#endif

