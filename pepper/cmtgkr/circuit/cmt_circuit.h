#ifndef CODE_PEPPER_CMTGKR_CIRCUIT_CMT_CIRCUIT_H_
#define CODE_PEPPER_CMTGKR_CIRCUIT_CMT_CIRCUIT_H_

#include <vector>

#include <common/mpnvector.h>

#include "circuit.h"

class CMTCircuit : public Circuit
{
private:
  int currentLevel;

protected:
  std::vector<Gate*> inGates;
  std::vector<Gate*> outGates;
  std::vector<Gate*> magicGates;

public:
  virtual size_t getInputSize() const;
  virtual size_t getOutputSize() const;
  virtual size_t getMagicSize() const;

  virtual void initializeInputs(const MPQVector& inputs, const MPQVector& magic = MPQVector(0));
  virtual void initializeOutputs(const MPQVector& outputs);

  void getInputs(MPQVector& inputs) const;
  void getOutputs(MPQVector& outputs) const;
  void getMagics(MPQVector& magics) const;

  protected:
  void makeShell(int nlevels);

  // Create the next level. If no level is specified
  // create the current_lvl-1 level.
  void makeLevel(int size, mle_fn add_fn, mle_fn mul_fn);
  void makeLevel(int level, int size, mle_fn add_fn, mle_fn mul_fn);

  CircuitLayer& clayer(int shift = 0);
};

#endif

