#ifndef CODE_PEPPER_CMTGKR_APPS_MATMUL_CIRCUIT_H_
#define CODE_PEPPER_CMTGKR_APPS_MATMUL_CIRCUIT_H_

#include "../circuit/basic_cmt_circuit.h"
#include "../base/cmt_circuit_builder.h"

class MatmulCircuit : public BasicCMTCircuit
{
  int d;
  bool useCmtPP;

  public:
  MatmulCircuit(int dd, bool cmtPP);

  void initializeInputs(const MPQVector& op, const MPQVector& magic);
  void initializeOutputs(const MPQVector& op);

  virtual void evaluate();

  protected:
  virtual void constructCircuit();
};

class MatmulCircuitBuilder : public CMTCircuitBuilder
{
  int d;
  bool useCmtPP;

  public:
  MatmulCircuitBuilder(int dd, bool cmtPP);

  MatmulCircuit* buildCircuit();
};

#endif
