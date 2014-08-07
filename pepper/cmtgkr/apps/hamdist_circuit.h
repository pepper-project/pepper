#ifndef CODE_PEPPER_CMTGKR_APPS_HAMDIST_CIRCUIT_H_
#define CODE_PEPPER_CMTGKR_APPS_HAMDIST_CIRCUIT_H_

#include "../circuit/basic_cmt_circuit.h"
#include "../base/cmt_circuit_builder.h"

class HamdistCircuit : public BasicCMTCircuit
{
  int d;
  bool useCmtPP;

  public:
  HamdistCircuit(int dd, bool cmtPP);

  void initializeInputs(const MPQVector& op, const MPQVector& magic);
  void initializeOutputs(const MPQVector& op);

  virtual void evaluate();

  protected:
  virtual void constructCircuit();
};

class HamdistCircuitBuilder : public CMTCircuitBuilder
{
  int d;
  bool useCmtPP;

  public:
  HamdistCircuitBuilder(int dd, bool cmtPP);

  HamdistCircuit* buildCircuit();
};

#endif

