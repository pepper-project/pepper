#ifndef CODE_PEPPER_CMTGKR_BASE_CMT_CIRCUIT_BUILDER_H_
#define CODE_PEPPER_CMTGKR_BASE_CMT_CIRCUIT_BUILDER_H_

#include "../circuit/cmt_circuit.h"

class CMTCircuitBuilder
{
  public:
  virtual ~CMTCircuitBuilder();
  virtual CMTCircuit* buildCircuit() = 0;
  virtual void destroyCircuit(CMTCircuit* c);
};

#endif

