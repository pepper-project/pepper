#ifndef CODE_PEPPER_CMTGKR_CIRCUIT_BASIC_CMT_CIRCUIT_H_
#define CODE_PEPPER_CMTGKR_CIRCUIT_BASIC_CMT_CIRCUIT_H_

#include "cmt_circuit.h"

class BasicCMTCircuit: public CMTCircuit
{
protected:
  mpz_t psub1;

public:
  BasicCMTCircuit();
  virtual ~BasicCMTCircuit();

  // Some common circuit patterns.
  void addToLastGate(int start, int end, int shift = 0);
  void makeMulGates(int start, int end, int shift, int stride = 1);
  void makeAddGates(int start, int end, int shift, int stride = 1);

  void makeReduceLayer(GateWiring::GateType type, int start, int end);

  // Related to FLT
  void makeFLTLvl2(int start, int end);
  void makeFLTGeneral(int start, int end, int bit_num);
  void makeFLTLast(int start, int end);

  int getFLTDepth();

  void addGatesFromLevel(std::vector<Gate*>& gates, int level, int start, int end);
};

#endif /* CODE_PEPPER_CMTGKR_CIRCUIT_BASIC_CMT_CIRCUIT_H_ */
