#ifndef CODE_PEPPER_CMTGKR_CIRCUIT_PWS_CIRCUIT_H_
#define CODE_PEPPER_CMTGKR_CIRCUIT_PWS_CIRCUIT_H_

#include <vector>
#include <map>

#include "pws_circuit_parser.h"
#include "cmt_circuit.h"
#include "../base/cmt_circuit_builder.h"

class PWSCircuit : public CMTCircuit
{
  private:
  PWSCircuitParser& parser;

  public:
  PWSCircuit(PWSCircuitParser& pp);

  Gate getGate(const GatePosition& pos);
  CircuitLayer& getGatePosLayer(int gatePosLayer);
  void evalGates(const std::vector<int>& start);
  void evalGates(const std::vector<int>& start, const std::vector<int>& end);
  virtual void evaluate();

  virtual void initializeInputs(const MPQVector& inputs, const MPQVector& magic = MPQVector(0));
  virtual void initializeOutputs(const MPQVector& outputs);

  protected:
  virtual void constructCircuit();

  private:
  void makeGateMapping(std::vector<Gate*>& gates, CircuitLayer& layer, const std::vector<int>& mapping);
  void makeGateMapping(std::vector<Gate*>& gates, CircuitLayer& layer, const std::map<int, int>& mapping, int offset);
};

class PWSCircuitBuilder : public CMTCircuitBuilder
{
  PWSCircuitParser& parser;

  public:
  PWSCircuitBuilder(PWSCircuitParser& pp);

  PWSCircuit* buildCircuit();
};


#endif

