
#include <string>
#include <vector>
#include <limits>
#include <iostream>

#include "../circuit/circuit.h"
#include "../circuit/pws_circuit_parser.h"

using namespace std;

struct CountStat
{
  size_t total;
  size_t min;
  size_t max;
  size_t num;

  CountStat() :
    total(0),
    min(numeric_limits<size_t>::max()),
    max(0),
    num(0)
  { }
};

template<typename T>
static void
print(const string str, const T& val)
{
  cout << str << " " << val << endl;
}

int main(int argc, char** argv)
{
  if (argc < 1)
  {
    cerr << "Missing PWS filename." << endl;
    return 1;
  }

  MPZVector prime(1);
  Circuit::loadPrime(prime[0], 128);

  PWSCircuitParser p(prime[0]);
  p.parse(argv[1]);

  CountStat numGates;
  int numIn = p.inGates.size();
  int numOut = p.outGates.size();
  int numMagic = p.magicGates.size();
  int numInConsts = p.inConstants.size();
  int numOutConsts = 0;

  const CircuitDescription& desc = p.circuitDesc;
  typedef CircuitDescription::const_iterator LayerIt;
  typedef LayerDescription::const_iterator GateIt;
  numGates.num = desc.size();
  for (LayerIt lit = desc.begin(); lit != desc.end(); ++lit)
  {
    const LayerDescription& layer = *lit;
    numGates.total += layer.size();
    numGates.min = min(numGates.min, layer.size());
    numGates.max = max(numGates.max, layer.size());
  }
  numGates.min = min(numGates.min, numGates.max);

  map<string, vector<int> >& outConsts = p.outConstants;
  map<string, vector<int> >::const_iterator it;
  for (it = outConsts.begin(); it != outConsts.end(); ++it)
  {
    const vector<int>& gateNums = it->second;
    numOutConsts += gateNums.size();
  }

  print("num_total_inputs", numIn + numMagic + numInConsts);
  print("num_input_vars", numIn);
  print("num_implicit_vars", numMagic);
  print("num_input_constants", numInConsts);

  cout << endl;
  print("num_total_outputs", numOut + numOutConsts);
  print("num_output_vars", numOut);
  print("num_output_constants", numOutConsts);

  cout << endl;
  print("num_total_gates", numGates.total);
  print("num_layers", numGates.num);
  print("num_avg_gates_per_layer", (double) numGates.total / numGates.num);
  print("num_min_gates_per_layer", numGates.min);
  print("num_max_gates_per_layer", numGates.max);

  cout << endl;
  print("num_local_ops",        p.opCount.numOps());
  print("num_local_multiplies", p.opCount.numMults);
  print("num_local_adds",       p.opCount.numAdds);
  print("num_local_int_divs",   p.opCount.numIntDivs);
  print("num_local_ineqs",      p.opCount.numIneqs);
  print("num_local_cmps",       p.opCount.numCmps);
}
