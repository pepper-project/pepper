#include <iostream>

#include "circuit/pws_circuit_parser.h"
#include "circuit/pws_circuit.h"

using namespace std;

int main(int argc, char **argv)
{
  if (argc > 1)
  {
    PWSCircuitParser parser;
    PWSCircuit c(parser);

    cout << "==== Constructing Circuit ====" << endl;
    parser.parse(argv[1], c.prime);
    cout << "==== Construction Complete ====" << endl;

    cout << "Constructing circuit" << endl;
    c.construct();
    MPQVector vec(c.getInputSize());
    for (size_t i = 0; i < vec.size(); i++)
      mpq_set_ui(vec[i], i, 1);

    parser.printCircuitDescription();
    //c.print();

    cout << "init input" << endl;
    c.initializeInputs(vec);

    cout << "evaluate" << endl;
    c.evaluate();

    //exit(1);
    c.print();
  }
  else
  {
    cout << "ERROR: Requires pws file." << endl;
  }
}

