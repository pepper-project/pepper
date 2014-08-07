#include "cmt_circuit_builder.h"

CMTCircuitBuilder::
~CMTCircuitBuilder()
{ }

void CMTCircuitBuilder::
destroyCircuit(CMTCircuit* c)
{
  delete c;
}

