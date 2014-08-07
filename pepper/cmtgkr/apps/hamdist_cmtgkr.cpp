
#include <cmtgkr/base/main_impl.h>
#include "hamdist_circuit.h"

int main(int argc, char** argv)
{
  return run_gkr_netapp<HamdistCircuitBuilder>(argc, argv, construct_circuit_switch);
}

