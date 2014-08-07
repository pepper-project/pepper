
#include <cmtgkr/base/main_impl.h>
#include "matmul_circuit.h"

int main(int argc, char** argv)
{
  return run_gkr_netapp<MatmulCircuitBuilder>(argc, argv, construct_circuit_switch);
}

