#include <stdint.h>
#include <db.h>

struct In { void placeholder; };
struct Out { int value; };

/*
  Microbenchmark to measure the cost of ramget
*/
void compute(struct In *input, struct Out *output){
  ramget(&(output->value), 0);
}
