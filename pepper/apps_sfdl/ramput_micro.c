#include <stdint.h>
#include <db.h>

struct In { int value; };
struct Out { void placeholder; };

/*
  Microbenchmark to measure the cost of ramput 
*/
void compute(struct In *input, struct Out *output){
  ramput(0, &(input->value));
}
