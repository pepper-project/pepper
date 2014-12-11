#include <stdint.h>

#define PATTERN_LENGTH 512
#define ALPHABET_LENGTH 16128

struct In { uint8_t input[PATTERN_LENGTH]; };
struct Out { uint8_t output[ALPHABET_LENGTH]; };

void compute(struct In *input, struct Out *output) {
    uint8_t i;
    for(i = 0; i < ALPHABET_LENGTH; i++) {
        output->output[i] = PATTERN_LENGTH;
    }

    for(i = 0; i < PATTERN_LENGTH - 1; i++) {
        uint8_t addr = input->input[i];
        output->output[addr] = PATTERN_LENGTH - 1 - i;
    }
}
