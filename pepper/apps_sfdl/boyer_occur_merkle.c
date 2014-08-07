#include <stdint.h>
#include <db.h>

#define PATTERN_LENGTH 8
#define ALPHABET_LENGTH 16

struct In { uint8_t input[PATTERN_LENGTH]; };
struct Out { uint8_t output[ALPHABET_LENGTH]; };

void compute(struct In *input, struct Out *output) {
    uint8_t i;
    uint8_t len = PATTERN_LENGTH;
    for(i = 0; i < ALPHABET_LENGTH; i++) {
        ramput(i, &len);
    }

    for(i = 0; i < PATTERN_LENGTH - 1; i++) {
        uint8_t addr = input->input[i];
        uint8_t data = PATTERN_LENGTH - 1 - i;
        ramput(addr, &data);
    }

    for(i = 0; i < ALPHABET_LENGTH; i++) {
        ramget(&(output->output[i]), i);
    }
}
