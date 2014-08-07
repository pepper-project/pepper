#ifndef TESTS_TEST_UTILS_H
#define TESTS_TEST_UTILS_H
#include <iostream>
#include <gmp.h>
#include <cstdlib>

using namespace std;

#define START_TEST(name) cout << "Test " << (name) << " function..." << flush
#define PASS() cout << "PASSED!" << endl;
#define RUN_TEST(passed)    \
  do {                  \
    if (!(passed))      \
      exit(1);          \
    else                \
      PASS();           \
  } while (0)

#define START_WORK(msg)   cout << (msg) << "..." << flush
#define DONE() cout << "DONE." << endl;

typedef void (*cmp_callback_func)(int);

static void do_nothing(int i) {};

void init(int *array, size_t n);
void init(mpz_t a, int nbits);
void init(mpz_t *array, size_t n, int nbits = 0);
void rand_init(int *array, size_t n);
void rand_init(mpz_t *array, size_t n, int nbits);
void rand(mpz_t rop, mpz_t op);
void clear(mpz_t *array, size_t n);
bool cmp(const mpz_t correct, const mpz_t actual, cmp_callback_func f = do_nothing);
bool cmp(const mpz_t *correct, const mpz_t *actual, int n,
         cmp_callback_func f = do_nothing);
bool cmp(const char *correct, const char *actual, int n,
         cmp_callback_func f = do_nothing);
#endif
