#ifndef CODE_PEPPER_LIBV_LIBV_H_
#define CODE_PEPPER_LIBV_LIBV_H_

#include <crypto/crypto.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <gmp.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <common/measurement.h>

#define ROLE_VERIFIER 0
#define ROLE_PROVER 1

#define PHASE_PROVER_COMMITMENT 1
#define PHASE_PROVER_DEDUCE_QUERIES 2
#define PHASE_PROVER_PCP 3

#define USE_GMP 1

typedef struct p_compressed {
  int i, j;
  mpz_t coefficient;
} poly_compressed;

class Venezia {
  private:
    Crypto * crypto;
    mpz_t temp1, temp2, temp3, temp4;

    // cache elgamal parameters for performance
    mpz_t pub_mod, order, gen;
    mpz_t omega;
    mpz_t *powers_of_omega;
    int size_omega_set;

  public:
    Venezia(int, int, int, int);
    Venezia(int, int, int, int, int);
    Venezia(int, int, int, int, int, int);
    ~Venezia();

    void init_venezia_state();
    void dot_product(uint32_t size, mpz_t * q, mpz_t * d, mpz_t output,
                     mpz_t prime);
    void dot_product_par(uint32_t size, mpz_t * q, mpz_t * d, mpz_t output,
                         mpz_t prime, double *par_time);
    void update_con_query(mpz_t con, mpz_t beta, mpz_t q, mpz_t prime);
    void update_con_query_vec(uint32_t size, mpz_t * con, mpz_t beta,
                              mpz_t * q, mpz_t prime);
    int get_crypto_type(int role);

    // VERIFIER:
    // creates a random commitment query of size "size" and stores the
    // result in the array r; the computation also submits another array
    // where the consistency query is maintained. if the computation's
    // prover holds multiple functions; a commitment query is created
    // per function
    void create_commitment_query(uint32_t size, mpz_t *r_q, mpz_t *con_q,
                                 mpz_t prime);

    void elgamal_get_public_key_modulus(mpz_t *mod);

    void create_lin_test_queries(uint32_t size, mpz_t * l1,
                                 mpz_t * l2, mpz_t * l3, mpz_t * con,
                                 int filled, mpz_t * con_coins,
                                 mpz_t prime);

    void create_zself_corr_queries(uint32_t size, uint32_t size_1, uint32_t size_2,
                                   mpz_t *q,
                                   mpz_t *c_coins1, int fill1, mpz_t *con_q1,
                                   mpz_t *c_coins2, int fill2, mpz_t *con_q2,
                                   mpz_t prime);

    void compute_set_v(int size, mpz_t *set_v, mpz_t prime);
    void compute_set_v(int size, mpz_t *set_v, mpz_t omega, mpz_t prime); 

    void evaluate_polynomial_at_random_point(int n, int size_2,
                                             mpz_t r_star, mpz_t d_star,
                                             int num_aij, int num_bij, int num_cij,
                                             mpz_t *set_v,
                                             poly_compressed *poly_A, poly_compressed *poly_B, poly_compressed *poly_C,
                                             mpz_t *eval_poly_A, mpz_t *eval_poly_B, mpz_t *eval_poly_C,
                                             mpz_t prime);

    void create_div_corr_test_queries(int n, uint32_t size_1, uint32_t size_2,
                                      mpz_t *f1_q1, mpz_t *f1_q2, mpz_t *f1_q3,
                                      mpz_t *f2_q1,
                                      mpz_t *c_coins1, int fill1, mpz_t *con_q1,
                                      mpz_t *c_coins2, int fill2, mpz_t *con_q2,
                                      mpz_t *corr_q, mpz_t *corr_q2,
                                      mpz_t d_star,
                                      int num_aij, int num_bij, int num_cij,
                                      mpz_t *set_v,
                                      poly_compressed *poly_A, poly_compressed *poly_B, poly_compressed *poly_C,
                                      mpz_t *eval_poly_A, mpz_t *eval_poly_B, mpz_t *eval_poly_C,
                                      mpz_t prime);


    // creates quad test queries c1_q (of size s_1), c2_q (of size s_2)
    // and then does c3_q = \op{c1_q}{c2_q} + r and returns r. since
    // there could be queries of at most 3 sizes, three pointers to
    // consistency queries are passed.
    void create_corr_test_queries(uint32_t size_1, mpz_t * c1_q,
                                  uint32_t size_2, mpz_t * c2_q,
                                  mpz_t * c3_q, mpz_t * r, mpz_t * con_q1,
                                  mpz_t * con_q2, mpz_t * con_q3,
                                  int filled1, mpz_t * con_coins1, int
                                  filled2, mpz_t * con_coins2, int filled3,
                                  mpz_t * con_coins3, mpz_t prime, bool opt);

    void create_corr_test_queries_reuse(uint32_t size_1, mpz_t * c1_q,
                                        uint32_t size_2, mpz_t * c2_q,
                                        mpz_t * c3_q, mpz_t * r, mpz_t * con_q1,
                                        mpz_t * con_q2, mpz_t * con_q3,
                                        int filled1, mpz_t * con_coins1, int
                                        filled2, mpz_t * con_coins2, int filled3,
                                        mpz_t * con_coins3, mpz_t prime, bool opt);

    void create_corr_test_queries_vproduct(uint32_t m, mpz_t * f1_q1,
                                           mpz_t * f1_q2, mpz_t * f_q1,
                                           mpz_t * f_q2, mpz_t * con,
                                           int filled, mpz_t * con_coins,
                                           mpz_t prime);

    // the computation specifies the matrix/vector obtained via
    // arithmetization and this function returns two vectors q_1 is
    // basically (a + q_2).
    void create_ckt_test_queries(uint32_t size, mpz_t * a, mpz_t * q_1,
                                 mpz_t * q_2, mpz_t * con_query, int filled,
                                 mpz_t * con_coins, mpz_t prime);

    // VERIFIER:
    bool consistency_test(uint32_t size, mpz_t con_answer, mpz_t
                          com_answer, mpz_t * answers, mpz_t * con_coins,
                          mpz_t prime);

    bool lin_test(mpz_t a1, mpz_t a2, mpz_t a3, mpz_t prime);

    bool corr_test(mpz_t a1, mpz_t a2, mpz_t a3, mpz_t a4, mpz_t prime);

    // the following just does \sum_{i=1}{size/2}(arr[2i] - arr[2i+1]) + c  =? 0
    bool ckt_test(uint32_t size, mpz_t * arr, mpz_t c, mpz_t prime);

    void get_random_rational_vec(uint32_t size, mpq_t * vec, int nbits);
    void get_random_rational_vec(uint32_t size, mpq_t * vec, int n_a, int n_b);
    void get_random_signedint_vec(uint32_t size, mpq_t * vec, int na);
    void get_random_priv(mpz_t x, mpz_t p);
    void get_random_vec_priv(uint32_t size, mpz_t * vec, mpz_t n);
    void get_random_vec_priv(uint32_t size, mpz_t * vec, int nbits);
    void get_random_vec_priv(uint32_t size, mpq_t * vec, int nbits);

    void get_random_pub(mpz_t x, mpz_t p);
    void get_random_vec_pub(uint32_t size, mpz_t * vec, mpz_t n);
    void get_random_vec_pub(uint32_t size, mpz_t * vec, int nbits);
    void get_random_vec_pub(uint32_t size, mpq_t * vec, int nbits);

    void elgamal_dec(mpz_t plain, mpz_t c1, mpz_t c2);
    void dot_product_enc(uint32_t size, mpz_t * q, mpz_t * d, mpz_t output);
    void dot_product_enc(uint32_t size, mpz_t * q, mpz_t * d, mpz_t output,
                         mpz_t output2);
    void add_enc(mpz_t res1, mpz_t res2, mpz_t c1_1, mpz_t c1_2, mpz_t c2_1, mpz_t c2_2);
    void outer_product(int size_1, mpz_t *q1, int size_2, mpz_t *q2, mpz_t *mask, mpz_t prime, mpz_t *out);
    void create_corr_test_queries_vproduct2(uint32_t m, uint32_t n,
                                            mpz_t * f1_q1,
                                            mpz_t * f1_q2,
                                            mpz_t * f_q1, mpz_t * f_q2,
                                            mpz_t * con, int filled,
                                            mpz_t * con_coins,
                                            mpz_t prime);
    void add_sign(uint32_t size, mpz_t *vec);
    void dump_seed_decommit_queries();
    void init_prng_decommit_queries();
    void generate_root_of_unity(int n, mpz_t prime);
    void get_root_of_unity(mpz_t *arg_omega); 
};
#endif  // CODE_PEPPER_LIBV_LIBV_H_
