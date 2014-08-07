#ifndef CODE_PEPPER_COMMON_CONSTRAINT_GEN_H_
#define CODE_PEPPER_COMMON_CONSTRAINT_GEN_H_
#include <gmp.h>
#include <common/utility.h>

class ConstraintGenerator 
{
public:
  ConstraintGenerator(mpz_t& prime);
  void init_v(mpz_t*, mpz_t*, mpz_t*, mpz_t*, int);
  void init_p(mpq_t*);
  void init_size();

  //Returns the index of the first constraint created, the sequential indices contain the rest
  //int create_new_constraints(int num_constraints);
  //Returns the ID of the first variable created, the sequential indices contain the rest
  int create_new_variables(int num_variables);
  //Returns the index of the created variable
  int create_new_variable();
 
  int create_new_constraint();
  void add_term(mpz_t cooef, int variableA, int variableB, int constraintID);
  void add_term(mpz_t cooef, int* variables, int variables_length, int constraintID);

  int create_distance_squared(int x, int y, int d);
  int create_product(int a, int b); 
  int create_constant(mpq_t& value);

  //Returns the ID of a variable, enforced to hold the sum of the terms corresponding to the ids in the input vector.
  int create_sum(std::vector<int>& terms); 
    
  //Build a comparison constraint with numerator resolution R and denominator resolution T
  void create_constraint_XltY(int X, int Y, int R, int T);
  void create_constraint_XgtY(int X, int Y, int R, int T);
  void create_constraint_XleY(int X, int Y, int R, int T);
  void create_constraint_XgeY(int X, int Y, int R, int T);

  void decompose_exp2bits(mpz_t a_, int bits, int bits_length);
  void create_constraint_is_some_exp2bits_representation(int Xb, int Xb_length);

  void decompose_bits(mpz_t a_, int bits, int bits_length);
  void create_constraint_is_some_bits_representation(int Xb, int Xb_length);

  //Misc
  void create_constraint_isBinary(int x);
  int create_optional_constraint_neq(int x, int y);  

  //Vector equality, inequality
  void create_constraint_eq_vec(int x, int y, int d); 
  int create_optional_constraint_neq_vec(int x, int y, int d); 

  //2D geometry: vertices and line segment intersection
  int create_optional_constraint_segment_not_contains_2d(int e, int x, int R, int T);
  int create_optional_constraint_segments_intersect_2d(int e, int f, int R, int T);

  //Returns the number of basic constraints generated so far
  int constraints_size();
  //Returns the number of variables generated so far
  int variables_size();
  //Returns whether or not the generator is in PROVER mode.
  bool is_mode_prover();

private:
  mpz_t temp_constraint, temp_constraint2, temp_constraint3, temp_constraint4, temp_constraint_upper, temp_constraint_upper2, one, neg1;
  mpq_t temp_constraintq, oneq, zeroq;
  int constraints_next, variables_next, mode, current_constraint, terms_in_constraint;

  //Verifier Specific (V cares about both constraints and variables)
  mpz_t *c_value, *ckt_cooefs1, *ckt_cooefs2, *gamma, *prime;
  int ckt_cooefs1_length;

  //Prover Specific (P cares about only variables)
  mpq_t *qF1;

  //Represents variable X as Xb_length bits starting at index Xb (return value). Note that no constraints are added checking that the representation has the correct value.
  //int create_unsigned_bitwise_representation(ConstraintVariable& X, int Xb_length);
  void init_state();
};
#endif
