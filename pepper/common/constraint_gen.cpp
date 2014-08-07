#include<common/constraint_gen.h>

#include <iostream>
#include <stdlib.h>
#include <string.h>

static int MODE_SIZE = 1, MODE_V = MODE_SIZE+1, MODE_P = MODE_V+1;

ConstraintGenerator::ConstraintGenerator(mpz_t& prime_) {
  alloc_init_scalar(temp_constraint);
  alloc_init_scalar(temp_constraint2);
  alloc_init_scalar(temp_constraint3);
  alloc_init_scalar(temp_constraint4);
  alloc_init_scalar(temp_constraint_upper);
  alloc_init_scalar(temp_constraint_upper2);
  alloc_init_scalar(temp_constraintq);
  alloc_init_scalar(one);
  alloc_init_scalar(neg1);
  mpz_set_si(one, 1);
  mpz_set_si(neg1, -1);
  prime = &prime_;
  alloc_init_scalar(oneq);
  alloc_init_scalar(zeroq);
  mpq_set_si(oneq, 1, 1);
  mpq_set_si(oneq, 0, 1);
}

void ConstraintGenerator::init_state() {
  variables_next = 0;
  constraints_next = 0;
  current_constraint = -1;
}

void ConstraintGenerator::init_v(mpz_t* c_value_, mpz_t* ckt_cooefs1_, mpz_t* ckt_cooefs2_, mpz_t* gamma_, int num_variables) {
  mode = MODE_V;
  c_value = c_value_;
  ckt_cooefs1 = ckt_cooefs1_;
  ckt_cooefs1_length = num_variables;
  ckt_cooefs2 = ckt_cooefs2_;
  gamma = gamma_;

  init_state();
}
void ConstraintGenerator::init_p(mpq_t* qF1_) {
  mode = MODE_P;
  qF1 = qF1_;
  init_state();
}
void ConstraintGenerator::init_size() {
  mode = MODE_SIZE;
  init_state();
}

int ConstraintGenerator::variables_size() {
  return variables_next;
}

int ConstraintGenerator::constraints_size() {
  return constraints_next;
}

bool ConstraintGenerator::is_mode_prover() {
  return mode == MODE_P;
}

int ConstraintGenerator::create_new_variable() {
  int availableVariable = variables_next;
  variables_next += 1;
  return availableVariable;
}

int ConstraintGenerator::create_new_variables(int num_variables) {
  if (num_variables <= 0) {
    cout << "ERROR: create_new_variables called with nonpositive argument" << endl;
    return -1;
  }
  int availableVariable = variables_next;
  variables_next += num_variables;
  return availableVariable;
}

int ConstraintGenerator::create_new_constraint() {
  int toRet = constraints_next;
  constraints_next += 1;
  return toRet;
}
void ConstraintGenerator::add_term(mpz_t cooeficient, int* variables, int variables_length, int constraintID) {
  if (variables_length <= 0) {
    cout << "ERROR: create_new_constraints called with nonpositive argument" << endl;
    return;
  }
  int i, j;
  if (variables_length >= 3) {
    i = variables[0];
    for(int inProduct = 2; inProduct < variables_length; inProduct++) {
      int intermediate_product = create_new_variable();
      int intermediate_product_verify = create_new_constraint();

      int nextFactor = variables[inProduct-1];
      if (mode == MODE_P) {
        mpq_mul(qF1[intermediate_product], qF1[i], qF1[nextFactor]);
      }

      add_term(one, intermediate_product, -1, intermediate_product_verify);
      add_term(neg1, i, nextFactor, intermediate_product_verify);

      i = intermediate_product;
    }
    j = variables[variables_length-1];
  } else {
    i = variables[0];
    j = -1;
    if (variables_length >= 2) {
      j = variables[1];
    }
  }
  add_term(cooeficient, i, j, constraintID);
}

void ConstraintGenerator::add_term(mpz_t cooeficient, int i, int j, int constraintID) {
  if (mode == MODE_V) {
    mpz_t& c = temp_constraint;
    mpz_mul(c, cooeficient, gamma[constraintID]);
    //Constant term
    if (i<0 && j <0) {
      mpz_add(*c_value, *c_value, c);
      return;
    }
    //Linear term
    if (j>=0 && i<0) {
      mpz_add(ckt_cooefs1[j], ckt_cooefs1[j], c);
      return;
    }
    if (i>=0 && j<0) {
      mpz_add(ckt_cooefs1[i], ckt_cooefs1[i], c);
      return;
    }
    //Quadratic term. At the moment, there are two places this could go in ckt_cooefs2.
    mpz_add(ckt_cooefs2[i*ckt_cooefs1_length + j], ckt_cooefs2[i*ckt_cooefs1_length+j], c);
  }
  if (mode == MODE_P || mode == MODE_SIZE) {
    //Nothing.
  }
}

/*
int ConstraintGenerator::create_new_constraints(int num_constraints)
{
  if (num_constraints <= 0){
    cout << "ERROR: create_new_constraints called with nonpositive argument" << endl;
    return -1;
  }
  int availableConstraint = constraints_next;
  constraints_next += num_constraints;
  return availableConstraint;
}
*/


int ConstraintGenerator::create_constant(mpq_t& qvalue) {
  int toRet = create_new_variable();
  if (mode == MODE_P) {
    mpq_set(qF1[toRet], qvalue);
  }

  mpz_t& c = temp_constraint_upper;
  convert_to_z(1, &c, &qvalue, *prime);

  int polyID = create_new_constraint();
  add_term(one, toRet, -1, polyID);
  mpz_neg(c, c);
  add_term(c, -1, -1, polyID);
  return toRet;
}

int ConstraintGenerator::create_distance_squared(int p1, int p2, int n) {
  int d = create_new_variable();
  if (mode == MODE_P) {
    mpq_t& c = temp_constraintq;
    mpq_set_si(qF1[d], 0, 1);
    for(int i = 0; i < n; i++) {
      mpq_sub(c, qF1[p1 + i], qF1[p2 + i]);
      mpq_mul(c, c, c);
      mpq_add(qF1[d], qF1[d], c);
    }
  }
  int polyID = create_new_constraint();
  for(int i = 0; i < n; i++) {
    add_term(one, p1 + i, p1 + i, polyID);
    add_term(neg1, p1 + i, p2 + i, polyID);
    add_term(neg1, p1 + i, p2 + i, polyID);
    add_term(one, p2 + i, p2 + i, polyID);
  }
  add_term(neg1, d, -1, polyID);
  return d;
}

int ConstraintGenerator::create_product(int a, int b) {
  int ab = create_new_variable();
  if (mode == MODE_P) {
    mpq_t& c = qF1[ab];
    mpq_set_si(c, 1, 1);
    if (a >= 0) {
      mpq_mul(c, c, qF1[a]);
    }
    if (b >= 0) {
      mpq_mul(c, c, qF1[b]);
    }
  }
  int polyID = create_new_constraint();
  add_term(one, a, b, polyID);
  add_term(neg1, ab, -1, polyID);
  return ab;
}




int ConstraintGenerator::create_sum(std::vector<int>& terms) {
  int sum = create_new_variable();
  if (mode == MODE_P) {
    mpq_set_si(qF1[sum], 0, 1);
    for(uint32_t k = 0; k < terms.size(); k++) {
      mpq_add(qF1[sum], qF1[sum], qF1[terms[k]]);
    }
  }
  int sum_test = create_new_constraint();
  for(uint32_t k = 0; k < terms.size(); k++) {
    add_term(one, terms[k], -1, sum_test);
  }
  add_term(neg1, sum, -1, sum_test);
  return sum;
}

void ConstraintGenerator::create_constraint_XleY(int X, int Y, int R, int T) {
  create_constraint_XgeY(Y, X, R, T);
}

void ConstraintGenerator::create_constraint_XltY(int X, int Y, int R, int T) {
  create_constraint_XgtY(Y, X, R, T);
}

void ConstraintGenerator::decompose_exp2bits(mpz_t a_, int bits, int bits_length) {
  //make a copy of a
  mpz_t& a = temp_constraint2;
  mpz_set(a, a_);

  //fill bits + 0 through bits + bits_length - 1 with the exp-2 decomposition of a
  int T = bits_length;
  //Evaluate 2^T
  mpz_t& z = temp_constraint;
  mpz_set_si(z, 1);
  mpz_mul_2exp(z, z, T);

  mpz_t& q = temp_constraint3;
  mpz_t& r = temp_constraint4;
  for(int b = 0; b < T; b++) {
    mpz_tdiv_qr(q, r, a, z);
    if (mpz_sgn(r) == 0) {
      mpq_set_z(qF1[bits + b], z);
      mpz_set(a, q);
    } else {
      mpq_set_ui(qF1[bits + b], 1, 1);
    }
    mpz_tdiv_q_2exp(z, z, 1);
  }
}
void ConstraintGenerator::decompose_bits(mpz_t a_, int bits, int bits_length) {
  //make a copy of a
  mpz_t& a = temp_constraint2;
  mpz_set(a, a_);

  //fill bits + 0 through bits + bits_length - 1 with the binary decomposition of a
  int R = bits_length;
  //Evaluate 2^{R-1}
  mpz_t& z = temp_constraint;
  mpz_set_si(z, 1);
  mpz_mul_2exp(z, z, R-1);

  mpz_t& sub = temp_constraint4;
  for(int b = 0; b < R; b++) {
    mpz_sub(sub, a, z);
    if (mpz_sgn(sub) >= 0) {
      mpq_set_z(qF1[bits + b], z);
      mpz_set(a, sub);
    } else {
      mpq_set_ui(qF1[bits + b], 0, 1);
    }
    mpz_tdiv_q_2exp(z, z, 1);
  }
}

void ConstraintGenerator::create_constraint_is_some_exp2bits_representation(int Xb, int Xb_length) {
  int T = Xb_length;
  //Evaluate 2^T
  mpz_t& z = temp_constraint_upper;
  mpz_t& negz = temp_constraint_upper2;
  mpz_set_si(z, 1);
  mpz_mul_2exp(z, z, T);
  for(int b = 0; b < T; b++) {
    int allowedValueConstraint = create_new_constraint();
    add_term(one, Xb+b, Xb+b, allowedValueConstraint);
    add_term(neg1, Xb+b, -1, allowedValueConstraint);
    mpz_neg(negz, z);
    add_term(negz, Xb+b, -1, allowedValueConstraint);
    add_term(z, -1, -1, allowedValueConstraint);
    mpz_tdiv_q_2exp(z, z, 1);
  }
}

void ConstraintGenerator::create_constraint_is_some_bits_representation(int Xb, int Xb_length) {
  int R = Xb_length;
  //Evaluate 2^{R-1}
  mpz_t& z = temp_constraint_upper;
  mpz_set_si(z, 1);
  mpz_mul_2exp(z, z, R-1);

  for(int b = 0; b < R; b++) {
    int allowedValueConstraint = create_new_constraint();
    add_term(z, Xb+b, -1, allowedValueConstraint);
    add_term(neg1, Xb+b, Xb+b, allowedValueConstraint);
    mpz_tdiv_q_2exp(z, z, 1);
  }
}

void ConstraintGenerator::create_constraint_XgeY(int X, int Y, int R, int T) {
  int XminY_num = create_new_variables(R);
  int XminY_den_min1 = create_new_variables(T);
  if (mode == MODE_P) {
    mpq_t& c = temp_constraintq;
    mpq_sub(c, qF1[X], qF1[Y]);
    if (mpq_sgn(c)==-1) {
      cout << "ERROR: Cheating prover. X is not greater than or equal to Y" << endl;
      return;
    }

    decompose_bits(mpq_numref(c), XminY_num, R);
    mpz_t& den = temp_constraint;
    mpz_sub(den, mpq_denref(c), one);
    decompose_bits(den, XminY_den_min1, T);
  }

  //Check that the binary representations of the num and den are even valid
  create_constraint_is_some_bits_representation(XminY_num, R);
  create_constraint_is_some_bits_representation(XminY_den_min1, T);

  //Check that they represent the correct quantities: x - y = sum(num_b) / (sum(den_b) + 1) <=> sum(den_b) x - sum(den_b) y + x - y - sum(num_b) = 0
  int NumTimesDenCheck = create_new_constraint();
  for(int i = 0; i < T; i++) {
    add_term(one, X, XminY_den_min1 + i, NumTimesDenCheck);
    add_term(neg1, Y, XminY_den_min1 + i, NumTimesDenCheck);
  }
  for(int i = 0; i < R; i++) {
    add_term(neg1, XminY_num + i, -1, NumTimesDenCheck);
  }
  add_term(one, X, -1, NumTimesDenCheck);
  add_term(neg1, Y, -1, NumTimesDenCheck);
}

void ConstraintGenerator::create_constraint_XgtY(int X, int Y, int R, int T) {
  int XminY_num_min1 = create_new_variables(R);
  int XminY_den = create_new_variables(T);
  if (mode == MODE_P) {
    mpq_t& c = temp_constraintq;
    mpq_sub(c, qF1[X], qF1[Y]);
    if (mpq_sgn(c)==-1) {
      cout << "ERROR: Cheating prover. X is not greater than or equal to Y" << endl;
      return;
    }

    mpz_t& num = temp_constraint;
    mpz_sub(num, mpq_numref(c), one);
    decompose_bits(num, XminY_num_min1, R);
    decompose_bits(mpq_denref(c), XminY_den, T);
  }

  //Check that the binary representations of the num and den are even valid
  create_constraint_is_some_bits_representation(XminY_num_min1, R);
  create_constraint_is_some_bits_representation(XminY_den, T);

  //Check that they represent the correct quantities: x - y = (sum(num_b) + 1) / (sum(den_b)) <=> sum(den_b) x - sum(den_b) y - sum(num_b) - 1 = 0
  int NumTimesDenCheck = create_new_constraint();
  for(int i = 0; i < T; i++) {
    add_term(one, X, XminY_den + i, NumTimesDenCheck);
    add_term(neg1, Y, XminY_den + i, NumTimesDenCheck);
  }
  for(int i = 0; i < R; i++) {
    add_term(neg1, XminY_num_min1 + i, -1, NumTimesDenCheck);
  }
  add_term(neg1, -1, -1, NumTimesDenCheck);

}


void ConstraintGenerator::create_constraint_eq_vec(int x, int y, int d) {
  for(int i = 0; i < d; i++) {
    int cstrnt = create_new_constraint();
    add_term(one, x + i, -1, cstrnt);
    add_term(neg1, y + i, -1, cstrnt);
  }
}

void ConstraintGenerator::create_constraint_isBinary(int x) {
  int allowedValueConstraint = create_new_constraint();
  add_term(one, x, -1, allowedValueConstraint);
  add_term(neg1, x, x, allowedValueConstraint);
}

int ConstraintGenerator::create_optional_constraint_neq_vec(int x, int y, int d) {
  int isNeqVec = create_new_variable();
  if (mode == MODE_P) {
    bool same = true;
    for(int i = 0; i < d; i++) {
      same &= (mpq_cmp(qF1[x+i], qF1[y+i]) == 0);
    }
    if (same) {
      mpq_set_ui(qF1[isNeqVec], 1, 1); //Set isNeqVec to false, i.e. 1
    } else {
      mpq_set_ui(qF1[isNeqVec], 0, 1); //Set isNeqVec to true, i.e. 0
    }
  }
  create_constraint_isBinary(isNeqVec);

  bool alreadyFoundDifferent = false;
  int isNeqVecCheck = create_new_constraint();
  //Number of differing dimensions to identify is 1 - qF1[isNeqVec].
  add_term(neg1, isNeqVec, -1, isNeqVecCheck);
  add_term(one, -1, -1, isNeqVecCheck);
  for(int i = 0; i < d; i++) {
    int isNeq = create_optional_constraint_neq(x + i, y + i);
    if (mode == MODE_P) {
      //If we can guarantee more than one, opt out of all but the first such dimension
      if (alreadyFoundDifferent) {
        mpq_set_ui(qF1[isNeq], 1, 1);
      }
      if (mpq_sgn(qF1[isNeq]) == 0) { //They are not equal
        alreadyFoundDifferent = true;
      }
    }
    //check that guaranteedDifferent + sum(optOut_i - 1) = 0
    add_term(one, isNeq, -1, isNeqVecCheck);
    add_term(neg1, -1, -1, isNeqVecCheck);
  }

  return isNeqVec;
}

int ConstraintGenerator::create_optional_constraint_neq(int x, int y) {
  int isNeq = create_new_variable();

  if (mode == MODE_P) {
    if (mpq_cmp(qF1[x], qF1[y]) == 0) {
      mpq_set_si(qF1[isNeq], 1, 1); //They're equal - must opt out (set value to false, or 1)
    } else {
      mpq_set_si(qF1[isNeq], 0, 1); //Not equal, so set the value to true (value = 0)
    }
  }

  create_constraint_isBinary(isNeq);

  //test for existence of an inverse of y - x
  int yMinxInv = create_new_variable();
  if (mode == MODE_P) {
    mpz_t& finv = temp_constraint;
    mpq_t& diff = temp_constraintq;
    mpq_sub(diff, qF1[y], qF1[x]);
    convert_to_z(1, &finv, &diff, *prime);
    mpz_invert(finv, finv, *prime);
    mpq_set_z(qF1[yMinxInv], finv);

    //mpq_mul(diff, qF1[yMinxInv], diff);
    //convert_to_z(1, &finv, &diff, *prime);
    //gmp_printf("%Zd\n", finv);
  }


  //Evaluate inv * (y-x) - 1 explicitly (because optOut can change outside of this method)
  int neqVar = create_new_variable();
  if (mode == MODE_P) {
    mpq_sub(qF1[neqVar], qF1[y], qF1[x]);
    mpq_mul(qF1[neqVar], qF1[neqVar], qF1[yMinxInv]);
    mpq_sub(qF1[neqVar], qF1[neqVar], oneq);
  }
  int neqCheck = create_new_constraint();
  add_term(one, yMinxInv, y, neqCheck);
  add_term(neg1, yMinxInv, x, neqCheck);
  add_term(neg1, -1, -1, neqCheck);
  add_term(neg1, neqVar, -1, neqCheck);

  //Verify that (isNeq = 0) implies (neqVar = 0), i.e. (1 - isNeq) * neqVar
  int optNeqCheck = create_new_constraint();
  add_term(one, isNeq, neqVar, optNeqCheck);
  add_term(neg1, neqVar, -1, optNeqCheck);

  return isNeq;
}

int ConstraintGenerator::create_optional_constraint_segment_not_contains_2d(int e, int x, int R, int T) {
  int notContains = create_new_variable();
  int uaSgn = create_new_variable();
  int uaNum = create_new_variable();
  int uaDen = create_new_variable();
  if (mode == MODE_P) {
    //compute ua numerator
    //compute ua denominator
    //Choose sgn_var to make denominator positive.
    //Now check that the numerator is smaller than the denominator (both are positive) to guarantee ua < 1
    mpq_set_si(qF1[uaDen], 1, 1);
  }

  int checkUANum = create_new_constraint();
  int checkUADen = create_new_constraint();
  create_constraint_isBinary(notContains);

  int ZERO = create_constant(zeroq);

  create_constraint_XgtY(uaDen, ZERO, R, T);
  //If ua <= 0, succeed
  int uaIsNumNeg = 0; //TODO create_optional_constraint_XleY(uaNum, ZERO, R, T);

  //If ua positive, check that num > den

  return notContains;
}

int ConstraintGenerator::create_optional_constraint_segments_intersect_2d(int e, int f, int R, int T) {
  int optOut = create_new_variable();
  create_constraint_isBinary(optOut);
  return optOut;
}
