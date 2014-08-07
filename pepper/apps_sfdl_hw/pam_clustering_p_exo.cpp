#include <apps_sfdl_hw/pam_clustering_p_exo.h>
#include <apps_sfdl_gen/pam_clustering_cons.h>

//This file will NOT be overwritten by the code generator, if it already
//exists. make clean will also not remove this file.

pam_clusteringProverExo::pam_clusteringProverExo() {
}

//Euclidean distance
static void getDistance(mpq_t target, const mpq_t* a, const mpq_t* b, mpq_t* tmps) {
  int d = pam_clustering_cons::d;
  mpq_set_ui(target, 0, 1);
  for(int i = 0; i < d; i++) {
    mpz_sub(mpq_numref(tmps[0]), mpq_numref(a[i]), mpq_numref(b[i]));
    mpz_mul(mpq_numref(tmps[0]), mpq_numref(tmps[0]), mpq_numref(tmps[0]));
    mpz_add(mpq_numref(target), mpq_numref(target), mpq_numref(tmps[0]));
  }
}

static void getPointCost(mpq_t target, const mpq_t* a, const mpq_t* A, mpq_t* tmps) {
  int k = pam_clustering_cons::k;
  int d = pam_clustering_cons::d;

  getDistance(target, a, A + 0, tmps+1);
  for(int i = 1; i < k; i++) {
    getDistance(tmps[0], a, A + i*d, tmps+1);
    if (mpz_cmp(mpq_numref(tmps[0]), mpq_numref(target)) < 0) {
      mpz_set(mpq_numref(target), mpq_numref(tmps[0]));
    }
  }
}

static void getCost(mpq_t target, const mpq_t* A, mpq_t* tmps) {
  int m = pam_clustering_cons::m;
  int d = pam_clustering_cons::d;

  mpq_set_ui(target, 0, 1);
  for(int i = 0; i < m; i++) {
    getPointCost(tmps[0], A + i*d, A, tmps+1);
    mpz_add(mpq_numref(target), mpq_numref(target), mpq_numref(tmps[0]));
  }
}

//Refer to apps_sfdl_gen/pam_clustering_cons.h for constants to use in this exogenous
//check.
void pam_clusteringProverExo::baseline(const mpq_t *input_q, int num_inputs, mpq_t *c, int num_outputs) {
  int m = pam_clustering_cons::m;
  int k = pam_clustering_cons::k;
  int d = pam_clustering_cons::d;
  int L = pam_clustering_cons::L;

  mpq_t* tmps;
  alloc_init_vec(&tmps, 15);

  mpq_t* backup;
  alloc_init_vec(&backup, d);

  mpq_t* A;
  alloc_init_vec(&A, d*m);
  for(int i = 0; i < d*m; i++) {
    mpz_set(mpq_numref(A[i]), mpq_numref(input_q[i]));
  }

  //Randomly choose medoids
  //Currently, just set the medoids to be the first k datapoints

  //Get initial cost
  getCost(tmps[0], A, tmps+1);

  //Swap phase
  for(int iter = 0; iter < L; iter++) {
    for(int i = 0; i < k; i++) {
      //Swap medoid i with a nonmedoid point
      for(int j = k; j < m; j++) {

        //swap A[j] to A[i]
        for(int u = 0; u < d; u++) {
          mpz_set(mpq_numref(backup[u]), mpq_numref(A[i*d + u]));
          mpz_set(mpq_numref(A[i*d + u]), mpq_numref(A[j*d + u]));
          mpz_set(mpq_numref(A[j*d + u]), mpq_numref(backup[u]));
        }
        //Get new cost
        getCost(tmps[1], A, tmps+2);

        //Is it an improvement?
        if (mpz_cmp(mpq_numref(tmps[1]), mpq_numref(tmps[0])) < 0) {
          mpz_set(mpq_numref(tmps[0]), mpq_numref(tmps[1]));
        } else {
          //Restore backup
          //swap A[j] to A[i]
          for(int u = 0; u < d; u++) {
            mpz_set(mpq_numref(backup[u]), mpq_numref(A[i*d + u]));
            mpz_set(mpq_numref(A[i*d + u]), mpq_numref(A[j*d + u]));
            mpz_set(mpq_numref(A[j*d + u]), mpq_numref(backup[u]));
          }
        }
      }
    }
  }

  //Classify points (Note - a medoid may end up in a different
  //cluster than its own, if two medoids have the same coordinates)

  for(int i = 0; i < m; i++) {
    mpq_set_ui(c[i], 0, 1);
    getDistance(tmps[0], A + 0, A + i*d, tmps+2);
    for(int j = 1; j < k; j++) {
      getDistance(tmps[1], A + j*d, A + i*d, tmps+2);
      if (mpz_cmp(mpq_numref(tmps[1]), mpq_numref(tmps[0])) < 0) {
        mpz_set(mpq_numref(tmps[0]), mpq_numref(tmps[1]));
        mpz_set_ui(mpq_numref(c[i]), j);
      }
    }
    //gmp_printf("Point %d at %Qd %Qd goes to %Qd %Qd, distance %Qd, should have been %Qd \n", i, A + i*d + 0, A + i*d + 1, A[c[i]*d] + 0, A[c[i]*d] + 1, tmps[0], output_q[i]);
  }
}

bool pam_clusteringProverExo::exogenous_check(const mpz_t* input, const mpq_t* input_q,
    int num_inputs, const mpz_t* output, const mpq_t* output_q, int num_outputs, mpz_t prime) {
#ifdef ENABLE_EXOGENOUS_CHECKING
  int m = pam_clustering_cons::m;
  int k = pam_clustering_cons::k;
  int d = pam_clustering_cons::d;
  int L = pam_clustering_cons::L;

  bool success = true;

  mpq_t *output_recomputed;
  alloc_init_vec(&output_recomputed, num_outputs);
  baseline(input_q, num_inputs, output_recomputed, num_outputs);

  for (int i=0; i<num_outputs; i++) {
    success &= (1-mpq_cmp(output_q[i], output_recomputed[i]));
  }

  return success;
#else
  return true;
#endif
};

