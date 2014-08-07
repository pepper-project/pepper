#define NUM_REPS_PCP 8
#define NUM_REPS_LIN 20

#define NUM_LIN_QUERIES 9
#define NUM_DIV_QUERIES 5

#define NUM_LIN_PCP_QUERIES (NUM_REPS_LIN * NUM_LIN_QUERIES + NUM_DIV_QUERIES)

#define Q1 0
#define Q2 1
#define Q3 2
#define Q4 3
#define Q5 4
#define Q6 5
#define Q7 6
#define Q8 7
#define Q9 8
#define Q10 (NUM_LIN_QUERIES * NUM_REPS_LIN + 0)
#define Q11 (NUM_LIN_QUERIES * NUM_REPS_LIN + 1)
#define Q12 (NUM_LIN_QUERIES * NUM_REPS_LIN + 2)
#define Q13 (NUM_LIN_QUERIES * NUM_REPS_LIN + 3)
#define Q14 (NUM_LIN_QUERIES * NUM_REPS_LIN + 4)

// number of times to run the verification and local computation before
// taking an average
#define NUM_VERIFICATION_RUNS 1
#define NUM_LOCAL_RUNS 1
