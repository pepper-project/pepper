#define NUM_REPS_PCP 8
#define NUM_REPS_LIN 20

#define NUM_LIN_QUERIES 6
#define NUM_DIV_QUERIES 4

#define NUM_LIN_PCP_QUERIES (NUM_REPS_LIN * NUM_LIN_QUERIES + NUM_DIV_QUERIES)

#define QUERY1 0
#define QUERY2 1
#define QUERY3 2
#define QUERY4 3
#define QUERY5 4
#define QUERY6 5
#define QUERY7 (NUM_LIN_QUERIES * NUM_REPS_LIN + 0)
#define QUERY8 (NUM_LIN_QUERIES * NUM_REPS_LIN + 1)
#define QUERY9 (NUM_LIN_QUERIES * NUM_REPS_LIN + 2)
#define QUERY10 (NUM_LIN_QUERIES * NUM_REPS_LIN + 3)

// number of times to run the verification and local computation before
// taking an average
#define NUM_VERIFICATION_RUNS 100
#define NUM_LOCAL_RUNS 100
