#define NUM_REPS_PCP 8 
#define NUM_REPS_LIN 25

#define NUM_LIN_QUERIES 15
#define NUM_DIV_QUERIES 7 

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
#define Q10 9 
#define Q11 10
#define Q12 11
#define Q13 12
#define Q14 13
#define Q15 14
#define Q16 (NUM_LIN_QUERIES * NUM_REPS_LIN + 0)
#define Q17 (NUM_LIN_QUERIES * NUM_REPS_LIN + 1)
#define Q18 (NUM_LIN_QUERIES * NUM_REPS_LIN + 2)
#define Q19 (NUM_LIN_QUERIES * NUM_REPS_LIN + 3)
#define Q20 (NUM_LIN_QUERIES * NUM_REPS_LIN + 4)
#define Q21 (NUM_LIN_QUERIES * NUM_REPS_LIN + 5)
#define Q22 (NUM_LIN_QUERIES * NUM_REPS_LIN + 6)

// number of times to run the verification and local computation before
// taking an average
#define NUM_VERIFICATION_RUNS 1
#define NUM_LOCAL_RUNS 1
