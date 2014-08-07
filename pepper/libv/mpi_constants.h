#ifndef CODE_PEPPER_LIBV_MPI_CONSTANTS_H_
#define CODE_PEPPER_LIBV_MPI_CONSTANTS_H_

#define MPI_COORD_RANK 1

// Used as tags.

#define MPI_PARAMS 0
#define MPI_INVOKE_PROVER 1
#define MPI_PROVER_FINISHED 2
#define MPI_TERMINATE 3
#define MPI_FILE_RECV 4
#define MPI_QUERY_CREATED 5

// This value should be higher than all other tags.
#define MPI_FILE_SEND 100

#endif
