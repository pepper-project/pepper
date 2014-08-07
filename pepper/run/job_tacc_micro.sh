#!/bin/bash       
#$ -V    # Inherit the submission environment
#$ -cwd  # Start job in submission directory
#$ -N micro # Job Name
#$ -j y  # Combine stderr and stdout
#$ -o $JOB_NAME.o$JOB_ID         # Name of the output file (eg.  myMPI.oJobID)
#$ -pe 1way 8    # Requests 1 tasks/node, 8 cores total
#$ -q normal     # Queue name "normal"
#$ -l h_rt=00:10:00      # Run time (hh:mm:ss)
#$ -M mailtosrinath@gmail.com    # Use email notification address
#$ -m be         # Email at Begin and End of job
#$ -P data 
#$ -A Pepper 
set -x   # Echo commands, use "set echo" with csh

module use ~/modulefiles
module load cuda
module load cuda_SDK
module load papi
module load gmp
module load chacha
module load encrypt
module load gmp
module load openssl

module load cuda
module load cuda_SDK
ibrun -n 1 -o 0 /home/01934/srinath/MICRO/vercomp/code/pepper/bin/micro2
