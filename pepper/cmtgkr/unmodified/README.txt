This document describes source code for the paper Verifiable Computation with Massively Parallel Interactive Proofs
by Justin Thaler, Mike Roberts, Michael Mitzenmacher, and Hanspeter Pfister. 

There are 4 separate .cc files, each of which is self-contained and can be compiled to an executable.
The .cc programs can be compiled as follows. Run the command:

g++ -o ex_name prog.cc -O3

where ex_name is the desired name for the executable, and prog.cc is the program you wish to compile.

The .cu programs contain GPU code in CUDA and Thrust. They can be compiled as follows, assuming Thrust is installed and 
nvcc is working. Run the command:
nvcc -o ex_name prog.cu -arch=sm_20 

(The -arch=sm_20 flag is only necessary for the program parallelF2.cu, not for parallelGKR.cu).
---------------------------------
Description of the programs:

sequentialGKR.cc contains a sequential implementation of the GKR protocol due to Cormode, Thaler, and Mitzenmacher. We have extended
this program to support matrix multiplication.

sequentialF2.cc is an implementation of the FFT-based prover for the specialized non-interactive protocol for F2 due to Cormode, Mitzenmacher, and Thaler.

parallelGKR.cu is a GPU-based implementation of the GKR protocol.

parallelF2.cu is a GPU-based implementation of the non-interactive F2 protocol.
------------------------
Usage for each program: 

The executables for sequentialF2.cc and parallelF2.cu should take two command line arguments: the desired length of the proof h, and the desired space usage of the verifier v. 

The executables for sequentialGKR.cc and parallelGKR.cu should take two command line arguments. The first specifies log(input size). The second specifies which protocol to run. 0 for F2, 1 for F0, 2 for matrix multiplication, and 3 for pattern matching. The default pattern length for the latter is 8.
