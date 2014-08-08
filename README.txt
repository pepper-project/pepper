This source code is released under a BSD-style license. See LICENSE
for more details.

I. Installation
  
  This codebase requires the following dependent packages. Get in touch
  with us if you face issues (see the contact information below)

  1. gmp with C++ extensions enabled
  2. PAPI
  3. Chacha pseudorandom number generator (use the assembly version for
  high performance)
  4. NTL with NTL_GMP_LIP=on 
  5. libconfig
  6. Cheetah template library
  7. OpenMPI
  8. CUDA packages (if you would like your installation to use GPUs,
  which would require setting USE_GPU to 1 in the makefile)
  9. GMPMEE 
 10. go
 11. python
 12. OpenSSL
 13. fcgi
 14. PBC (from Stanford)
 15. ant
 16. boost
 17. LevelDB
 18. KyotoCabinet
 19. libzm (from https://github.com/herumi/ate-pairing.git)
 20. Java (1.7 or higher)
 21. LLVM/Clang. (See compiler/buffetfsm/README.md for build information.)

II. Configuration

  - To select different verifiable computation protocols (Zaatar,
    Ginger, Pinocchio, etc.), please see common/utility.h
  
  - The compiler generates two types of constraints depending on the
    value of FRAMEWORK in the makefile
    
    - FRAMEWORK should be set to ZAATAR for using Zaatar's and
      Pinocchio's verification machinery.

    - FRAMEWORK should be set to GINGER to use ginger's verification
      machinery.

III. Running examples
  
  (1) With a single machine, for testing purposes.

    Suppose the computation is pepper/apps_sfdl/mm.c, then
    use the following commands.

    cd pepper
    ./run/run_pepper.sh mm
    
    (Note: The first time a new computation is compiled, the above
    command will fail with "No target to compile ... _p_exo.o". In that
    case, re-run the above command and it should work.)
    
  (2) With a cluster, for experiments.

    Please refer to
    http://www.tacc.utexas.edu/user-services/user-guides/longhorn-user-guide
    for launching a job on TACC. 

IV. What is new in this version?
  
  (1) a better implementation of RAM operations by adapting
  ideas from TinyRAM
    
    - For an example of this, see apps_sfdl/fast_ram_test.c

  (2) a C-to-C compiler for transforming data dependent loops into finite
  state machines amenable to efficient compilation. Automatically invoked
  when the source text uses the "buffet::fsm" C++11-style attribute.

  NOTE that you will need to build a local copy of LLVM and Clang with
  patches enabling this new attribute. See compiler/buffetfsm/README.md
  for instructions on how to do this.

V. Contact
Please contact
    srinath at cs dot utexas dot edu
or
    rsw at cs dot nyu dot edu
for any questions and comments.
