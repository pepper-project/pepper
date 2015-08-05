This source code is released under a BSD-style license. See LICENSE
for more details.

I. Installation and first steps

  1. Please run
    git submodule init
    git submodule update
  to pull in the submodules this project relies on.
  
  2. This codebase depends on several external libraries. Please
  see the INSTALLING.md file for more information on setting up
  dependencies.

  3. See compiler/buffetfsm/README.md for instructions on building
  the patched Clang/LLVM libraries we need for Buffet's C-to-C
  compiler.

  4. We recommend using the libsnark backend. If you've pulled in the
  submodules (Step 1, above), you've already got a local copy. Finish
  preparing it as follows:
    cd libsnark
    ./prepare-depends.sh
    make lib STATIC=1 NO_PROCPS=1
  and make sure that USE_LIBSNARK=1 is set in the pepper/flags file.

  5. Now that everything is set up, you're ready to run some verified
  computations! Please have a look at GETTINGSTARTED.md for a quick
  overview of the process.

  6. If you want to run tinyram programs, make sure you've pulled in
  the submodules as described above, then cd `tinyram/doc`. If you have
  pandoc and LaTeX available, `make` will generate useful documentation.
  The README.md file in that directory has instructions on how to run
  tinyram computations.

II. Configuration

  - To select different verifiable computation protocols (Zaatar,
    Ginger, Pinocchio, etc.), please see pepper/common/utility.h.
    Note that when USE_LIBSNARK=1 is set in pepper/flags, the
    correct backend is selected automatically.
  
  - The compiler generates two types of constraints depending on the
    value of FRAMEWORK in the makefile
    
    - FRAMEWORK should be set to ZAATAR for using Zaatar's and
      Pinocchio's verification machinery. This is the default
      setting, and the correct one to use with the libsnark,
      Pinocchio, and Zaatar backends.

    - FRAMEWORK should be set to GINGER to use ginger's verification
      machinery.

III. What is new in this version?

  (1) Updated code which interfaces with libsnark. Our fork of
      libsnark is now identical to the latest available version from
      https://github.com/scipr-lab/libsnark, except for a
      configuration option set in the Makefile.

  (2) Minor bugfix.

IV. Contact
Please contact
    srinath at cs dot utexas dot edu
or
    rsw at cs dot nyu dot edu
for any questions and comments.
