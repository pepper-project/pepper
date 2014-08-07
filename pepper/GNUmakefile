#
# This makefile system follows the structuring conventions
# recommended by Peter Miller in his excellent paper:
#
#	Recursive Make Considered Harmful
#	http://aegis.sourceforge.net/auug97.pdf
#
OBJDIR := obj
BINDIR := bin
PTXDIR := ptx
LIBDIR := lib
GENSRCDIR := apps_sfdl_gen
HWSRCDIR := apps_sfdl_hw

include flags

TOP = .

# try to infer the correct GCCPREFIX
ifndef GCCPREFIX
GCCPREFIX := 
endif

CUDA_HARDWARE := nvidia
CUDA_BASE := /usr/local/cuda

CC	:= $(GCCPREFIX)gcc -pipe
CPP	:= $(GCCPREFIX)mpicxx -pipe
AS	:= $(GCCPREFIX)as
AR	:= $(GCCPREFIX)ar rcs
LD	:= $(GCCPREFIX)ld
OBJCOPY	:= $(GCCPREFIX)objcopy
OBJDUMP	:= $(GCCPREFIX)objdump
NM	:= $(GCCPREFIX)nm
NVCC    := $(CUDA_BASE)/bin/nvcc

# Native commands
NCC	:= gcc $(CC_VER) -pipe
TAR	:= gtar
PERL	:= perl
PYTHON := python2

CHECKER := $(PYTHON) cpplint.py --verbose 5 

# Compiler flags
CFLAGS := $(CFLAGS) -I$(TOP) -I$(TOP)/include -MD -m64 
CFLAGS += -Wall -Wno-format -Wno-unused -fopenmp -O2

# temporary directory
CFLAGS += -DFOLDER_STATE=\"/tmp/$(USER)/computation_state\"

# directory of persisted states
CFLAGS += -DFOLDER_PERSIST_STATE=\"/tmp/$(USER)/persist_state\"
CFLAGS += -DFOLDER_TMP=\"/tmp/$(USER)/tmp_state\"

# Add -fno-stack-protector if the option exists.
CFLAGS += $(shell $(CC) -fno-stack-protector -E -x c /dev/null >/dev/null 2>&1 && echo -fno-stack-protector)

CUDA_CFLAGS := --use_fast_math -DMP_USE_64BIT=1 -gencode arch=compute_20,code=sm_20
CUDA_IFLAGS := -I$(CUDA_BASE)/include -I/$(CUDA_BASE)/samples/common/inc

ifeq ($(USE_GPU), 1) 
CUDA_LIBS := -L/usr/lib/nvidia-current-updates -lcudart -lcuda
CFLAGS += -DUSE_GPU=1
else
CFLAGS += -DUSE_GPU=0
endif

CFLAGS += -DNUM_PROVERS_MAX=$(NUM_PROVERS_MAX)

ifdef COMPILE_DEV
CFLAGS += -g -rdynamic
endif

ifeq ($(USE_DB_BLOCK_STORE), 1)
CFLAGS += -DUSE_DB_BLOCK_STORE
endif

# DB flags
ifeq ($(DB_HASH_FUNC),null)
CFLAGS += -DDB_HASH_FUNC_IS_NULL
endif
CFLAGS += -DDB_NUM_ADDRESSES=$(DB_NUM_ADDRESSES) -DDB_HASH_NUM_BITS=$(DB_HASH_NUM_BITS)
CFLAGS += -DRAM_CELL_NUM_BITS=$(RAM_CELL_NUM_BITS)

CFLAGS += -DFAST_RAM_WORD_WIDTH=$(FAST_RAM_WORD_WIDTH)
CFLAGS += -DFAST_RAM_ADDRESS_WIDTH=$(FAST_RAM_ADDRESS_WIDTH)

# Common linker flags
LDFLAGS := -L/usr/local/lib -lm -lgmp -lgmpxx -lgomp -lconfig++ -lgmpmee -lntl -lrt -lfcgi -lpbc -lkyotocabinet
LDFLAGS += -L/usr/local/lib/chacha -lchacha -lpapi -lleveldb -L/usr/local/lib/libzm -lzm
ifeq ($(BUILD_MYSQL_MICRO), 1)
LDFLAGS += -L/usr/local/mysql/lib -lmysqlclient
endif
# LDFLAGS += -lcurl
# LDFLAGS += -ldb_cxx

# common include flags
IFLAGS := -I/usr/include -I/usr/local/include -pthread
IFLAGS += -I/usr/local/include/chacha
IFLAGS += -I/usr/local/include/pbc
#IFLAGS += -I/usr/local/BerkeleyDB.6.0/include
IFLAGS += -I/usr/local/include/libzm

ifeq ($(BUILD_MYSQL_MICRO), 1)
IFLAGS += -I/usr/local/mysql/include/
endif

# Needed flag for libzm
CFLAGS += -DMIE_ATE_USE_GMP

ifeq ($(USE_GPU), 1) 
CUDA_LDFLAGS := -L/usr/local/cuda/lib64 -L/usr/lib/nvidia-current-updates -lcudart -lcuda -lssl -lcrypto
endif

GCC_LIB := $(shell $(CC) $(CFLAGS) -print-libgcc-file-name)

# Lists that the */Makefrag makefile fragments will add to
OBJDIRS :=

# Make sure that 'all' is the first target
all:

# Target to build and run test code
test: 

# Eliminate default suffix rules
.SUFFIXES:

# Delete target files if there is an error (or make is interrupted)
.DELETE_ON_ERROR:

# make it so that intermediate .o files and generated source files are not deleted
.PRECIOUS: %.o $(OBJDIR)/%.o $(GENSRCFILES) 

# Allow use of % in wildcards, using $$wildcard()
.SECONDEXPANSION:

# Set to nothing (i.e., V = ) to enable verbose outputs.
V = @

MAKEFILE_DEP_LINE = $@: GNUmakefile $(wildcard $(dir $<)Makefrag)

# Include Makefrags for subdirectories
include common/Makefrag
include crypto/Makefrag
include libv/Makefrag
include storage/Makefrag
include include/Makefrag
include misc/Makefrag
include net/Makefrag
include apps_tailored/Makefrag
ifeq ($(USE_GPU), 1)
include lib_gpu/Makefrag
include lib_gpu_tests/Makefrag
include include/Makefrag
endif
ifneq ($(BUILD_CMT), 0)
include cmtgkr/Makefrag
endif
include apps_sfdl/Makefrag

APPS_TAILORED_BINS = $(patsubst apps_tailored/%.cpp, $(BINDIR)/%, $(EXEC_SRCFILES))
APPS_SFDL_BINS = $(patsubst %, $(BINDIR)/%, $(COMPUTATIONS))
MISC_BINS = $(patsubst misc/%.cpp, $(BINDIR)/%, $(MISC_SRCFILES))

COMMON_LIB_OBJFILES = \
	$(COMMON_OBJFILES) \
	$(CRYPTO_OBJFILES) \
	$(LIBV_OBJFILES) \
	$(MERKLE_OBJFILES) \
	$(INCLUDE_OBJFILES) \
	$(PROVER_OBJFILES) \
	$(VERIFIER_OBJFILES)

# How to build apps
$(APPS_TAILORED_BINS) : $(BINDIR)/% : $(OBJDIR)/apps_tailored/%.o $(COMMON_LIB_OBJFILES) $(APPS_TAILORED_OBJFILES)
	@mkdir -p $(@D)
	@echo + mk $@
	$(V)$(CPP) $(CFLAGS) $(IFLAGS) $(CUDA_IFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)

$(MISC_BINS) : $(BINDIR)/% : $(OBJDIR)/misc/%.o $(COMMON_LIB_OBJFILES) $(MISC_OBJFILES)
	@mkdir -p $(@D)
	@echo + mk $@
	$(V)$(CPP) $(CFLAGS) $(IFLAGS) $(CUDA_IFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)


$(APPS_SFDL_BINS) : $(BINDIR)/% : $(EXAMPLES_OBJFILES) $(OBJDIR)/$(GENSRCDIR)/%.o $(COMMON_LIB_OBJFILES) $(APPS_SFDL_OBJFILES) 
	@mkdir -p $(@D)
	@echo + mk $@
	$(V)$(CPP) $(CFLAGS) $(IFLAGS) $(CUDA_IFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)

all: $(APPS_SFDL_BINS) $(APPS_TAILORED_BINS) $(MISC_BINS)

# Run the code generation only (don't compile all the way to executables)
gensrc: $(GENSRCFILES)

computation_state:
	mkdir -p $@

run-%: $(BINDIR)/%_v $(BINDIR)/%_p computation_state
	rm -rf computation_state/*
	./run/$*.sh $(ARGS)

#
# This target informs make how our compiler works. It produces
# specially-named cpp and h files in $(GENSRCDIR) using .sfdl files.
# HANDWRITTEN FILES DO NOT GO ON THIS LIST!!!
#
$(GENSRCDIR)/%.cpp \
$(GENSRCDIR)/%_v.h \
$(GENSRCDIR)/%_v.cpp \
$(GENSRCDIR)/%_p.h \
$(GENSRCDIR)/%_p.cpp \
$(GENSRCDIR)/%_cons.h \
$(GENSRCDIR)/%_cons.cpp : apps_sfdl/%.sfdl
	@echo + compile $< $(ZCC_CARGS)
	@mkdir -p $(GENSRCDIR)
	@mkdir -p $(BINDIR)
	@cd ../compiler/; \
	./zcc -f ../pepper/$< \
		      -d ../pepper \
		      $(ZCC_CARGS) \
		      -prefix $(patsubst apps_sfdl/%.sfdl,$(GENSRCDIR)/%,$<) \

#
# Same as above, but for .c file compilation
# HANDWRITTEN FILES DO NOT GO ON THIS LIST!!!
#
$(GENSRCDIR)/%.cpp \
$(GENSRCDIR)/%_v.h \
$(GENSRCDIR)/%_v.cpp \
$(GENSRCDIR)/%_p.h \
$(GENSRCDIR)/%_p.cpp \
$(GENSRCDIR)/%_cons.h \
$(GENSRCDIR)/%_cons.cpp \
$(GENSRCDIR)/%_v_inp_gen.h \
$(GENSRCDIR)/%_v_inp_gen.cpp : apps_sfdl/%.c apps_sfdl/*.h ../compiler/cstdinc/*
	@echo + compile $< $(ZCC_CARGS)
	@mkdir -p $(GENSRCDIR)
	@mkdir -p $(BINDIR)
	@cd ../compiler/; \
	./zcc -f ../pepper/$< \
		      -d ../pepper \
		      $(ZCC_CARGS) \
		      -prefix $(patsubst apps_sfdl/%.c,$(GENSRCDIR)/%,$<) \

$(OBJDIR)/%.o: %.cu
	@echo + nvcc $<
	@mkdir -p $(@D)
	$(V)$(NVCC) $(CUDA_CFLAGS) $(CUDA_IFLAGS) -c $< -o $@
	$(V)$(NVCC) $(CUDA_CFLAGS) $(CUDA_IFLAGS) -M $< | sed 's/^$(notdir $*).o/$(subst /,\/,$@)/g' > $(@:o=d)
	$(V)echo "$(MAKEFILE_DEP_LINE)" >> $(@:o=d)

$(OBJDIR)/%.o: %.cpp
	@echo + cpp $<
	@mkdir -p $(@D)
	$(V)$(CPP) $(CFLAGS) -DMP_USE_64BIT $(IFLAGS) $(CUDA_IFLAGS) -c $< -o $@
	$(V)echo "$(MAKEFILE_DEP_LINE)" >> $(@:o=d)

# How to build vanilla binaries.
$(BINDIR)/%: $(OBJDIR)/%.o
	@echo + mk $@
	@mkdir -p $(@D)
	$(V)$(CPP) $(CFLAGS) $(IFLAGS) $(CUDA_IFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
	
# For running binaries.
run-%: $(BINDIR)/%
	./$(BINDIR)/$*

# For deleting the build
clean: clean-gensrc
	rm -rf $(LIBDIR) $(OBJDIR) $(BINDIR) $(PTXDIR) computation_state/*
	rm -f ../compiler/*circuit*
	rm -f ../compiler/*spec*
	rm -f ../compiler/*.c
	rm -f ../compiler/*.defines
	rm -f ../compiler/backend/*.pyc
	rm -f ../compiler/backend/tmptime
	rm -f ../compiler/tmptime
	cd ../compiler/frontend; make clean; 

clean-gensrc: 
	rm -rf $(GENSRCDIR)
	
# This magic automatically generates makefile dependencies
# for header files included from C source files we compile,
# and keeps those dependencies up-to-date every time we recompile.
# See 'mergedep.pl' for more information.
$(OBJDIR)/.deps: $(foreach dir, $(OBJDIRS), $(wildcard $(OBJDIR)/$(dir)/*.d))
	@mkdir -p $(@D)
	@$(PERL) mergedep.pl $@ $^

-include $(OBJDIR)/.deps

always: 
	@:

.PHONY: always
