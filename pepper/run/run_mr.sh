#!/bin/bash
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 [program_name]" >&2
  exit 1
fi

export LD_LIBRARY_PATH=${HOME}/pepper_deps/lib:${LD_LIBRARY_PATH}

echo "LOG: Building executables"
make SFDL_FILES="" C_FILES="$1_map $1_red" BUILD_MIP=0 BUILD_CMT=0 

# determine the number of mappers to run
#num_map=`grep "NUM_MAPPERS" apps_sfdl/$1_map.c | awk '{print $3}'`
#num_red=`grep "NUM_REDUCERS" apps_sfdl/$1_map.c | awk '{print $3}'`

num_map=`grep "NUM_MAPPERS" apps_sfdl/$1.h | awk '{print $3}'`
num_red=`grep "NUM_REDUCERS" apps_sfdl/$1.h | awk '{print $3}'`
./run/prepare.sh

echo "-----------------------------------------------------"
echo "LOG: Running the mapper batch $1_map"
mpirun -np 2 ./bin/$1_map -b $num_map

echo "-----------------------------------------------------"
echo "LOG: Running the reducer batch $1_red"
mpirun -np 2 ./bin/$1_red -b $num_red

rm -rf temp_block_store* 
