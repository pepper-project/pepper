#!/bin/bash
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <program_name> [options]" >&2
  exit 1
fi

comp=$1
shift

echo "LOG: Building executables."
make SFDL_FILES="" C_FILES="${comp}_map ${comp}_red" BUILD_CMT=0 BUILD_MIP=0 || exit 1
make "bin/mip/apps/${comp}_mip" BUILD_CMT=0 BUILD_MIP=1 || exit 1

# determine the number of mappers to run
num_map=`grep "#define NUM_MAPPERS" apps_sfdl/${comp}_map.c | awk '{print $3}'`
num_red=`grep "#define NUM_REDUCERS" apps_sfdl/${comp}_map.c | awk '{print $3}'`
size_input_red=`grep "SIZE_INPUT" apps_sfdl/${comp}_red.c | awk '{print $3}'`
size_input_map=`grep "#define SIZE_INPUT" apps_sfdl/${comp}_map.c | awk '{print $3}'`

./run/prepare.sh

#echo "-----------------------------------------------------"
#echo "LOG: Running the mapper batch $1_map"
#mpirun -np 2 ./bin/$1_map -b $num_map
#
#echo "-----------------------------------------------------"
#echo "LOG: Running the reducer batch $1_red"
#mpirun -np 2 ./bin/$1_red -b $num_red

echo "LOG: Starting MapReduce computation: ${comp}."
mpirun -np 3 "./bin/mip/apps/${comp}_mip" $@ -s "${comp}" "bin/${comp}" ${num_map} ${num_red}

rm -rf temp_block_store* 
