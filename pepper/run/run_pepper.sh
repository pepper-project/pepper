#!/bin/bash
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 [program_name]" >&2
  exit 1
fi

SRC_FILE=$1
GENERATE_STATES="--gen-states 1"
SHARED_BSTORE_PATH="default_shared_db"
export LD_LIBRARY_PATH=${HOME}/pepper_deps/lib:${LD_LIBRARY_PATH}


echo "LOG: Building executables"
make SFDL_FILES="" C_FILES="$SRC_FILE" BUILD_CMT=0 BUILD_MIP=0
make SFDL_FILES="" C_FILES="$SRC_FILE" BUILD_CMT=0 BUILD_MIP=0

echo "LOG: Running $SRC_FILE"
shift
for (( i=0; i<$#; i++ ))
do
  case "$1" in
    "--no-gen-states")
      GENERATE_STATES="--gen-states 0"
      PRESERVE_INPUT="--no-gen-states"
      ;;
    "--gen-states")
      GENERATE_STATES="--gen-states 1"
      ;;
    "--shared-bstore-path")
      SHARED_BSTORE_PATH=$2
      ;;
  esac
  shift
done

if [ "$GENERATE_STATES" == "--gen-states 1" ]; then
  echo "LOG: Will generate new states if any."
elif [ "$GENERATE_STATES" == "--gen-states 0" ]; then
  echo "LOG: Will use existing states. (default)"
fi
SHARED_BSTORE_PATH_FLAG="--shared-bstore-path $SHARED_BSTORE_PATH"
echo "LOG: Using shared store named $SHARED_BSTORE_PATH"


./run/prepare.sh $PRESERVE_INPUT
mpirun -np 2 ./bin/$SRC_FILE -p 1 -b 1 -r 1 -i 10 -v 0 $GENERATE_STATES $SHARED_BSTORE_PATH_FLAG

rm -rf temp_block_store*



