#!/bin/bash
PRESERVE_INPUT=$1

mkdir -p /tmp/$USER/
if [ "$PRESERVE_INPUT" != "--no-gen-states" ]; then
  echo "LOG: Removing states."
  rm -rf /tmp/$USER/computation_state/*
fi
