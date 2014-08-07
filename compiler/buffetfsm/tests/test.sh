#!/bin/bash

gcc -E -P "$1" | ../buffetfsm -
