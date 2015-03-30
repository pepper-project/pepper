#!/bin/bash

rsltdir=cmt_gkr_dat
cpu=yes
gpu=yes
export LD_LIBRARY_PATH=${HOME}/pepper_deps/lib:${LD_LIBRARY_PATH}

if [ $# -gt 0 ]
then
    if [ $1 = data ]
    then
        gpu=
    elif [ $1 = gpu ]
    then
        cpu=
    fi
fi

run_bin() {
    local bin outfile rslt ngates

    bin=$1
    outfile=$4

    rslt=`${bin} $2 $3`
    ngates=`echo -n "${rslt}" | grep 'num_gates is: [0-9]*$' | grep -o '[0-9]*$'`

    echo -n 'Par: '
    echo -n "$i   " | tee -a ${outfile}
    echo -n "${rslt}" | egrep -e '^[0-9]+[[:space:]]+[0-9]' | tr -d "\n" | tee -a ${outfile}
    echo -e "\t${ngates}" | tee -a ${outfile}
}

run() {
    local nameseq namepar rslt ngates rstldir

    rsltdir=$4

    mkdir -p ${rsltdir}

    nameseq=${rsltdir}/seq_$1_logn_2_$2.rslt
    namepar=${rsltdir}/par_$1_logn_2_$2.rslt

    echo '===================='
    echo "Begin test for $1."
    echo "Going from 2 (cause the code segfaults at 1) to $2."
    echo '===================='

    echo -e "Iter N\tProveT\tVerT\tProofS\tRounds\tngates" > ${nameseq}
    echo -e "Iter N\tProveT\tVerT\tProofS\tRounds\tngates" > ${namepar}

    echo -e "Iter N\tProveT\tVerT\tProofS\tRounds\tngates"
    echo '---------'

    for i in `seq -w 2 $2`
    do
        test ${gpu} && run_bin ./bin/parallelGKR $i $3 ${namepar}
        test ${cpu} && run_bin ./bin/sequentialGKR $i $3 ${nameseq}
        echo '---------'
    done
}

# The interface is:
#   run <name> <log2 n> <computation code>
# n is the input size.
# The computation codes are:
#   - 0 - F2
#   - 1 - F0
#   - 2 - MatMult
#   - 3 - PM
#run F2 20 0 ${rsltdir}
#run F0 18 1 ${rsltdir}
run MatMult 5 2 ${rsltdir}
#run PM 16 3 ${rsltdir}

