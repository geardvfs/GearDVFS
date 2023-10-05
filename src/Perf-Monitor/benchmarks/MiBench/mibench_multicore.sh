#!/bin/bash
end=$((SECONDS+2))
n_core=$2
name=$1
n=0
while [ $SECONDS -lt $end ]; do
    # Do what you want.
    for ((i=0; i<=$n_core; i++))
    do
      taskset -c $i $name 1 &
      pids[${i}]=$!
    done
    for pid in ${pids[*]}; do
       wait $pid
    done
    let n++
done
echo "Event Counts of Benchmark $name: $n"
