#!/bin/bash

run_benchmark() {
    threads=$1
    echo "RUNNING BENCHMARK WITH NUMBER OF THREADS: ${threads}"
    out=parallel_threads_${threads}.csv
    STAN_NUM_THREADS=$threads ./parallel-v3 --benchmark_repetitions=4 --benchmark_out=${out}_tmp --benchmark_out_format=csv
    tail -n +9 ${out}_tmp | grep -v "_mean" | grep -v "_median" | grep -v "_stddev" > ${out}
    rm ${out}_tmp
}

run_benchmark 6
run_benchmark 4
run_benchmark 2
run_benchmark 1
