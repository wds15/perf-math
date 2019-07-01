#! /bin/bash

make tbb-scale

REPL=10

echo "1 CORE"
STAN_NUM_THREADS=1 ./tbb-scale --benchmark_repetitions=$REPL

echo "2 CORES"
STAN_NUM_THREADS=2 ./tbb-scale --benchmark_repetitions=$REPL

echo "4 CORES"
STAN_NUM_THREADS=4 ./tbb-scale --benchmark_repetitions=$REPL
