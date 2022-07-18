!#/bin/bash

for i in `seq 10`
do
    julia Julia/linear_regression.jl >> Julia_benchmarks_linear_regression.txt
done