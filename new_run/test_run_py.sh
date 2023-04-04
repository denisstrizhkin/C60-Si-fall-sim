#!/bin/bash

./run.py \
  --omp-threads 0 \
  --mpi-cores 6 \
  \
  --temperature 0 \
  --energy 2 \
  --runs 1 \
  --run-time 1000 \
  \
  --results-dir './res_test' \
  --input-file '../input_files/fall0.input.data' \

