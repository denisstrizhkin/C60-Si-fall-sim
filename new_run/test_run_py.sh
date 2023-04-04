#!/bin/bash

./run.py \
  --omp-threads 0 \
  --mpi-cores 6 \
  \
  --temperature 0 \
  --energy 2 \
  --runs 10 \
  --run-time 10000 \
  \
  --results-dir './res_test' \
  --input-file '../input_files/fall0_2.input.data' \
