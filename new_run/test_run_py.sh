#!/bin/bash

./run.py \
  --omp-threads 4 \
  --mpi-cores 3 \
  \
  --temperature 300 \
  --energy 8 \
  --runs 300 \
  --run-time 1100 \
  \
  --results-dir 'new_run/aaa' \
  --input-file 'input_files/fall300_2.input.data' \
