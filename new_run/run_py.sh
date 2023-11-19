#!/bin/bash

./run.py \
  --omp-threads 0 \
  --mpi-cores 8 \
  \
  --temperature "1e-6" \
  --energy 14 \
  --runs 50 \
  --run-time 1005 \
  \
  --results-dir 'new_run/test_size_20_20_14keV' \
  --input-file 'input_files/fall_0K_20_20_31.input.data' \
