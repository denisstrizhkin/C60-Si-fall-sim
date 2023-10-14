#!/bin/bash

./run.py \
  --omp-threads 0 \
  --mpi-cores 10 \
  \
  --temperature 0 \
  --energy 8 \
  --runs 5 \
  --run-time 5000 \
  \
  --results-dir 'new_run/test_test_multik_tersoff_single_8keV_0K_24_24_31' \
  --input-file 'input_files/fall_0K_24_24_31.input.data' \
