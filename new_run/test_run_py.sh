#!/bin/bash

./run.py \
  --omp-threads 0 \
  --mpi-cores 10 \
  \
  --temperature 0 \
  --energy 8 \
  --runs 50 \
  --run-time 5000 \
  \
  --results-dir 'new_run/tersoff_single_8keV_0K_80_80_95' \
  --input-file 'input_files/fall_0K_80_80_95.input.data' \
