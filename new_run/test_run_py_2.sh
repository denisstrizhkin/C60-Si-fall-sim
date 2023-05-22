#!/bin/bash

./run.py \
  --omp-threads 1 \
  --mpi-cores 10 \
  \
  --temperature 1000 \
  --energy 14 \
  --runs 50 \
  --run-time 10000 \
  \
  --results-dir './res_airebo_50_1000K_14kev' \
  --input-file '../input_files/fall1000_2.input.data' \
