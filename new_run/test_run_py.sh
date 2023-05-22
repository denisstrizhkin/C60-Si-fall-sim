#!/bin/bash

./run.py \
  --omp-threads 0 \
  --mpi-cores 6 \
  \
  --temperature 300 \
  --energy 8 \
  --runs 300 \
  --run-time 5000 \
  \
  --results-dir './res_300_0K_8kev' \
  --input-file '../input_files/fall300_2.input.data' \
