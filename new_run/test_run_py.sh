#!/bin/bash

./run.py \
  --omp-threads 0 \
  --mpi-cores 6 \
  \
  --temperature 0 \
  --energy 8 \
  --runs 1 \
  --run-time 2000 \
  \
  --results-dir './res_40' \
  --input-file '../input_files/fall0_2.input.data' \
