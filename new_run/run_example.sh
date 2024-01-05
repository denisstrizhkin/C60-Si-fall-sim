#!/bin/bash

export LAMMPS_POTENTIALS=/usr/share/lammps/potentials

./run.py \
  --omp-threads 0 \
  --mpi-cores 8 \
  \
  --temperature "1e-6" \
  --energy 8 \
  --runs 5 \
  --run-time 3000 \
  \
  --results-dir '../results/test' \
  \
  --input-file '../input_files/fall_0K_20_20_31.input.data' \
  --mol-file '../input_files/mol.C60' \
  --estop-table '../input_files/elstop-table.txt' \
  \
  --script-dir './'
