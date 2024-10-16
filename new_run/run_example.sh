#!/bin/bash

export LAMMPS_POTENTIALS=/usr/share/lammps/potentials

input_file="../input_files/fall_0K_40_40_31_block_10_ver5.input.data"

./run.py \
  --omp-threads 0 \
  --mpi-cores 8 \
  \
  --temperature "0" \
  --energy 8 \
  --runs 300 \
  --run-time 5000 \
  \
  --results-dir '../results/block_single_0K_8kev_300_block' \
  \
  --input-file "${input_file}" \
  --mol-file '../input_files/mol.C60' \
  --estop-table '../input_files/elstop-table.txt' \
  --graphene-file '../input_files/graphene.data' \
  \
  --script-dir './'
