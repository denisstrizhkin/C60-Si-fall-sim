#!/bin/bash

export LAMMPS_POTENTIALS=/usr/share/lammps/potentials

input_dir="../results/block_single_0K_8kev_100_multik/run_87"
input_file="${input_dir}/input.data"

./run.py \
  --omp-threads 0 \
  --mpi-cores 8 \
  \
  --runs 87 \
  --results-dir '../results/block_single_0K_8kev_100_multik' \
  \
  --input-file "${input_file}" \
  --mol-file '../input_files/mol.C60' \
  --estop-table '../input_files/elstop-table.txt' \
  \
  --script-dir './' \
  --input-vars "${input_dir}/vars.json"
