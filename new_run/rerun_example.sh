#!/bin/bash

export LAMMPS_POTENTIALS=/usr/share/lammps/potentials

input_dir="../results/multifall_width_40_700K_40_40_8keV_300/run_300/"
input_file="${input_dir}/input.data"

./run.py \
  --omp-threads 0 \
  --mpi-cores 8 \
  \
  --runs 600 \
  --results-dir '../results/multifall_width_40_700K_40_40_8keV_300' \
  \
  --input-file "${input_file}" \
  --mol-file '../input_files/mol.C60' \
  --estop-table '../input_files/elstop-table.txt' \
  \
  --script-dir './' \
  --input-vars "${input_dir}/vars.json"
