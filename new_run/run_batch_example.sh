#!/bin/bash

export LAMMPS_POTENTIALS=/usr/share/lammps/potentials

function test_size {
  ./run.py \
    --omp-threads 0 \
    --mpi-cores 8 \
    \
    --temperature "1e-6" \
    --energy $1 \
    --runs 50 \
    --run-time $3 \
    \
    --results-dir "../results/test_0K_${2}_31_${1}keV" \
    --input-file "../input_files/fall_0K_${2}_31.input.data" \
    --mol-file '../input_files/mol.C60' \
    --estop-table '../input_files/elstop-table.txt' \
    \
    --script-dir './'
}

# test_size 8  10_10 5000
# test_size 8  15_15 5000
# test_size 8  20_20 5000
# test_size 8  30_30 5000
test_size 8  40_40 5000
# test_size 8  60_60 5000
# test_size 8  80_80 5000

# test_size 14 10_10 10000
# test_size 14 15_15 10000
# test_size 14 20_20 10000
# test_size 14 30_30 10000
test_size 14 40_40 10000
# test_size 14 60_60 10000
# test_size 14 80_80 10000
# test_size 14 90_90 10000
