#!/usr/bin/env fish

set -l LAMMPS_DIR /mnt/data/lammps
set -l INPUT_FILES $LAMMPS_DIR/input_files
set -l DIR (pwd)
set -l ENV /mnt/data/lammps/C60-Si-fall-sim/.venv

set -l temperature 0
set -l energy 8
set -l steps 5000
set -l n_runs 4
set -l angle1 0
set -l angle2 0

set -l RESULTS_DIR $DIR/results/{$temperature}K_{$energy}keV

mpirun -np 10 $ENV/bin/python $DIR/run.py rerun \
    --omp-threads 0 \
    --runs $n_runs \
    --results-dir $RESULTS_DIR
