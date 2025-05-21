#!/usr/bin/env fish

set -l LAMMPS_DIR /mnt/data/lammps
set -l INPUT_FILES $LAMMPS_DIR/input_files
set -l DIR (pwd)
set -l ENV /mnt/data/lammps/C60-Si-fall-sim/.venv

set -l temperature 0
set -l energy 8
set -l steps 5000
set -l n_runs 2
set -l angle1 0
set -l angle2 0

set -l RESULTS_DIR $DIR/results/{$temperature}K_{$energy}keV

mpirun -np 10 $ENV/bin/python $DIR/run.py run \
    --omp-threads 0 \
    --is-multifall \
    --temperature $temperature \
    --angle1 $angle1 \
    --angle2 $angle2 \
    --energy $energy \
    --runs $n_runs \
    --run-time $steps \
    --results-dir $RESULTS_DIR \
    --input-file $INPUT_FILES/fall_{$temperature}K_40_40_31.input.data \
    --cluster-file $INPUT_FILES/data.C_60 \
    --elstop-table $INPUT_FILES/elstop-table.txt
