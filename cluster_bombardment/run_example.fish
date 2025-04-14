#!/usr/bin/env fish

set -l LAMMPS_DIR $HOME/lammps
set -l INPUT_FILES $LAMMPS_DIR/input_files
set -l DIR $LAMMPS_DIR/single/angle_C60
set -l ENV $LAMMPS_DIR/env

set -l energy 8
set -l steps 5000
set -l n_runs 100
set -l angle1 $argv[1]
set -l angle2 $argv[2]

set -l RESULTS_DIR $DIR/results/0K_8keV_angles_{$angle1}_{$angle2}

mpirun $ENV/bin/python $DIR/run.py \
    --omp-threads $n_omp \
    --temperature 0 \
    --angle1 $angle1 \
    --angle2 $angle2 \
    --energy $energy \
    --runs $n_runs \
    --run-time $steps \
    --results-dir $RESULTS_DIR \
    --input-file $INPUT_FILES/fall_0K_40_40_31.input.data \
    --cluster-file $INPUT_FILES/data.C_60 \
    --elstop-table $INPUT_FILES/elstop-table.txt
