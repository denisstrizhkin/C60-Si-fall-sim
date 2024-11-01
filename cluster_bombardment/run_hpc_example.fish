#!/usr/bin/env fish

set -l n_nodes 4
set -l n_omp 2
set -l n_cores (math (nproc) / $n_omp)

set -l LAMMPS_DIR $HOME/lammps
set -l INPUT_FILES $LAMMPS_DIR/input_files
set -l DIR $LAMMPS_DIR/single/base
set -l ENV $LAMMPS_DIR/env

set -l n_runs 3

echo "#!/bin/sh
#SBATCH --time=00:30:00
#SBATCH --nodes=$n_nodes
#SBATCH --ntasks-per-node=$n_cores
#SBATCH --cpus-per-task=$n_omp

set -x

module load lammps/29.08.2024

mpirun $ENV/bin/python $DIR/run.py \
    --omp-threads $n_omp \
    --temperature 0 \
    --energy 8 \
    --runs $n_runs \
    --run-time 2000 \
    --results-dir $DIR/results/test \
    --input-file $INPUT_FILES/fall_0K_40_40_31.input.data \
    --cluster-file $INPUT_FILES/data.C_60 \
    --elstop-table $INPUT_FILES/elstop-table.txt \
" | sbatch -o output
