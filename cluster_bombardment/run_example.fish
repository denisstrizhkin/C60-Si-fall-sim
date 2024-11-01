#!/usr/bin/env fish

set INPUT_FILES /mnt/data/lammps/input_files
set LAMMPS_POTENTIALS /usr/share/lammps/potentials

set n_runs 1
set energy 8
set temperature 0
set cluster Ar12_60

mpirun -n 4 $VIRTUAL_ENV/bin/python run.py \
    --omp-threads 4 \
    --temperature $temperature \
    --energy $energy \
    --runs $n_runs \
    --run-time 5000 \
    --results-dir ./results/example_{$cluster}_{$temperature}K_{$energy}kev_{$n_runs} \
    --input-file $INPUT_FILES/fall_{$temperature}K_40_40_31.input.data \
    --cluster-file $INPUT_FILES/data.$cluster \
    --elstop-table $INPUT_FILES/elstop-table.txt
