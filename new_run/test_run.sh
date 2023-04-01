#!/bin/bash

mpirun -np 10 lmp \
  -in in.fall \
  -suffix gpu \
  -package gpu 0 \
  \
  -var input_file '../input_files/fall0.input.data' \
  -var mol_file '../input_files/mol.C60' \
  -var elstop_table '../input_files/elstop-table.txt ' \
  \
  -var lattice 5.43 \
  \
  -var Si_top 82.78 \
  \
  -var C60_x 0 \
  -var C60_y 0 \
  -var C60_z_offset 30 \
  \
  -var step 0.001 \
  -var temperature 0 \
  -var fall_steps 5000 \
  -var energy 2 \
