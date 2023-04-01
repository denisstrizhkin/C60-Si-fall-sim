#!/bin/bash

lmp -in in.fall \
  -var input_file '../input_files/fall0.input.data' \
  -var mol_file '../input_files/mol.C60' \
  \
  -var lattice 5.43 \
  \
  -var Si_top 82.78 \
  \
  -var C60_x 0 \
  -var C60_y 0 \
  -var C60_z_offset 30 \
