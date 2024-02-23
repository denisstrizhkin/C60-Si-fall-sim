#!/bin/sh

export LAMMPS_POTENTIALS=/usr/share/lammps/potentials

python graphene.py
mpirun -n 10 lmp -in graphene.in
