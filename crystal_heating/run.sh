#!/usr/bin/env bash

mpirun -n 8 lmp -in heating.in
sed -i -E 's/^([0-9]) a/2 a/' output.data
sed -i '/^1 28/a 2 12.011' output.data
