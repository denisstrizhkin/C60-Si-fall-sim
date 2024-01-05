#!/bin/sh

distrobox rm -f lammps
distrobox create --nvidia --name lammps -i lammps
