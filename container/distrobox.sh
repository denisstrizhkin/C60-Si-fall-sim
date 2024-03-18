#!/bin/sh

distrobox rm -f lammps
distrobox create --nvidia --name lammps --home "${HOME}/.local/lammps" -i lammps
