#!/bin/bash

mpirun -np 4 lmp -sf omp -pk omp 4 -in in.rdf
