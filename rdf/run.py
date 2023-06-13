import subprocess
import time
import os
import sys
from typing import List
from pathlib import Path


MPI_CORES: int = 3
OPENMP_THREADS: int = 4

DOCKER_BASE: List[str] = [
  'docker', 'run', '--rm',
  '-v', f'{os.getcwd()}:/var/workdir',
  '--user', f'{os.getuid()}:{os.getgid()}'
]

LAMMPS_IN: Path = Path('./in.coord')


def lammps_run(
  in_file: Path, vars: List[str]=None,
  omp_threads: int=4, mpi_cores: int=3
  ):
  mpirun_base = [
    'mpirun', '-np', str(mpi_cores),
    'lmp', '-in', str(in_file)
  ]

  if (omp_threads <= 0):
    args = mpirun_base + [
      '-sf', 'gpu',
      '-pk', 'gpu', '0',
    ] + vars
    run_args = DOCKER_BASE + [
      '--gpus', 'all',
      'lammpsopencl'
    ]
  else:
    args = mpirun_base + [
      '-sf', 'omp',
      '-pk', 'omp', str(omp_threads),
    ] + vars
    run_args = DOCKER_BASE + [
      'lammpsmpi'
    ]

  print(args)
  run_args = run_args + [ ' '.join(args) ]
  print(run_args)

  process = subprocess.Popen(run_args, encoding='utf-8')
  while process.poll() is None:
    time.sleep(1)

  if process.returncode != 0:
    print("FAIL")
    sys.exit()


def main():
  lammps_run(LAMMPS_IN, [
    '-var', 'dump_name', 'test.dump',
    '-var', 'C_cutoff', '2',
    '-var', 'Si_cutoff', '2'
  ], 0, 3)


if  __name__ == '__main__':
  main()
