#!/usr/bin/env python

from pathlib import Path
import numpy as np
import subprocess
import time
import sys

sys.path.append('../')
from util import Dump


MPI_CORES = 10
OPENMP_THREADS = 1

LAMMPS_IN = Path('./in.coord')

BASE_ARGS = [
  'mpirun', '-np', str(MPI_CORES),
  'lmp', '-sf', 'omp', '-pk', 'omp', str(OPENMP_THREADS),
  '-in', LAMMPS_IN
]

COORD_DUMP_STR = 'id c_coord_num_C c_coord_num_Si v_coord_num_Sum'


def cals_coord_nums(C_cutoff, Si_cutoff):
  dump_name = f'dump.coord_num_C_{C_cutoff}_Si_{Si_cutoff}'
  dump_path = Path(dump_name)

  args = BASE_ARGS + [
    '-var', 'dump_name', dump_path,
    '-var', 'C_cutoff', str(C_cutoff),
    '-var', 'Si_cutoff', str(Si_cutoff)
  ]
  process = subprocess.Popen(args, encoding='utf-8')
  while process.poll() is None:
    time.sleep(1)

  if process.returncode != 0:
    sys.exit()

  return Dump(dump_path, COORD_DUMP_STR)
  

def get_dump_info(dump):
  C = dump['c_coord_num_C'][:]
  Si = dump['c_coord_num_Si'][:]
  Sum = dump['v_coord_num_Sum'][:]

  C_avg = C.mean()
  Si_avg = Si.mean()
  Sum_avg = Sum.mean()

  C_max = C.max()
  Si_max = Si.max()
  Sum_max = Sum.max()

  info = np.zeros((3, int(Sum_max) + 1))
  for i in range(0, int(Sum_max) + 1):
    info[0,i] = len(C[C == i])
    info[1,i] = len(Si[Si == i])
    info[2,i] = len(Sum[Sum == i])

  return dump.name + ' ' + ' '.join(str(x) for x in [
    C_avg, Si_avg, Sum_avg, C_max, Si_max, Sum_max
  ]), info


def main():
  C_cutoffs = [ 1.4, 1.5, 1.6, 1.7 ]
  #Si_cutoffs = [ 2.0, 2.1, 2.2, 2.3, 2.4, 2.5 ]
  Si_cutoffs = [ 1.85, 1.9, 1.95, 2.0, 2.05, 2.1, 2.15, 2.2, 2.25, 2.3, 2.35, 2.4 ]

  output = []
  output_count = []
  header = 'name C_sum Si_sum Sum_sum C_avg Si_avg Sum_avg C_max Si_max Sum_max'
  for i in range(0, len(Si_cutoffs)):
    dump = cals_coord_nums(1.7, Si_cutoffs[i])
    dump_info, info = get_dump_info(dump)
    output_count.append((dump.name, info))
    output.append(dump_info)

    #dump = cals_coord_nums(Si_cutoffs[i], Si_cutoffs[i])
    #dump_info, info = get_dump_info(dump)
    #output_count.append((dump.name, info))
    #output.append(dump_info)

  print('### RESULTS ###')
  print(header)
  for s in output:
    print(s)

  with open("results.txt", "w", encoding="utf-8") as file:
    file.writelines('\n'.join([header] + output))

  with open("results_count.txt", "w", encoding="utf-8") as file:
    for pair in output_count:
      file.writelines(
        '\n' + pair[0] + '\n' +
        ' '.join(
          map(str, map(
            int,
            np.linspace(0,len(pair[1][0]) - 1,len(pair[1][0]))
          ))
        ) + '\n')
      np.savetxt(file, pair[1], fmt='%d')
    

if __name__ == '__main__':
  main()
