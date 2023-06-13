#!/usr/bin/env python

import subprocess
import time
import os
import sys
from typing import List
from pathlib import Path


sys.path.append('../')
import util


def main():
  util.lammps_run(Path('./in.coord'), [
    '-var', 'dump_name', 'test.dump',
    '-var', 'C_cutoff', '2',
    '-var', 'Si_cutoff', '2'
  ], 4, 3)


if  __name__ == '__main__':
  main()
