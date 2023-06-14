#!/usr/bin/env python

import subprocess
import time
import os
import sys
from typing import List
from pathlib import Path


sys.path.append('../')
import util
from util import Dump


def main() -> None:
  # util.lammps_run(Path('./in.coord'), [
  #   ('dump_name', 'test.dump'),
  #   ('C_cutoff', '2'),
  #   ('Si_cutoff', '2')
  # ], 4, 3)
  print(calc_zero_lvl(Path('input_files/fall0.input.data')))


if  __name__ == '__main__':
  main()
