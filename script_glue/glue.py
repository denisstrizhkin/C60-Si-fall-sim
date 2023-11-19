#!/usr/bin/env python

import sys
from pathlib import Path

sys.path.append('../')
import util


IN_FILE = Path('./in.script_glue')
INPUT_DATA_FILE = Path('./input.data')    
OUTPUT_DATA_FILE = Path('./output.data')

ZERO_LVL = util.calc_zero_lvl(INPUT_DATA_FILE, Path('../in.zero_lvl'))
LATTICE = 5.43


if __name__ == '__main__':
    offset = 2.0
    
    vars = [
        ('input_data', str(INPUT_DATA_FILE)),
        ('output_data', str(OUTPUT_DATA_FILE)),
        ('zero_lvl', str(ZERO_LVL)),
        ('lattice', str(LATTICE)),
        ('z_offset', str(offset)),
    ]
    
    util.lammps_run(
        in_file = IN_FILE,
        vars = vars,
    )
