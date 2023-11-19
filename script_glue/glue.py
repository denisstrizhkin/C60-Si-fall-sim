#!/usr/bin/env python

import sys
from pathlib import Path
import numpy as np

sys.path.append('../')
import util


IN_GLUE = Path('./in.script_glue')
IN_ANALYZE = Path('./in.analyze')
IN_ZERO = Path('../in.zero_lvl')

INPUT_DATA_FILE = Path('./input.data')    
OUTPUT_DATA_FILE = Path('./output.data')

OSCILLATIONS_FILE = Path('./oscillations.txt')

ZERO_LVL = util.calc_zero_lvl(INPUT_DATA_FILE, IN_ZERO)
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
        in_file = IN_GLUE,
        vars = vars,
    )

    new_zero_lvl = util.calc_zero_lvl(OUTPUT_DATA_FILE, IN_ZERO)
    vars = [
        ('input_data', str(OUTPUT_DATA_FILE)),
        ('oscillations_dump', str(OSCILLATIONS_FILE)),
        ('zero_lvl', str(new_zero_lvl)),
        ('lattice', str(LATTICE)),
    ]

    util.lammps_run(
        in_file = IN_ANALYZE,
        vars = vars,
    )

    oscillations = np.loadtxt(OSCILLATIONS_FILE)
    print(oscillations)
    print(oscillations.shape)
