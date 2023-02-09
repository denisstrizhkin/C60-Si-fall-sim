from pylammpsmpi import LammpsLibrary
import numpy as np
from pathlib import Path
import os

OUT_DIR = Path("./results")
INPUT_DIR = Path("./input_files")
SCRIPT_DIR = Path("./new_run")
OMP_THREADS = 2
MPI_CORES = 4
N_RUNS = 1
ENERGY = 8e3
TEMPERATURE = 1000
LATTICE = 5.43
STEP = 1e-3
SI_TOP = 15.3
C60_X = 0
C60_Y = 0
C60_Z = 20 + LATTICE * SI_TOP
C60_VEL = -np.sqrt(ENERGY) * 5.174
BOX_WIDTH = 12
BOX_BOTTOM = -16
RUN_TIME = 500
IS_ALL_DUMP = True
ALL_DUMP_INTERVAL = 20
TMP = Path("/tmp")

if TEMPERATURE == 0:
    ZERO_LVL = 83.19
    INPUT_FILE = INPUT_DIR / "fall.input.data"
elif TEMPERATURE == 300:
    ZERO_LVL = 82.4535
    INPUT_FILE = INPUT_DIR / "fall300.input.data"
elif TEMPERATURE == 700:
    ZERO_LVL = 83.391
    INPUT_FILE = INPUT_DIR / "fall700.input.data"
elif TEMPERATURE == 1000:
    ZERO_LVL = 84.0147
    INPUT_FILE = INPUT_DIR / "fall1000.input.data"


def main():
    if not OUT_DIR.exists():
        os.mkdir(OUT_DIR)

    lmp = LammpsLibrary(cores=MPI_CORES)
    lmp.command(f'log {OUT_DIR / "log.init"}')

    if OMP_THREADS <= 0:
        lmp.command("package gpu 0")
        lmp.command("suffix gpu")
    else:
        lmp.command(f"package omp {OMP_THREADS}")
        lmp.command("suffix omp")

    lmp.file(str(SCRIPT_DIR / "in.init"))
    lmp.command(f"read_data {INPUT_FILE}")
    lmp.command(f'write_restart {TMP / "restart.init"}')

    for i in range(N_RUNS):
        run_num = i + 1
        run_dir = OUT_DIR / f"run_{run_num}"
        if not run_dir.exists():
            os.mkdir(run_dir)

        lmp.clear()
        lmp.command(f'read_restart {TMP / "restart.init"}')
        lmp.command(f'log {run_dir / "log.lammps"}')
        lmp.command(
            f"lattice diamond {LATTICE} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1"
        )

        def new_var(name, value):
            lmp.command(f"variable {name} equal {value}")

        def rnd_coord(coord):
            return coord + (np.random.rand() - 0.5) * LATTICE

        new_var("step", STEP)
        new_var("C60_x", rnd_coord(C60_X))
        new_var("C60_y", rnd_coord(C60_Y))
        new_var("C60_z", C60_Z)
        new_var("C60_vel", C60_VEL)
        new_var("box_width", BOX_WIDTH)
        new_var("box_bottom", BOX_BOTTOM)
        new_var("Si_top", SI_TOP)
        new_var("temperature", TEMPERATURE)
        new_var("zero_lvl", ZERO_LVL)

        lmp.command(f'molecule C60 {INPUT_DIR / "mol.C60"}')
        lmp.command(
            "create_atoms 1 single ${C60_x} ${C60_y} ${C60_z} mol C60 1 units box"
        )

        lmp.file(str(SCRIPT_DIR / "in.regions"))
        lmp.file(str(SCRIPT_DIR / "in.potentials"))
        lmp.file(str(SCRIPT_DIR / "in.groups"))
        lmp.file(str(SCRIPT_DIR / "in.computes"))
        lmp.file(str(SCRIPT_DIR / "in.thermo"))
        lmp.file(str(SCRIPT_DIR / "in.fixes"))

        lmp.command(f'dump all all custom 2000 {run_dir / "all.dump"} id type x y z')
        lmp.command('velocity C60 set NULL NULL ${C60_vel} sum yes units box')

        lmp.run(RUN_TIME)


if __name__ == "__main__":
    main()
