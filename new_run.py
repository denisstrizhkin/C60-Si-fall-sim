#!/bin/python3

from pylammpsmpi import LammpsLibrary
import numpy as np
from pathlib import Path
import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Si bombardment with C60 simulation."
    )

    parser.add_argument(
        "--temperature",
        action="store",
        required=False,
        default=700,
        type=int,
        help="Set temperature of the simulation. (K)",
    )

    parser.add_argument(
        "--energy",
        action="store",
        required=False,
        default=8,
        type=int,
        help="Set fall energy of the simulation. (keV)"
    )

    parser.add_argument(
        "--runs",
        action="store",
        required=False,
        default=2,
        type=int,
        help="Number of simulations to run."
    )

    parser.add_argument(
        "--omp-threads",
        action="store",
        required=False,
        default=2,
        type=int,
        help="Set number of OpenMP threads. (if set to 0 use GPU)"
    )

    parser.add_argument(
        "--mpi-cores",
        action="store",
        required=False,
        default=4,
        type=int,
        help="Set number of MPI cores."
    )

    return parser.parse_args()


ARGS = parse_args()
OUT_DIR = Path("./results")
INPUT_DIR = Path("./input_files")
SCRIPT_DIR = Path("./new_run")
OMP_THREADS = ARGS.omp_threads
MPI_CORES = ARGS.mpi_cores
N_RUNS = ARGS.runs
ENERGY = ARGS.energy
TEMPERATURE = ARGS.temperature
LATTICE = 5.43
STEP = 1e-3
SI_TOP = 15.3
C60_X = 0
C60_Y = 0
C60_Z = 20 + LATTICE * SI_TOP
C60_VEL = -np.sqrt(ENERGY) * 5.174
BOX_WIDTH = 12
BOX_BOTTOM = -16
RUN_TIME = 100
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


def set_suffix(lmp):
    if OMP_THREADS <= 0:
        lmp.command("package gpu 0")
        lmp.command("suffix gpu")
    else:
        lmp.command(f"package omp {OMP_THREADS}")
        lmp.command("suffix omp")


def new_var(lmp, name, value):
    lmp.command(f"variable {name} delete")
    lmp.command(f"variable {name} equal {value}")


def recalc_zero_lvl(lmp):
    lmp.command("variable outside_id atom id")
    outside_id = lmp.extract_variable("outside_id", "outside", 1)
    outside_id = outside_id[np.nonzero(outside_id)].astype(int)
    outside_z = lmp.gather_atoms("x", ids=outside_id)[:, 2]
    outside_z = np.sort(outside_z)[-20:]
    max_outside_z = outside_z.mean()

    lmp.command(
        "region surface block -${box_width} ${box_width} -${box_width} ${box_width} " +
        f"$({max_outside_z - 1.35} / v_Si_lattice) $({max_outside_z} / v_Si_lattice) units lattice"
    )
    lmp.command("group surface region surface")
    lmp.command("group outside_surface intersect surface outside")

    lmp.command("compute ave_outside_z outside_surface reduce ave z")
    ave_outside_z = lmp.extract_compute("ave_outside_z", 0, 0)
    delta = max_outside_z - ave_outside_z
    zero_lvl = ave_outside_z + delta * 2
    new_var(lmp, "zero_lvl", zero_lvl)

    lmp.print(f"'max_outside_z {max_outside_z}'")
    lmp.print(f"'ave_outside_z: {ave_outside_z}'")
    lmp.print(f"'delta: {delta}'")
    lmp.print(f"'new zer_lvl: {zero_lvl}'")


def main():
    if not OUT_DIR.exists():
        os.mkdir(OUT_DIR)

    lmp = LammpsLibrary(cores=MPI_CORES)
    lmp.command(f'log {OUT_DIR / "log.init"}')

    set_suffix(lmp)

    lmp.file(str(SCRIPT_DIR / "in.init"))
    lmp.command(f"read_data {INPUT_FILE}")
    lmp.command(f'write_restart {TMP / "restart.init"}')

    for i in range(N_RUNS):
        run_num = i + 1
        run_dir = OUT_DIR / f"run_{run_num}"
        if not run_dir.exists():
            os.mkdir(run_dir)

        lmp.clear()
        set_suffix(lmp)
        lmp.command(f'read_restart {TMP / "restart.init"}')
        lmp.command(f'log {run_dir / "log.lammps"}')
        lmp.command(
            f"lattice diamond {LATTICE} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1"
        )

        def rnd_coord(coord):
            return coord + (np.random.rand() - 0.5) * LATTICE

        new_var(lmp, "step", STEP)
        new_var(lmp, "C60_x", rnd_coord(C60_X))
        new_var(lmp, "C60_y", rnd_coord(C60_Y))
        new_var(lmp, "C60_z", C60_Z)
        new_var(lmp, "C60_vel", C60_VEL)
        new_var(lmp, "box_width", BOX_WIDTH)
        new_var(lmp, "box_bottom", BOX_BOTTOM)
        new_var(lmp, "Si_top", SI_TOP)
        new_var(lmp, "temperature", TEMPERATURE)
        new_var(lmp, "zero_lvl", ZERO_LVL)
        new_var(lmp, "Si_lattice", LATTICE)

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

        lmp.command(f'dump all all custom {ALL_DUMP_INTERVAL} {run_dir / "all.dump"} id type x y z')
        lmp.command('velocity C60 set NULL NULL ${C60_vel} sum yes units box')

        lmp.run(RUN_TIME)

        recalc_zero_lvl(lmp)
        lmp.file(str(SCRIPT_DIR / "in.clusters"))
        lmp.command(
            f'dump clusters clusters custom 1 {run_dir / "clusters.dump"} id x y z vx vy vz type c_clusters c_atom_ke')
        lmp.command(f'dump final all custom 1 {run_dir / "final.dump"} id x y z vx vy vz type c_clusters c_atom_ke')
        lmp.run(0)

        """
        vac_ids = self.lmp.get_atom_variable("vacancy_id", "si_all")
        vac_ids = vac_ids[vac_ids != 0]
        vac_group_command = "group vac id " + " ".join(vac_ids.astype(int).astype(str))

        atom_cluster = self.lmp.get_atom_vector_compute("clusters")
        atom_x = self.lmp.numpy.extract_atom("x")
        atom_id = self.lmp.numpy.extract_atom("id")
        atom_type = self.lmp.numpy.extract_atom("type")
        mask, cluster_ids = self.get_clusters_mask(atom_x, atom_cluster)

        clusters_table = self.get_clusters_table(cluster_ids)
        save_table(self.clusters_table, clusters_table, mode='a')
        rim_info = self.get_rim_info(atom_id[~mask & (atom_cluster != 0)])
        save_table(self.rim_table, rim_info, mode='a')

        carbon_hist = self.get_carbon_hist(atom_x, atom_type, mask)
        save_table(self.carbon_dist, carbon_hist, header=str(self.sim_num), mode='a')
        carbon_info = self.get_carbon_info(atom_id[~mask & (atom_type == 2)])
        save_table(self.carbon_table, carbon_info, mode='a')
        """


if __name__ == "__main__":
    main()
