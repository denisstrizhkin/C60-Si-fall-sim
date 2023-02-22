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
        "--run-time",
        action="store",
        required=False,
        default=None,
        type=int,
        help="Run simulation this amount of steps."
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
ENERGY = ARGS.energy * 1e3
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
IS_ALL_DUMP = True
ALL_DUMP_INTERVAL = 20
TMP = Path("/tmp")

if ARGS.run_time is not None:
    RUN_TIME = ARGS.run_time
elif ENERGY < 8_000:
    RUN_TIME = 10_000
else:
    RUN_TIME = ENERGY * (5 / 4)

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
    del_var(lmp, name)
    lmp.command(f"variable {name} equal {value}")


def del_var(lmp, name):
    lmp.command(f"variable {name} delete")


def del_comp(lmp, name):
    lmp.command(f"uncompute {name}")


def extract_ids_var(lmp, name, group):
    ids = lmp.extract_variable(name, group, 1)
    if len(ids) == 0:
        return []
    else:
        return ids[np.nonzero(ids)].astype(int)


def get_clusters_mask(atom_x, atom_cluster):
    mask_1 = atom_cluster != 0
    cluster_ids = set(np.unique(atom_cluster[mask_1]).flatten())

    mask_2 = atom_x[:, 2] < (ZERO_LVL + 2.0)
    no_cluster_ids = set(np.unique(atom_cluster[mask_2]).flatten())
    cluster_ids = list(cluster_ids.difference(no_cluster_ids))

    mask = np.isin(atom_cluster, cluster_ids)
    return mask, np.asarray(cluster_ids).astype(int)


def recalc_zero_lvl(lmp):
    global ZERO_LVL
    lmp.command("variable outside_id atom id")
    outside_id = extract_ids_var(lmp, "outside_id", "outside")
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
    ZERO_LVL = ave_outside_z + delta * 2
    new_var(lmp, "zero_lvl", ZERO_LVL)

    lmp.print(f"'max_outside_z {max_outside_z}'")
    lmp.print(f"'ave_outside_z: {ave_outside_z}'")
    lmp.print(f"'delta: {delta}'")
    lmp.print(f"'new zer_lvl: {ZERO_LVL}'")


def get_vacancies_group_cmd(lmp):
    vac_ids = lmp.extract_variable("vacancy_id", "Si", 1)
    vac_ids = vac_ids[vac_ids != 0]
    return "group vac id " + " ".join(vac_ids.astype(int).astype(str))


def get_clusters_table(lmp, cluster_ids, sim_num):
    table = np.array([])

    for cluster_id in cluster_ids:
        var = f"is_cluster_{cluster_id}"
        group = f"cluster_{cluster_id}"
        lmp.command(f'variable {var} atom "c_clusters=={cluster_id}"')
        lmp.command(f"group {group} variable {var}")
        lmp.command(f"compute {cluster_id}_c C60 reduce sum v_{var}")
        lmp.command(f"compute {cluster_id}_si Si reduce sum v_{var}")
        lmp.command(f"compute {cluster_id}_mom {group} momentum")
        lmp.command(f"variable {cluster_id}_mass equal mass({group})")
        lmp.command(f'fix print all print 1 "c_{cluster_id}_mom[1] c_{cluster_id}_c c_{cluster_id}_si"')
        lmp.run(0)

        comp_c = lmp.extract_compute(f"{cluster_id}_c", 0, 0)
        comp_si = lmp.extract_compute(f"{cluster_id}_si", 0, 0)
        lmp.print(f"$(c_{cluster_id}_mom[1])")
        comp_mom = lmp.extract_compute(f"{cluster_id}_mom", 0, 1)
        var_mass = lmp.extract_variable(f"{cluster_id}_mass", None, 0)

        var_ek = 2 * 5.1875 * 1e-5 * (comp_mom[0] ** 2 + comp_mom[1] ** 2 + comp_mom[2] ** 2) / (2 * var_mass)
        var_angle = np.atan(comp_mom[2] / np.sqrt(comp_mom[0] ** 2 + comp_mom[1] ** 2))
        var_angle = 90 - var_angle * 180 / np.pi,

        table = np.concatenate(
            (table, np.array([sim_num, comp_si, comp_c, var_mass, *comp_mom, var_ek, var_angle]))
        )

        del_var(lmp, var)
        del_var(lmp, f"{cluster_id}_mass")
        del_comp(lmp, f"{cluster_id}_c")
        del_comp(lmp, f"{cluster_id}_si")
        del_comp(lmp, f"{cluster_id}_mom")
        lmp.command(f"group {group} delete")

    return table.reshape((table.shape[0] // 9, 9))


def save_table(filename, table, header="", dtype="f", precision=5, mode='w'):
    fmt_str = ""

    if dtype == "d":
        fmt_str = "%d"
    elif dtype == "f":
        fmt_str = f"%.{precision}f"

    with open(filename, f"{mode}b") as file:
        np.savetxt(file, table, delimiter="\t", fmt=fmt_str, header=header)


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

        vacs_group_cmd = get_vacancies_group_cmd(lmp)
        atom_cluster = lmp.extract_compute("clusters", 1, 1)
        atom_id = lmp.gather_atoms("id")
        atom_x = lmp.gather_atoms("x")
        atom_type = lmp.gather_atoms("type")
        mask, cluster_ids = get_clusters_mask(atom_x, atom_cluster)

        clusters_table = get_clusters_table(lmp, cluster_ids, run_num)
        save_table(OUT_DIR, clusters_table, mode='a')
        print("test")
        """
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
