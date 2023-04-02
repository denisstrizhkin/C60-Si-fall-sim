#!/bin/python3

from pylammpsmpi import LammpsLibrary
import numpy as np
from pathlib import Path
import argparse
import os
import tempfile


class Dump:
    def __init__(self, dump_path: Path, dump_str: str):
        self.data = np.loadtxt(dump_path, skiprows=9)
        self.keys = dump_str.split()

        if len(set(self.keys)) != len(self.keys):
            raise ValueError('dump keys must be unique')

    def __getitem__(self, key: str):
        if key not in self.keys:
            raise ValueError(f'no such key: {key}')

        return self.data[:, self.keys.index(key)]
   

class ClusterAtom:
    def __init__(self, z, vx, vy, vz, mass, type):
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass
        self.type = type


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
    
    parser.add_argument(
        "--results-dir",
        action="store",
        required=False,
        default='./results',
        type=str,
        help="Set directory path where to store computational results.",
    )

    parser.add_argument(
        "--input-file",
        action="store",
        required=True,
        type=str,
        help="Set input file.",
    )

    return parser.parse_args()


ARGS = parse_args()

OUT_DIR = Path(ARGS.results_dir)

INPUT_FILE = Path(ARGS.input_file)
INPUT_DIR = INPUT_FILE.parent
MOL_FILE = INPUT_DIR / 'mol.C60'
ELSTOP_TABLE = INPUT_DIR / 'elstop-table.txt'

SCRIPT_DIR = Path("./")

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
C60_Z_OFFSET = 30

IS_ALL_DUMP = True
ALL_DUMP_INTERVAL = 20

TMP = Path(tempfile.gettempdir())

if ARGS.run_time is not None:
    RUN_TIME = ARGS.run_time
elif ENERGY < 8_000:
    RUN_TIME = 10_000
else:
    RUN_TIME = ENERGY * (5 / 4)

if TEMPERATURE == 0:
    ZERO_LVL = 83.19
elif TEMPERATURE == 300:
    ZERO_LVL = 82.4535
elif TEMPERATURE == 700:
    ZERO_LVL = 83.391
elif TEMPERATURE == 1000:
    ZERO_LVL = 84.0147


def set_suffix(lmp):
    if OMP_THREADS <= 0:
        lmp.command("package gpu 0")
        lmp.command("suffix gpu")
    else:
        lmp.command(f"package omp {OMP_THREADS}")
        lmp.command("suffix omp")


def extract_ids_var(lmp, name, group):
    ids = lmp.extract_variable(name, group, 1)
    if len(ids) == 0:
        return []
    else:
        return ids[np.nonzero(ids)].astype(int)


def get_cluster_dic(cluster_dump: Dump):
    clusters = cluster_dump['c_clusters']

    cluster_dic = dict()
    for cluster_id in set(clusters):
        cluster_dic[cluster_id] = []
    
    z = cluster_dump['z']
    vx = cluster_dump['vx']
    vy = cluster_dump['vy']
    vz = cluster_dump['vz']
    mass = cluster_dump['c_mass']
    type = cluster_dump['type']
   
    for i in range(0, len(z)):
        cluster = ClusterAtom(z = z[i], vx = vx[i], vy = vy[i], vz = vz[i], mass = mass[i], type = type[i])
        cluster_dic[clusters[i]].append(cluster)

    keys_to_delete = []
    for key in cluster_dic.keys():
        for cluster in cluster_dic[key]:
            if cluster.z < (ZERO_LVL + 2.0):
                keys_to_delete.append(key)
                break;

    for key in keys_to_delete:
       cluster_dic.pop(key) 

    return cluster_dic


def recalc_zero_lvl(lmp):
    global ZERO_LVL

    lmp.variable('outside_id', 'atom', 'id')
    outside_id = extract_ids_var(lmp, "outside_id", "outside")
    outside_z = lmp.gather_atoms("x", ids=outside_id)[:, 2]
    outside_z = np.sort(outside_z)[-20:]
    max_outside_z = outside_z.mean()

    lmp.region('surface', 'block', 'EDGE', 'EDGE', 'EDGE', 'EDGE', max_outside_z - 1.35, max_outside_z, 'units', 'box')
    lmp.group('surface', 'region', 'surface')
    lmp.group('outside_surface', 'intersect', 'surface', 'outside')

    lmp.compute('ave_outside_z', 'outside_surface', 'reduce', 'ave', 'z')
    ave_outside_z = lmp.extract_compute("ave_outside_z", 0, 0)
    delta = max_outside_z - ave_outside_z
    ZERO_LVL = ave_outside_z + delta * 2

    lmp.variable('zero_lvl', 'index', ZERO_LVL)

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

    input_file = INPUT_FILE
    for i in range(N_RUNS):    
        lmp = LammpsLibrary(cores=MPI_CORES)
        
        run_num = i + 1
        run_dir = OUT_DIR / f"run_{run_num}"
        if not run_dir.exists():
            os.mkdir(run_dir)

        lmp.command(f'log {run_dir / "log.lammps"}')
        set_suffix(lmp)

        def rnd_coord(coord):
            return coord + (np.random.rand() - 0.5) * LATTICE

        lmp.variable('input_file', 'index', f'"{input_file}"')
        lmp.variable('mol_file', 'index', f'"{MOL_FILE}"')
        lmp.variable('elstop_table', 'index', f'"{ELSTOP_TABLE}"')
  
        lmp.variable('lattice', 'index', LATTICE)

        lmp.variable('Si_top', 'index', 83)

        lmp.variable('C60_x', 'index', C60_X)
        lmp.variable('C60_y', 'index', C60_Y)
        lmp.variable('C60_z_offset', 'index', C60_Z_OFFSET)

        lmp.variable('step', 'index', STEP)
        lmp.variable('temperature', 'index', TEMPERATURE)
        lmp.variable('fall_steps', 'index', RUN_TIME)
        lmp.variable('energy', 'index', ENERGY)

        lmp.variable('zero_lvl', 'index', ZERO_LVL)

        lmp.file(str(SCRIPT_DIR / "in.fall"))
        recalc_zero_lvl(lmp)

        lmp.file(str(SCRIPT_DIR / "in.clusters"))

        dump_cluster_path = run_dir / 'dump.clusters'
        dump_cluster_str = 'id x y z vx vy vz type c_mass c_clusters c_atom_ke'

        lmp.command(f'dump clusters clusters custom 1 {dump_cluster_path} {dump_cluster_str}')
        lmp.command(f'dump final all custom 1 {run_dir / "dump.final"} id x y z vx vy vz type c_clusters c_atom_ke')
        
        lmp.run(0)
        lmp.undump('clusters')
        lmp.undump('final')

        var_group_cmd = get_vacancies_group_cmd(lmp)
        dump_cluster = Dump(dump_cluster_path, dump_cluster_str)

        cluster_dic = get_cluster_dic(dump_cluster)
        print(cluster_dic)
        exit()
        """
        clusters_table = self.get_clusters_table(cluster_ids).astype(float)
        save_table(self.clusters_table, clusters_table, mode='a')
        rim_info = self.get_rim_info(atom_id[~mask & (atom_cluster != 0)])
        save_table(self.rim_table, rim_info, mode='a')

        carbon_hist = self.get_carbon_hist(atom_x, atom_type, mask)
        save_table(self.carbon_dist, carbon_hist, header=str(self.sim_num), mode='a')
        carbon_info = self.get_carbon_info(atom_id[~mask & (atom_type == 2)])
        save_table(self.carbon_table, carbon_info, mode='a')

        self.lmp.run(0)
        self.lmp.command("unfix print_cluster")
        if len(cluster_ids) != 0:
            cluster_group_command = "group cluster id " + " ".join(
                atom_id[np.where(np.in1d(atom_cluster, cluster_ids))].astype(int).astype(str)
            )
            self.lmp.command(cluster_group_command)
            self.lmp.command("delete_atoms group cluster")
        self.lmp.command("write_data tmp.input.data")
        self.input_file_path = './tmp.input.data'

        self.lmp_stop()
        self.lmp_start()
        self.lmp.command(f"read_restart {self.vacancies_restart_file}")
        self.potentials()
        self.lmp.command(vac_group_command)
        self.lmp.command("group si_all type 1")
        self.lmp.command("compute voro_vol si_all voronoi/atom only_group")
        self.lmp.command("compute clusters vac cluster/atom 3")
        self.lmp.command(
            f"dump clusters vac custom 20 {self.results_dir}/crater_{self.sim_num}.dump \
id x     y z vx vy vz type c_clusters"
        )
        self.lmp.run(0)

        clusters = self.lmp.get_atom_vector_compute("clusters")
        clusters = clusters[clusters != 0]
        crater_info = self.get_crater_info(clusters)
        save_table(self.crater_table, crater_info, mode='a')

        self.lmp.close()"""

if __name__ == "__main__":
    main()
