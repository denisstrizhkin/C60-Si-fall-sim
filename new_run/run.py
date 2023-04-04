#!/bin/python3

from pylammpsmpi import LammpsLibrary
import numpy as np
from pathlib import Path
import argparse
import os
from os import path
import tempfile
from typing import List


class Dump:
    def __init__(self, dump_path: Path, dump_str: str):
        self.data = np.loadtxt(dump_path, ndmin=2, skiprows=9)
        self.keys = dump_str.split()

        if len(set(self.keys)) != len(self.keys):
            raise ValueError('dump keys must be unique')

    def __getitem__(self, key: str):
        if key not in self.keys:
            raise ValueError(f'no such key: {key}')
       
        if len(self.data) == 0:
            return []

        return self.data[:, self.keys.index(key)]
   

class Atom:
    def __init__(self, x = 0, y = 0, z = 0, vx = 0, vy = 0, vz = 0, mass = 0, type = 0, id = 0):
        self.x = x
        self.y = y
        self.z = z
        self.vx = vx
        self.vy = vy
        self.vz = vz
        self.mass = mass
        self.type = type
        self.id = id


class Cluster:
    def __init__(self, clusters: List[Atom]):
        self.mass = 0
        self.count_Si = 0
        self.count_C = 0
        self.mx = 0
        self.my = 0
        self.mz = 0

        for cluster in clusters:
            self.mx += cluster.vx * cluster.mass
            self.my += cluster.vy * cluster.mass
            self.mz += cluster.vz * cluster.mass
            self.mass += cluster.mass 

            if cluster.type == SI_ATOM_TYPE:
                self.count_Si += 1
            else:
                self.count_C += 1

        self.ek = 2 * 5.1875 * 1e-5 * (self.mx ** 2 + self.my ** 2 + self.mz ** 2) / (2 * self.mass)
        
        self.angle = np.arctan(self.mz / np.sqrt(self.mx ** 2 + self.my ** 2))
        self.angle = 90 - self.angle * 180 / np.pi


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
if not OUT_DIR.exists():
    os.mkdir(OUT_DIR)

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

SI_ATOM_TYPE = 1
C_ATOM_TYPE = 2


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


CLUSTERS_TABLE = OUT_DIR / "clusters_table.txt"
RIM_TABLE = OUT_DIR / "rim_table.txt"
CARBON_TABLE = OUT_DIR / "carbon_table.txt"
CRATER_TABLE = OUT_DIR / "crater_table.txt"
CARBON_DIST = OUT_DIR / "carbon_dist.txt"


def write_header(header_str, table_path):
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# " + header_str + "\n")


write_header("sim_num N_Si N_C mass Px Py Pz Ek angle", CLUSTERS_TABLE)
write_header("sim_num N r_mean r_max z_mean z_max", RIM_TABLE)
write_header("sim_num N r_mean r_max", CARBON_TABLE)
write_header("sim_num N V S z_mean z_min", CRATER_TABLE)
write_header("z count", CARBON_DIST)


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
   
    x = cluster_dump['x']
    y = cluster_dump['y']
    z = cluster_dump['z']
    vx = cluster_dump['vx']
    vy = cluster_dump['vy']
    vz = cluster_dump['vz']
    mass = cluster_dump['c_mass']
    type = cluster_dump['type']
    id = cluster_dump['id']
   
    for i in range(0, len(z)):
        cluster = Atom(
            x = x[i], y = y[i], z = z[i],
            vx = vx[i], vy = vy[i], vz = vz[i],
            mass = mass[i], type = type[i], id = id[i]
        )
        cluster_dic[clusters[i]].append(cluster)

    keys_to_delete = []
    for key in cluster_dic.keys():
        for cluster in cluster_dic[key]:
            if cluster.z < (ZERO_LVL + 2.0):
                keys_to_delete.append(key)
                break;

    rim_atoms = []
    for key in keys_to_delete:
       rim_atoms += cluster_dic.pop(key) 

    return cluster_dic, rim_atoms


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


def get_clusters_table(cluster_dic, sim_num):
    table = np.array([])

    for key in cluster_dic.keys():
        cluster = cluster_dic[key]
        table = np.concatenate(
            (table, np.array([
                sim_num, cluster.count_Si, cluster.count_C, cluster.mass,
                cluster.mx, cluster.my, cluster.mz,
                cluster.ek, cluster.angle
            ]))
        )

    return table.reshape((table.shape[0] // 9, 9))


def get_rim_info(rim_atoms, fu_x, fu_y, sim_num):
    if len(rim_atoms) == 0:
        return np.array([])

    r = []
    z = []

    for atom in rim_atoms:
        r.append(np.sqrt((atom.x - fu_x) ** 2 + (atom.y - fu_y) ** 2))
        z.append(atom.z)

    r = np.array(r)
    z = np.array(z)

    return np.array(
        [
            [
                sim_num,
                len(rim_atoms),
                r.mean(),
                r.max(),
                z.mean() - ZERO_LVL,
                z.max() - ZERO_LVL,
            ]
        ]
    )


def get_carbon(dump_final, carbon_sputtered):
    x = dump_final['x']
    y = dump_final['y']
    z = dump_final['z']
    id = dump_final['id']
    type = dump_final['type']

    carbon = []
    for i in range(0, len(x)):
        if id[i] not in carbon_sputtered and type[i] == C_ATOM_TYPE:
            carbon.append(Atom(x = x[i], y = y[i], z = z[i], id = id[i]))
    
    return carbon


def get_carbon_hist(carbon):
    z_coords = []
    for c in carbon:
        z_coords.append(np.around(c.z - ZERO_LVL, 1))
    z_coords = np.array(z_coords)

    right = int(np.ceil(z_coords.max(initial=float('-inf'))))
    left = int(np.floor(z_coords.min(initial=float('+inf'))))
    hist, bins = np.histogram(z_coords, bins=(right - left), range=(left, right))
    length = len(hist)
    hist = np.concatenate(
        ((bins[1:] - 0.5).reshape(length, 1), hist.reshape(length, 1)), axis=1
    )

    return hist


def get_carbon_info(carbon, fu_x, fu_y, sim_num):
    if len(carbon) == 0:
        return np.array([])

    r = []
    for atom in carbon:
        r.append(np.sqrt((atom.x - fu_x) ** 2 + (atom.y - fu_y) ** 2))
    r = np.array(r)

    return np.array([[sim_num, len(carbon), r.mean(), r.max()]])


def get_crater_info(lmp, dump_crater: Dump, sim_num):
    id = dump_crater['id']
    z = dump_crater['z']
    clusters = dump_crater['c_clusters']

    crater_id = np.bincount(id.astype(int)).argmax()
    atoms = []
    for i in range(0, len(id)):
        if clusters[i] == crater_id:
            atoms.append(Atom(z = z[i], id = id[i]))
    

    voronoi = lmp.extract_compute("voro_vol", 1, 2, width=2)
    cell_vol = np.median(voronoi, axis=0)[0]
    crater_vol = cell_vol * len(atoms)

    surface_count = 0
    z = []
    for atom in atoms:
        z.append(atom.z)
        if atom.z > -2.4 * 0.707 + ZERO_LVL:
            surface_count += 1
    z = np.array(z)

    cell_surface = 7.3712
    surface_area = cell_surface * surface_count

    return np.array(
        [
            [
                sim_num,
                len(atoms),
                crater_vol,
                surface_area,
                z.mean() - ZERO_LVL,
                z.min() - ZERO_LVL,
            ]
        ]
    )


def save_table(filename, table, header="", dtype="f", precision=5, mode='w'):
    fmt_str = ""

    if dtype == "d":
        fmt_str = "%d"
    elif dtype == "f":
        fmt_str = f"%.{precision}f"

    with open(filename, f"{mode}b") as file:
        np.savetxt(file, table, delimiter="\t", fmt=fmt_str, header=header)


def carbon_dist_parse(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    lines_dic = {}
    sim_num = -1
    for i in range(1, len(lines)):
        tokens = lines[i].strip().split()
        if lines[i][0] == "#":
            sim_num = int(tokens[1])
            lines_dic[sim_num] = []
        else:
            lines_dic[sim_num].append(list(map(float, tokens)))

    z_min = 0
    z_max = 0
    for key in lines_dic.keys():
        z_min = min(lines_dic[key][0][0], z_min)
        z_max = max(lines_dic[key][len(lines_dic[key]) - 1][0], z_max)

    bins = np.linspace(z_min, z_max, int(z_max - z_min) + 1)
    table = np.zeros((len(lines_dic) + 1, len(bins) + 1))

    sim_nums = list(lines_dic.keys())
    for i in range(0, len(sim_nums)):
        table[i + 1][0] = sim_nums[i]
        for pair in lines_dic[sim_nums[i]]:
            index = int(pair[0] - z_min)
            table[i + 1][index + 1] = pair[1]

    for i in range(0, len(bins)):
        table[0][i + 1] = bins[i]

    header_str = "simN " + " ".join(list(map(str, bins)))

    output_path = path.splitext(file_path)[0] + "_parsed" + path.splitext(file_path)[1]
    save_table(output_path, table.T, header_str)


def clusters_parse(file_path):
    clusters = np.loadtxt(file_path, ndmin=2, skiprows=1)
    clusters = clusters[:, :3]

    clusters_dic = {}
    for cluster in clusters:
        cluster_str = "Si" + str(int(cluster[1])) + "C" + str(int(cluster[2]))
        if cluster_str not in clusters_dic.keys():
            clusters_dic[cluster_str] = {}

        sim_num = int(cluster[0])
        if not cluster[0] in clusters_dic[cluster_str]:
            clusters_dic[cluster_str][sim_num] = 0

        clusters_dic[cluster_str][sim_num] += 1

    # total_sims = len(np.unique(clusters[:, 0]))
    total_sims = 50
    total_clusters = len(clusters_dic.keys())

    table = np.zeros((total_sims, total_clusters + 1))
    cluster_index = 0
    for key in clusters_dic.keys():
        for sim_num in clusters_dic[key].keys():
            table[sim_num - 1][cluster_index + 1] = clusters_dic[key][sim_num]
            table[sim_num - 1, 0] = sim_num
        cluster_index += 1

    header_str = "simN\t" + "\t".join(clusters_dic.keys())
    output_path = path.splitext(file_path)[0] + "_parsed" + path.splitext(file_path)[1]
    save_table(output_path, table, header_str, dtype="d")


def clusters_parse_sum(file_path):
    clusters = np.loadtxt(file_path, ndmin=2, skiprows=1)
    
    clusters = clusters[:, :3]

    clusters_dic = {}
    for cluster in clusters:
        sim_num = int(cluster[0])
        if sim_num not in clusters_dic.keys():
            clusters_dic[sim_num] = {}
            clusters_dic[sim_num]["Si"] = 0
            clusters_dic[sim_num]["C"] = 0
        clusters_dic[sim_num]["Si"] += cluster[1]
        clusters_dic[sim_num]["C"] += cluster[2]

    total_sims = len(clusters_dic.keys())
    table = np.zeros((total_sims, 4))

    keys = list(clusters_dic.keys())
    for i in range(0, len(keys)):
        sim_num = keys[i]
        table[i][0] = keys[i]
        table[i][1] = clusters_dic[sim_num]["Si"]
        table[i][2] = clusters_dic[sim_num]["C"]
        table[i][3] = table[i][1] + table[i][2]

    header_str = "simN Si C"
    output_path = (
            path.splitext(file_path)[0] + "_parsed_sum" + path.splitext(file_path)[1]
    )

    save_table(output_path, table, header_str, dtype="d")


def clusters_parse_angle_dist(file_path):
    clusters = np.loadtxt(file_path, ndmin=2, skiprows=1)

    clusters_sim_num_n = clusters[:, :2]
    clusters_sim_num_n[:, 1] = clusters[:, 1] + clusters[:, 2]

    clusters_enrg_ang = clusters[:, :-2]
    clusters_enrg_ang[:, 0] /= clusters_sim_num_n[:, 1]

    num_bins = (85 - 5) // 10 + 1
    num_sims = 50 + 1

    number_table = np.zeros((num_bins, num_sims))
    energy_table = np.zeros((num_bins, num_sims))

    number_table[:, 0] = np.linspace(5, 85, 9)
    energy_table[:, 0] = np.linspace(5, 85, 9)

    for i in range(0, len(clusters)):
        angle_index = int(np.floor(clusters_enrg_ang[i, 1])) // 10
        sim_index = int(clusters_sim_num_n[i, 0])

        if angle_index >= num_bins:
            continue

        number_table[angle_index, sim_index] += clusters_sim_num_n[i, 1]
        energy_table[angle_index, sim_index] += clusters_enrg_ang[i, 1]

    header_str_number = "angle N1 N2 N3 ... N50"
    output_path_number = (
            path.splitext(file_path)[0]
            + "_parsed_number_dist"
            + path.splitext(file_path)[1]
    )
    save_table(output_path_number, number_table, header_str_number)

    header_str_energy = "angle E1 E2 E3 ... E50"
    output_path_energy = (
            path.splitext(file_path)[0]
            + "_parsed_energy_dist"
            + path.splitext(file_path)[1]
    )
    save_table(output_path_energy, energy_table, header_str_energy)


def main():
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
            return coord + (np.random.rand() * 2 - 1) * LATTICE * 10

        lmp.variable('input_file', 'index', f'"{input_file}"')
        lmp.variable('mol_file', 'index', f'"{MOL_FILE}"')
        lmp.variable('elstop_table', 'index', f'"{ELSTOP_TABLE}"')
  
        lmp.variable('lattice', 'index', LATTICE)

        lmp.variable('Si_top', 'index', 83)

        fu_x = rnd_coord(C60_X)
        fu_y = rnd_coord(C60_Y)
        vacs_restart_file = TMP / 'vacs.restart'

        lmp.variable('C60_x', 'index', fu_x)
        lmp.variable('C60_y', 'index', fu_y)
        lmp.variable('C60_z_offset', 'index', C60_Z_OFFSET)

        lmp.variable('step', 'index', STEP)
        lmp.variable('temperature', 'index', TEMPERATURE)
        lmp.variable('energy', 'index', ENERGY)

        lmp.variable('zero_lvl', 'index', ZERO_LVL)
        lmp.variable('vacs_restart_file', 'index', f'"{vacs_restart_file}"')

        lmp.file(str(SCRIPT_DIR / "in.fall"))
        recalc_zero_lvl(lmp)

        lmp.file(str(SCRIPT_DIR / "in.clusters"))
        lmp.command(
            f'fix temp_time all print 10 "$(time) $(temp)" file {OUT_DIR}/temp_time.txt screen no'
        )
        lmp.command(
            f'fix penrg_time all print 10 "$(time) $(pe)" file {OUT_DIR}/penrg_time.txt screen no'
        )
        lmp.run(RUN_TIME)

        dump_cluster_path = run_dir / 'dump.clusters'
        dump_cluster_str = 'id x y z vx vy vz type c_mass c_clusters c_atom_ke'
        lmp.command(f'dump clusters clusters custom 1 {dump_cluster_path} {dump_cluster_str}')

        dump_final_path = run_dir / 'dump.final'
        dump_final_str = 'id x y z vx vy vz type c_clusters c_atom_ke'
        lmp.command(f'dump final all custom 1 {dump_final_path} {dump_final_str}')
        
        lmp.run(0)
        lmp.undump('clusters')
        lmp.undump('final')

        vac_group_cmd = get_vacancies_group_cmd(lmp)
        dump_cluster = Dump(dump_cluster_path, dump_cluster_str)
        dump_final = Dump(dump_final_path, dump_final_str)

        cluster_dic_atoms, rim_atoms = get_cluster_dic(dump_cluster)

        cluster_dic = dict()
        carbon_sputtered = set()
        for key in cluster_dic_atoms.keys():
            atoms = cluster_dic_atoms[key]
            for atom in atoms:
                if atom.type == C_ATOM_TYPE:
                    carbon_sputtered.add(atom.id)
            cluster_dic[key] = Cluster(cluster_dic_atoms[key])

        clusters_table = get_clusters_table(cluster_dic, run_num).astype(float)
        save_table(CLUSTERS_TABLE, clusters_table, mode='a')

        rim_info = get_rim_info(rim_atoms, fu_x, fu_y, run_num)
        save_table(RIM_TABLE, rim_info, mode='a')

        carbon = get_carbon(dump_final, carbon_sputtered)
        carbon_hist = get_carbon_hist(carbon)
        save_table(CARBON_DIST, carbon_hist, header=str(run_num), mode='a')
        carbon_info = get_carbon_info(carbon, fu_x, fu_y, run_num)
        save_table(CARBON_TABLE, carbon_info, mode='a')

        if len(cluster_dic.keys()) != 0:
            ids_to_delete = []
            for key in cluster_dic_atoms.keys():
                for atom in cluster_dic_atoms[key]:
                    ids_to_delete.append(atom.id)
            ids_to_delete = np.array(ids_to_delete)

            cluster_group_command = "group cluster id " + " ".join(ids_to_delete.astype(int).astype(str))
            lmp.command(cluster_group_command)
            lmp.delete_atoms('group', 'cluster')

        lmp.command("unfix tbath")
        lmp.command("fix tbath nve temp/berendsen ${temperature} ${temperature} 0.001")
        lmp.run(4000) 

        lmp.command("unfix tbath")
        lmp.command("fix tbath thermostat temp/berendsen ${temperature} ${temperature} 0.001")
        lmp.run(1000)

        input_file = TMP / 'tmp.input.data'
        lmp.write_data(f'"{input_file}"')

        lmp.close()
        lmp = LammpsLibrary(cores=MPI_CORES)
        
        lmp.read_restart(f'"{vacs_restart_file}"')
        lmp.command("pair_style tersoff/zbl\npair_coeff * * SiC.tersoff.zbl Si C\nneighbor 3.0 bin")
       
        lmp.command(vac_group_cmd)
        lmp.command("group si_all type 1")
        lmp.command("compute voro_vol si_all voronoi/atom only_group")
        lmp.command("compute clusters vac cluster/atom 3")

        dump_crater_path = run_dir / 'dump.crater'
        dump_crater_str = 'id x y z vx vy vz type c_clusters'
        lmp.command(
            f"dump clusters vac custom 20 {dump_crater_path} {dump_crater_str}"
        )
        lmp.run(0)

        dump_crater = Dump(dump_crater_path, dump_crater_str)
        crater_info = get_crater_info(lmp, dump_crater, run_num)
        save_table(CRATER_TABLE, crater_info, mode='a')

        lmp.close()

    clusters_parse(CLUSTERS_TABLE)
    clusters_parse_sum(CLUSTERS_TABLE)
    clusters_parse_angle_dist(CLUSTERS_TABLE)
    carbon_dist_parse(CARBON_DIST)

    os.system(f'tar cvzf {OUT_DIR}.tar.gz {OUT_DIR}/*')
    print("*** FINISHED COMPLETELY ***")


if __name__ == "__main__":
    main()
