#!/usr/bin/env python

import numpy as np
from pathlib import Path
import argparse
import tempfile
import json
import shutil

import lammps_util
from lammps_util import Dump, Atom, Cluster


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Si bombardment with C60 simulation."
    )

    parser.add_argument(
        "--temperature",
        action="store",
        required=False,
        default=700,
        type=float,
        help="Set temperature of the simulation. (K)",
    )

    parser.add_argument(
        "--energy",
        action="store",
        required=False,
        default=8,
        type=float,
        help="Set fall energy of the simulation. (keV)",
    )

    parser.add_argument(
        "--runs",
        action="store",
        required=False,
        default=2,
        type=int,
        help="Number of simulations to run.",
    )

    parser.add_argument(
        "--run-time",
        action="store",
        required=False,
        default=None,
        type=int,
        help="Run simulation this amount of steps.",
    )

    parser.add_argument(
        "--omp-threads",
        action="store",
        required=False,
        default=2,
        type=int,
        help="Set number of OpenMP threads. (if set to 0 use GPU)",
    )

    parser.add_argument(
        "--mpi-cores",
        action="store",
        required=False,
        default=4,
        type=int,
        help="Set number of MPI cores.",
    )

    parser.add_argument(
        "--results-dir",
        action="store",
        required=False,
        default="./results",
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

    parser.add_argument(
        "--input-vars",
        action="store",
        required=False,
        type=str,
        help="Set input vars.",
    )

    parser.add_argument(
        "--mol-file",
        action="store",
        required=True,
        type=str,
        help="Set C60 molecule file.",
    )

    parser.add_argument(
        "--estop-table",
        action="store",
        required=True,
        type=str,
        help="Set electron stopping table file.",
    )

    parser.add_argument(
        "--script-dir",
        action="store",
        required=True,
        type=str,
        help="Set directory containing input scripts",
    )

    return parser.parse_args()


ARGS = parse_args()

OUT_DIR: Path = Path(ARGS.results_dir)
if not OUT_DIR.exists():
    OUT_DIR.mkdir()

lammps_util.setup_root_logger(OUT_DIR / "run.log")

INPUT_VARS: dict[str, str] = {}
if ARGS.input_vars is not None:
    f_path = Path(ARGS.input_vars)
    with open(f_path, mode="r") as f:
        INPUT_VARS = json.load(f)

INPUT_FILE: Path = Path(ARGS.input_file)
MOL_FILE: Path = Path(ARGS.mol_file)
ELSTOP_TABLE: Path = Path(ARGS.estop_table)

SCRIPT_DIR: Path = Path(ARGS.script_dir)

OMP_THREADS: int = ARGS.omp_threads
MPI_CORES: int = ARGS.mpi_cores

N_RUNS: int = ARGS.runs
IS_MULTIFALL: bool = True

C60_X: float = 0
C60_Y: float = 0
C60_WIDTH: int = 20

IS_ALL_DUMP: bool = True
ALL_DUMP_INTERVAL: int = 20

TMP: Path = Path(tempfile.gettempdir()) / OUT_DIR.name
if not TMP.exists():
    TMP.mkdir()

SI_ATOM_TYPE: int = 1
C_ATOM_TYPE: int = 2

if "run_i" not in INPUT_VARS:
    INPUT_VARS["run_i"] = str(0)

if "zero_lvl" not in INPUT_VARS:
    INPUT_VARS["zero_lvl"] = str(
        lammps_util.calc_zero_lvl(INPUT_FILE, SCRIPT_DIR / "in.zero_lvl")
    )

if "temperature" not in INPUT_VARS:
    INPUT_VARS["temperature"] = str(ARGS.temperature)

if "energy" not in INPUT_VARS:
    INPUT_VARS["energy"] = str(ARGS.energy)

if "run_time" not in INPUT_VARS:
    energy = float(INPUT_VARS["energy"])
    if ARGS.run_time is not None:
        run_time = ARGS.run_time
    elif energy < 8_000:
        run_time = 10_000
    else:
        run_time = int(energy * (5 / 4))
    INPUT_VARS["run_time"] = str(run_time)

if "C60_z_offset" not in INPUT_VARS:
    INPUT_VARS["C60_z_offset"] = str(100)

if "step" not in INPUT_VARS:
    INPUT_VARS["step"] = str(1e-6)

if "lattice" is not INPUT_VARS:
    INPUT_VARS["lattice"] = str(5.43)

CLUSTERS_TABLE: Path = OUT_DIR / "clusters_table.txt"
RIM_TABLE: Path = OUT_DIR / "rim_table.txt"
CARBON_TABLE: Path = OUT_DIR / "carbon_table.txt"
CRATER_TABLE: Path = OUT_DIR / "crater_table.txt"
CARBON_DIST: Path = OUT_DIR / "carbon_dist.txt"
SURFACE_TABLE: Path = OUT_DIR / "surface_table.txt"
COORD_NUM_TABLE: Path = OUT_DIR / "coord_num_table.txt"


def write_header(header_str, table_path):
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# " + header_str + "\n")


if INPUT_VARS["run_i"] == str(0):
    write_header("sim_num N_Si N_C mass Px Py Pz Ek angle", CLUSTERS_TABLE)
    write_header("sim_num N r_mean r_max z_mean z_max", RIM_TABLE)
    write_header("sim_num N r_mean r_max", CARBON_TABLE)
    write_header("sim_num N V S z_mean z_min", CRATER_TABLE)
    write_header("z count", CARBON_DIST)
    write_header("sim_num sigma", SURFACE_TABLE)
    # write_header("sim_num id Si C Sum", COORD_NUM_TABLE)


def extract_ids_var(lmp, name, group):
    ids = lmp.extract_variable(name, group, 1)
    if len(ids) == 0:
        return []
    else:
        return ids[np.nonzero(ids)].astype(int)


def get_cluster_dic(cluster_dump: Dump, zero_lvl: float):
    clusters = cluster_dump["c_clusters"]

    cluster_dic: dict = dict()
    for cluster_id in set(clusters):
        cluster_dic[cluster_id] = []

    x = cluster_dump["x"]
    y = cluster_dump["y"]
    z = cluster_dump["z"]
    vx = cluster_dump["vx"]
    vy = cluster_dump["vy"]
    vz = cluster_dump["vz"]
    mass = cluster_dump["c_mass"]
    type = cluster_dump["type"]
    id = cluster_dump["id"]

    for i in range(0, len(z)):
        cluster = Atom(
            x=x[i],
            y=y[i],
            z=z[i],
            vx=vx[i],
            vy=vy[i],
            vz=vz[i],
            mass=mass[i],
            type=type[i],
            id=id[i],
        )
        cluster_dic[clusters[i]].append(cluster)

    keys_to_delete = []
    for key in cluster_dic.keys():
        for cluster in cluster_dic[key]:
            if cluster.z < (zero_lvl + 2.0):
                keys_to_delete.append(key)
                break

    rim_atoms = []
    for key in keys_to_delete:
        rim_atoms += cluster_dic.pop(key)

    return cluster_dic, rim_atoms


def get_vacancies_group_cmd(lmp):
    vac_ids = lmp.extract_variable("vacancy_id", "Si", 1)
    vac_ids = vac_ids[vac_ids != 0]
    return (
        "group vac id " + " ".join(vac_ids.astype(int).astype(str)),
        len(vac_ids) != 0,
    )


def get_clusters_table(cluster_dic, sim_num):
    table = np.array([])

    for key in cluster_dic.keys():
        cluster = cluster_dic[key]
        table = np.concatenate(
            (
                table,
                np.array(
                    [
                        sim_num,
                        cluster.count_si,
                        cluster.count_c,
                        cluster.mass,
                        cluster.mx,
                        cluster.my,
                        cluster.mz,
                        cluster.ek,
                        cluster.angle,
                    ]
                ),
            )
        )

    return table.reshape((table.shape[0] // 9, 9))


def get_rim_info(rim_atoms, fu_x, fu_y, sim_num, zero_lvl: float) -> np.ndarray:
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
                z.mean() - zero_lvl,
                z.max() - zero_lvl,
            ]
        ]
    )


def get_carbon(dump_final, carbon_sputtered):
    x = dump_final["x"]
    y = dump_final["y"]
    z = dump_final["z"]
    id = dump_final["id"]
    type = dump_final["type"]

    carbon = []
    for i in range(0, len(x)):
        if id[i] not in carbon_sputtered and type[i] == C_ATOM_TYPE:
            carbon.append(Atom(x=x[i], y=y[i], z=z[i], id=id[i]))

    return carbon


def get_carbon_hist(carbon, zero_lvl: float):
    z_coords = []
    for c in carbon:
        z_coords.append(np.around(c.z - zero_lvl, 1))
    z_coords = np.array(z_coords)

    right = int(np.ceil(z_coords.max(initial=float("-inf"))))
    left = int(np.floor(z_coords.min(initial=float("+inf"))))
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


def get_crater_info(dump_crater: Dump, sim_num: int, zero_lvl: float) -> np.ndarray:
    id = dump_crater["id"]
    z = dump_crater["z"]
    clusters = dump_crater["c_clusters"]

    crater_id = np.bincount(clusters.astype(int)).argmax()
    atoms = []
    for i in range(0, len(id)):
        if clusters[i] == crater_id:
            atoms.append(Atom(z=z[i], id=id[i]))

    cell_vol = float(np.median(dump_crater["c_voro_vol[1]"]))
    crater_vol = cell_vol * len(atoms)

    surface_count = 0
    z = []
    for atom in atoms:
        z.append(atom.z)
        if atom.z > -2.4 * 0.707 + zero_lvl:
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
                z.mean() - zero_lvl,
                z.min() - zero_lvl,
            ]
        ]
    )


def main() -> None:
    lattice = float(INPUT_VARS["lattice"])
    zero_lvl = float(INPUT_VARS["zero_lvl"])
    c60_z_offset = float(INPUT_VARS["C60_z_offset"])

    energy = float(INPUT_VARS["energy"])
    run_time = float(INPUT_VARS["run_time"])
    step = float(INPUT_VARS["step"])
    temperature = float(INPUT_VARS["temperature"])

    input_file: Path = INPUT_FILE
    run_i = int(INPUT_VARS["run_i"])

    while run_i < N_RUNS:
        run_num = run_i + 1
        run_dir: Path = OUT_DIR / f"run_{run_num}"
        if not run_dir.exists():
            run_dir.mkdir()

        def rnd_coord(coord: float) -> float:
            return coord + (np.random.rand() * 2 - 1) * lattice * C60_WIDTH

        if "C60_x" in INPUT_VARS and run_i == int(INPUT_VARS["run_i"]):
            fu_x = float(INPUT_VARS["C60_x"])
        else:
            fu_x = rnd_coord(C60_X)

        if "C60_y" in INPUT_VARS and run_i == int(INPUT_VARS["run_i"]):
            fu_y = float(INPUT_VARS["C60_y"])
        else:
            fu_y = rnd_coord(C60_Y)

        vacs_restart_file: Path = TMP / "vacs.restart"

        dump_cluster_path: Path = run_dir / "dump.cluster"
        dump_final_path: Path = run_dir / "dump.final"
        dump_during_path: Path = run_dir / "dump.during"
        dump_crater_path: Path = run_dir / "dump.crater"
        dump_crater_id_path: Path = run_dir / "dump.crater_id"

        log_file: Path = run_dir / "log.lammps"
        write_file = TMP / "tmp.input.data"

        backup_input_file: Path = run_dir / "input.data"
        if input_file != backup_input_file:
            shutil.copy(input_file, backup_input_file)
        input_file = backup_input_file

        vars = {
            "run_i": str(run_i),
            "input_file": str(input_file),
            "mol_file": str(MOL_FILE),
            "elstop_table": str(ELSTOP_TABLE),
            "lattice": str(lattice),
            "Si_top": str(zero_lvl + 0.5),
            "C60_z_offset": str(c60_z_offset),
            "C60_y": str(0),
            "C60_x": str(0),
            "step": str(step),
            "temperature": str(temperature),
            "energy": str(energy),
            "zero_lvl": str(zero_lvl),
            "vacs_restart_file": str(vacs_restart_file),
            "run_time": str(run_time),
            "dump_cluster": str(dump_cluster_path),
            "dump_final": str(dump_final_path),
            "dump_during": str(dump_during_path),
            "dump_crater_id": str(dump_crater_id_path),
            "write_file": str(write_file),
        }

        vars_path: Path = run_dir / "vars.json"
        with open(vars_path, encoding="utf-8", mode="w") as f:
            json.dump(vars, f, indent=2)

        if (
            lammps_util.lammps_run(
                SCRIPT_DIR / "in.fall",
                vars,
                omp_threads=OMP_THREADS,
                mpi_cores=MPI_CORES,
                log_file=log_file,
            )
            != 0
        ):
            continue

        dump_cluster = Dump(dump_cluster_path)
        dump_final = Dump(dump_final_path)

        cluster_dic_atoms, rim_atoms = get_cluster_dic(dump_cluster, zero_lvl)

        cluster_dic = dict()
        carbon_sputtered = set()
        for key in cluster_dic_atoms.keys():
            atoms = cluster_dic_atoms[key]
            for atom in atoms:
                if atom.type == C_ATOM_TYPE:
                    carbon_sputtered.add(atom.id)
            cluster_dic[key] = Cluster(cluster_dic_atoms[key], SI_ATOM_TYPE)

        clusters_table = get_clusters_table(cluster_dic, run_num).astype(float)
        rim_info = get_rim_info(rim_atoms, fu_x, fu_y, run_num, zero_lvl)

        carbon = get_carbon(dump_final, carbon_sputtered)
        carbon_hist = get_carbon_hist(carbon, zero_lvl)
        carbon_info = get_carbon_info(carbon, fu_x, fu_y, run_num)

        ids_to_delete = []
        if len(cluster_dic.keys()) != 0:
            for key in cluster_dic_atoms.keys():
                for atom in cluster_dic_atoms[key]:
                    ids_to_delete.append(atom.id)
            ids_to_delete = list(map(int, ids_to_delete))

        dump_final_no_cluster_path = run_dir / "dump.final_no_cluster"
        lammps_util.dump_delete_atoms(
            dump_final_path, dump_final_no_cluster_path, ids_to_delete
        )
        if IS_MULTIFALL:
            write_file_no_clusters = TMP / "tmp_no_cluster.input.data"
            lammps_util.input_delete_atoms(
                write_file, write_file_no_clusters, ids_to_delete
            )
            input_file = write_file_no_clusters

        dump_final_no_cluster = Dump(dump_final_no_cluster_path)
        sigma = lammps_util.calc_surface(
            dump_final_no_cluster, run_dir, lattice, zero_lvl, C60_WIDTH
        )

        dump_cluster_id = Dump(dump_crater_id_path)
        if len(dump_cluster_id["id"]) > 0 and not IS_MULTIFALL:
            vac_ids = " ".join(map(str, map(int, dump_cluster_id["id"])))

            if (
                lammps_util.lammps_run(
                    SCRIPT_DIR / "in.crater",
                    {
                        "input_file": str(input_file),
                        "dump_crater": str(dump_crater_path),
                        "vac_ids": vac_ids,
                    },
                )
                != 0
            ):
                continue

            dump_crater = Dump(dump_crater_path)
            crater_info = get_crater_info(dump_crater, run_num, zero_lvl)
            lammps_util.save_table(CRATER_TABLE, crater_info, mode="a")

        lammps_util.save_table(CLUSTERS_TABLE, clusters_table, mode="a")
        lammps_util.save_table(RIM_TABLE, rim_info, mode="a")
        lammps_util.save_table(CARBON_DIST, carbon_hist, header=str(run_num), mode="a")
        lammps_util.save_table(CARBON_TABLE, carbon_info, mode="a")
        # save_table(run_dir / 'surface_table.txt', surface_data, mode='w')
        lammps_util.save_table(SURFACE_TABLE, [[run_num, sigma]], mode="a")

        dump_final_no_cluster_path.unlink()

        run_i += 1

    lammps_util.clusters_parse(CLUSTERS_TABLE, N_RUNS)
    lammps_util.clusters_parse_sum(CLUSTERS_TABLE, N_RUNS)
    lammps_util.clusters_parse_angle_dist(CLUSTERS_TABLE, N_RUNS)
    lammps_util.carbon_dist_parse(CARBON_DIST)

    lammps_util.create_archive(OUT_DIR)
    print("*** FINISHED COMPLETELY ***")

    with open(Path("./runs.log"), encoding="utf-8", mode="a") as f:
        f.write(f"{OUT_DIR.name} {INPUT_FILE.name} {energy} {temperature} {N_RUNS}\n")


if __name__ == "__main__":
    main()
