#!/usr/bin/env python

import sys
import numpy as np
from pathlib import Path
import argparse
import tempfile
import json
import shutil
import operator

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
        "--graphene-file",
        action="store",
        required=False,
        type=str,
        help="Set graphene data file.",
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
# GRAPHENE_FILE: Path = Path(ARGS.graphene_file)
GRAPHENE_FILE = None
MOL_FILE: Path = Path(ARGS.mol_file)
ELSTOP_TABLE: Path = Path(ARGS.estop_table)

SCRIPT_DIR: Path = Path(ARGS.script_dir)

OMP_THREADS: int = ARGS.omp_threads
MPI_CORES: int = ARGS.mpi_cores

N_RUNS: int = ARGS.runs
IS_MULTIFALL: bool = False

C60_X: float = 0
C60_Y: float = 0
C60_WIDTH: int = 20

CRYSTAL_X: float = 0
CRYSTAL_Y: float = 0

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
    # INPUT_VARS["zero_lvl"] = str(
    #     lammps_util.calc_zero_lvl(INPUT_FILE, SCRIPT_DIR / "in.zero_lvl")
    # )
    INPUT_VARS["zero_lvl"] = str(82.7813)

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
    INPUT_VARS["C60_z_offset"] = str(150)

if "step" not in INPUT_VARS:
    INPUT_VARS["step"] = str(1e-3)

if "lattice" not in INPUT_VARS:
    INPUT_VARS["lattice"] = str(5.43)

CLUSTERS_TABLE: Path = OUT_DIR / "clusters_table.txt"
SPUTTER_COUNT_TABLE = OUT_DIR / "sputter_count_table.txt"
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
    write_header("sim_num N_Si N_C N_Sum", SPUTTER_COUNT_TABLE)
    # write_header("sim_num id Si C Sum", COORD_NUM_TABLE)


def get_cluster_dict(
    cluster_atoms_dict: dict[int, list[Atom]]
) -> tuple[dict[int, Cluster], set[int]]:
    cluster_dict = dict()
    carbon_sputtered = set()

    for cid, atoms in cluster_atoms_dict.items():
        for atom in atoms:
            if atom.type == C_ATOM_TYPE:
                carbon_sputtered.add(atom.id)
        cluster_dict[cid] = Cluster(atoms, SI_ATOM_TYPE)

    return cluster_dict, carbon_sputtered


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


def get_rim_info(
    rim_atoms: list[Atom], fu_x: float, fu_y: float, sim_num: int, zero_lvl: float
) -> np.ndarray:
    if len(rim_atoms) == 0:
        return np.array([])

    def radius(atom: Atom) -> float:
        dx = atom.x - fu_x
        dy = atom.y - fu_y
        return np.sqrt(dx**2 + dy**2)

    r = np.fromiter(map(radius, rim_atoms), float)
    z = np.fromiter(map(operator.attrgetter("z"), rim_atoms), float)

    return np.array(
        [
            sim_num,
            len(rim_atoms),
            r.mean(),
            r.max(),
            z.mean() - zero_lvl,
            z.max() - zero_lvl,
        ]
    )


def get_carbon(dump_final: Dump, carbon_sputtered: set[int]) -> list[Atom]:
    x = dump_final["x"]
    y = dump_final["y"]
    z = dump_final["z"]
    id = dump_final["id"]
    type = dump_final["type"]

    carbon: list[Atom] = []
    for i, atom_id in enumerate(id):
        if atom_id not in carbon_sputtered and type[i] == C_ATOM_TYPE:
            carbon.append(Atom(x=x[i], y=y[i], z=z[i], id=atom_id))

    return carbon


def get_carbon_hist(carbon: list[Atom], zero_lvl: float) -> np.ndarray:
    z_coords = np.fromiter(map(operator.attrgetter("z"), carbon), float)
    if len(z_coords) == 0:
        return np.asarray([[-0.5, 0.0], [0.5, 0.0]])
    z_coords = np.around(z_coords - zero_lvl, 1)
    right = int(np.ceil(z_coords.max())) + 1
    left = int(np.floor(z_coords.min())) - 1
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


def main() -> None:
    lattice = float(INPUT_VARS["lattice"])
    zero_lvl = float(INPUT_VARS["zero_lvl"])
    c60_z_offset = float(INPUT_VARS["C60_z_offset"])

    energy = float(INPUT_VARS["energy"])
    run_time = int(INPUT_VARS["run_time"])
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
            fu_x = 0
            # fu_x = float(INPUT_VARS["C60_x"])
        else:
            fu_x = rnd_coord(C60_X)
        if "C60_y" in INPUT_VARS and run_i == int(INPUT_VARS["run_i"]):
            fu_y = float(INPUT_VARS["C60_y"])
        else:
            # fu_y = rnd_coord(C60_Y)
            xlo = 59.56073644453781
            xhi = 48.49525382424502
            delta = xhi - xlo
            fu_y = (np.random.rand() * delta) + xlo
        if "crystal_x" in INPUT_VARS and run_i == int(INPUT_VARS["run_i"]):
            crystal_x = float(INPUT_VARS["crystal_x"])
        else:
            crystal_x = 0
            crystal_x = rnd_coord(CRYSTAL_X)
        if "crystal_y" in INPUT_VARS and run_i == int(INPUT_VARS["run_i"]):
            crystal_y = float(INPUT_VARS["crystal_y"])
        else:
            crystal_y = 0
            # crystal_y = rnd_coord(CRYSTAL_Y)

        dump_cluster_path = run_dir / "dump.cluster"
        dump_final_path = run_dir / "dump.final"
        dump_during_path = run_dir / "dump.during"
        dump_crater_path = run_dir / "dump.crater"

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
            "graphene_data": str(GRAPHENE_FILE),
            "lattice": str(lattice),
            "C60_z_offset": str(c60_z_offset),
            "C60_y": str(fu_y),
            "C60_x": str(fu_x),
            "crystal_x": str(crystal_x),
            "crystal_y": str(crystal_y),
            "step": str(step),
            "temperature": str(temperature),
            "energy": str(energy),
            "zero_lvl": str(zero_lvl),
            "run_time": str(run_time),
            "dump_final": str(dump_final_path),
            "dump_during": str(dump_during_path),
            "write_file": str(write_file),
            "energy_file": str(run_dir / "energy.txt"),
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
            sys.exit(1)

        dump_final = Dump(dump_final_path)
        lammps_util.create_clusters_dump(
            dump_final.name, dump_final.timesteps[0][0], dump_cluster_path
        )
        dump_cluster = Dump(dump_cluster_path)

        cluster_atoms_dict, rim_atoms = lammps_util.get_cluster_atoms_dict(dump_cluster)
        cluster_dict, carbon_sputtered = get_cluster_dict(cluster_atoms_dict)

        clusters_table = get_clusters_table(cluster_dict, run_num).astype(float)
        rim_info = get_rim_info(rim_atoms, fu_x, fu_y, run_num, zero_lvl)

        carbon = get_carbon(dump_final, carbon_sputtered)
        carbon_hist = get_carbon_hist(carbon, zero_lvl)
        carbon_info = get_carbon_info(carbon, fu_x, fu_y, run_num)

        ids_to_delete: list[int] = []
        for atoms in cluster_atoms_dict.values():
            for atom in atoms:
                ids_to_delete.append(atom.id)

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

        if not IS_MULTIFALL:
            lammps_util.create_crater_dump(
                dump_crater_path,
                dump_final,
                input_file,
                offset_x=crystal_x,
                offset_y=crystal_y,
            )
            dump_crater = Dump(dump_crater_path)
            crater_info = lammps_util.get_crater_info(dump_crater, run_num, zero_lvl)
            lammps_util.save_table(CRATER_TABLE, crater_info, mode="a")

        lammps_util.save_table(CLUSTERS_TABLE, clusters_table, mode="a")
        lammps_util.save_table(RIM_TABLE, rim_info, mode="a")
        lammps_util.save_table(CARBON_DIST, carbon_hist, header=str(run_num), mode="a")
        lammps_util.save_table(CARBON_TABLE, carbon_info, mode="a")
        # save_table(run_dir / 'surface_table.txt', surface_data, mode='w')
        lammps_util.save_table(SURFACE_TABLE, [[run_num, sigma]], mode="a")

        # dump_final_no_cluster_path.unlink()

        run_i += 1

    lammps_util.clusters_parse(CLUSTERS_TABLE, N_RUNS)
    lammps_util.clusters_parse_sum(CLUSTERS_TABLE, N_RUNS)
    lammps_util.clusters_parse_angle_dist(CLUSTERS_TABLE, N_RUNS)
    lammps_util.carbon_dist_parse(CARBON_DIST)

    print("*** FINISHED COMPLETELY ***")

    with open(Path("./runs.log"), encoding="utf-8", mode="a") as f:
        f.write(f"{OUT_DIR.name} {INPUT_FILE.name} {energy} {temperature} {N_RUNS}\n")


if __name__ == "__main__":
    main()
