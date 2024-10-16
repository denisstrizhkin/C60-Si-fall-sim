#!/usr/bin/env python

import sys
import numpy as np
from pathlib import Path
import tempfile
import json
import operator
import shutil
from typing import Optional

import lammps_util  # type: ignore
from lammps_util import Dump, Atom, Cluster

from lammps import lammps  # type: ignore
from mpi4py import MPI
from lammps_mpi4py import LammpsMPI

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

SI_ATOM_TYPE: int = 1
C_ATOM_TYPE: int = 2
IS_MULTIFALL: bool = False
C60_WIDTH: int = 20

def hyphenize(field: str):
    return field.replace("_", "-")

class Arguments(BaseSettings, cli_parse_args=True):
    class Config:
        alias_generator = hyphenize

    temperature: float = Field(
        default=1e6, description="Set temperature of the simulation. (K)"
    )
    energy: float = Field(
        default=8, description="Set fall energy of the simulation. (keV)"
    )
    runs: int = Field(default=1, description="Number of simulations to run.")
    run_time: int = Field(
        default=1000, description="Run simulation this amount of steps."
    )
    omp_threads: int = Field(
        default=2, description="Set number of OpenMP threads. (if set to 0 use GPU)"
    )
    results_dir: str = Field(
        description="Set directory path where to store computational results."
    )
    input_file: str = Field(description="Set input file.")
    input_vars: Optional[str] = Field(default=None, description="Set input vars.")
    cluster_file: str = Field(description="Set cluster file.")
    elstop_table: str = Field(description="Set electron stopping table file.")


class RunVars(BaseModel):
    run_i: int = Field(default=0)
    lattice: float = Field(default=5.43)

    C60_x_offset: float = Field(default=0)
    C60_y_offset: float = Field(default=0)
    C60_z_offset: float = Field(default=150)
    C60_x: float
    C60_y: float

    crystal_x: float
    crystal_y: float

    step: float = Field(default=1e-3)
    temperature: float
    energy: float
    zero_lvl: float
    run_time: int

    input_file: Path
    output_file: Path
    dump_final: Path
    dump_during: Path
    dump_crater: Path
    dump_cluster: Path
    energy_file: Optional[Path] = None

    cluster_file: Path
    elstop_table: Path
    graphene_data: Optional[str] = None


class Tables(BaseModel):
    clusters: Path
    sputter_count: Path
    carbon: Path
    carbon_dist: Path
    crater: Path
    rim: Path
    surface: Path
    coord_num: Path


def write_header(header_str, table_path):
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# " + header_str + "\n")


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


def process_args() -> tuple[RunVars, Path, Path, int]:
    args = Arguments()

    out_dir: Path = Path(args.results_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    tmp_dir: Path = out_dir / "tmp"
    if not tmp_dir.exists():
        tmp_dir.mkdir()

    runs: int = args.runs

    lammps_util.setup_root_logger(out_dir / "run.log")

    run_vars: RunVars
    if args.input_vars is not None:
        f_path = Path(args.input_vars)
        with open(f_path, mode="r") as f:
            json_data = json.load(f)
            run_vars = RunVars.model_construct(json_data)
    else:
        run_vars = RunVars.model_construct(
            input_file=Path(args.input_file),
            cluster_file=Path(args.cluster_file),
            elstop_table=Path(args.elstop_table),
            zero_lvl=82.7813,
            temperature=args.temperature,
            energy=args.energy,
        )

    return run_vars, out_dir, tmp_dir, runs


def setup_tables(run_vars: RunVars, out_dir: Path) -> Tables:
    tables = Tables(
        clusters=out_dir / "clusters_table.txt",
        sputter_count=out_dir / "sputter_count_table.txt",
        carbon=out_dir / "carbon_table.txt",
        carbon_dist=out_dir / "carbon_dist.txt",
        crater=out_dir / "crater_table.txt",
        rim=out_dir / "rim_table.txt",
        surface=out_dir / "surface_table.txt",
        coord_num=out_dir / "coord_num_table.txt",
    )

    if run_vars.run_i == 0:
        write_header("sim_num N_Si N_C mass Px Py Pz Ek angle", tables.clusters)
        write_header("sim_num N r_mean r_max", tables.carbon)
        write_header("z count", tables.carbon_dist)
        write_header("sim_num N V S z_mean z_min", tables.crater)
        write_header("sim_num N r_mean r_max z_mean z_max", tables.rim)
        write_header("sim_num sigma", tables.surface)
        write_header("sim_num N_Si N_C N_Sum", tables.coord_num)

    return tables


def main(lmp: LammpsMPI) -> None:
    try:
        run_vars, out_dir, tmp_dir, n_runs = process_args()
    except SystemExit as e:
        print(e)
        return
    
    tables = setup_tables(run_vars, out_dir)
    for run_i in range(run_vars.run_i, n_runs):
        run_num = run_i + 1
        run_dir: Path = out_dir / f"run_{run_num}"
        if not run_dir.exists():
            run_dir.mkdir()

        def rnd_coord(offset: float) -> float:
            return offset + (np.random.rand() * 2 - 1) * run_vars.lattice * C60_WIDTH

        run_vars.dump_during = run_dir / "dump.during"
        run_vars.dump_final = run_dir / "dump.final"
        run_vars.dump_crater = run_dir / "dump.crater"
        run_vars.dump_cluster = run_dir / "dump.cluster"
        run_vars.output_file = tmp_dir / "tmp.input.data"

        if not run_vars.C60_x or run_i != run_vars.run_i:
            run_vars.C60_x = rnd_coord(run_vars.C60_x_offset)

        if not run_vars.C60_y or run_i != run_vars.run_i:
            run_vars.C60_y = rnd_coord(run_vars.C60_y_offset)

        if not run_vars.crystal_x or run_i != run_vars.run_i:
            run_vars.crystal_x = rnd_coord(0)

        if not run_vars.crystal_y or run_i != run_vars.run_i:
            run_vars.crystal_y = rnd_coord(0)

        log_file: Path = run_dir / "log.lammps"

        backup_input_file: Path = run_dir / "input.data"
        if input_file != backup_input_file:
            shutil.copy(input_file, backup_input_file)
        input_file = backup_input_file

        with open(run_dir / "vars.json", mode="w") as f:
            json.dump(run_vars.model_dump_json(), f, indent=2)

        lmp.file("in.fall")

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
        dump_final_no_cluster = Dump(dump_final_no_cluster_path)
        sigma = lammps_util.calc_surface(
            dump_final_no_cluster, run_dir, lattice, zero_lvl, C60_WIDTH
        )
        if IS_MULTIFALL:
            write_file_no_clusters = tmp_dir / "tmp_no_cluster.input.data"
            lammps_util.input_delete_atoms(
                write_file, write_file_no_clusters, ids_to_delete
            )
            input_file = write_file_no_clusters
        else:
            dump_init_path = input_file.parent / "dump.input"
            lammps_util.create_dump_from_input(input_file, dump_init_path)
            dump_init = Dump(dump_init_path)
            input_file_no_block = input_file.with_stem(input_file.stem + "_no_block")
            lammps_util.input_delete_atoms(
                input_file,
                input_file_no_block,
                dump_init["id"][np.where(dump_init["z"] > zero_lvl + 10)],
            )
            lammps_util.create_crater_dump(
                dump_crater_path,
                dump_final_no_cluster,
                input_file_no_block,
                offset_x=crystal_x,
                offset_y=crystal_y,
            )
            dump_crater = Dump(dump_crater_path)
            crater_info = lammps_util.get_crater_info(dump_crater, run_num, zero_lvl)
            lammps_util.save_table(tables.crater, crater_info, mode="a")

        lammps_util.save_table(tables.clusters, clusters_table, mode="a")
        lammps_util.save_table(tables.rim, rim_info, mode="a")
        lammps_util.save_table(
            tables.carbon_dist, carbon_hist, header=str(run_num), mode="a"
        )
        lammps_util.save_table(tables.carbon, carbon_info, mode="a")
        lammps_util.save_table(tables.surface, [[run_num, sigma]], mode="a")

    lammps_util.clusters_parse(tables.clusters, n_runs)
    lammps_util.clusters_parse_sum(tables.clusters, n_runs)
    lammps_util.clusters_parse_angle_dist(tables.clusters, n_runs)
    lammps_util.carbon_dist_parse(tables.carbon_dist)

    print("*** FINISHED COMPLETELY ***")

    with open(Path("./runs.log"), encoding="utf-8", mode="a") as f:
        f.write(
            f"{out_dir.name} {run_vars.input_file.name} {run_vars.energy} {run_vars.temperature} {n_runs}\n"
        )


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    lmp = lammps()
    lmpmpi = LammpsMPI(lmp, comm, 0)
    if comm.Get_rank() == 0:
        main(lmpmpi)
        lmpmpi.close()
    else:
        lmpmpi.listen()
