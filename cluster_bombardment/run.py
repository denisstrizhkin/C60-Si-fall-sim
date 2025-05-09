#!/usr/bin/env python

from pathlib import Path
import json
import operator
import shutil
from typing import Optional
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings

import lammps_util
from lammps_util import Dump, Atom, Cluster

import lammps_mpi4py
from lammps_mpi4py import LammpsMPI


SI_ATOM_TYPE: int = 1
C_ATOM_TYPE: int = 2
IS_MULTIFALL: bool = False

C60_WIDTH: int = 20
CLUSTER_AMASS: float = 12.011
CLUSTER_COUNT: int = 60


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
    angle1: float = Field(default=0, description="Vector angle 1")
    angle2: float = Field(default=0, description="Vector angle 2")
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


class Vector3D(BaseModel):
    x: float = Field(default=0)
    y: float = Field(default=0)
    z: float = Field(default=0)


class RunVars(BaseModel):
    model_config = ConfigDict(revalidate_instances="always")

    run_i: int = Field(default=1)
    lattice: float = Field(default=5.43)

    cluster_offset: Vector3D = Field(default=Vector3D(z=150))
    cluster_position: Vector3D
    cluster_velocity: Vector3D

    crystal_offset: Vector3D

    step: float = Field(default=1e-3)
    energy: float
    angle1: float
    angle2: float
    temperature: float
    zero_lvl: float
    run_time: int

    input_file: Path
    output_file: Path
    dump_initial: Path
    dump_final: Path
    dump_during: Path
    energy_file: Optional[Path] = None
    cluster_xyz_file: Path

    cluster_file: Path
    elstop_table: Path
    graphene_data: Optional[str] = None


class Tables(BaseModel):
    clusters: Path
    sputter_count: Path
    carbon: Path
    carbon_dist: Path
    rim: Path
    surface: Path
    coord_num: Path


def write_header(header_str, table_path):
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# " + header_str + "\n")


def get_cluster_dict(
    cluster_atoms_dict: dict[int, list[Atom]],
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
    rim_atoms: list[Atom], cluster_pos: Vector3D, sim_num: int, zero_lvl: float
) -> np.ndarray:
    if len(rim_atoms) == 0:
        return np.array([])

    def radius(atom: Atom) -> float:
        dx = atom.x - cluster_pos.x
        dy = atom.y - cluster_pos.y
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


def get_carbon_hist(carbon: list[Atom], zero_lvl: float) -> npt.NDArray[np.float64]:
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


def get_carbon_info(
    carbon: list[Atom], cluster_pos: Vector3D, sim_num: int
) -> npt.NDArray[np.float64]:
    if len(carbon) == 0:
        return np.array([])

    def radius(atom: Atom) -> float:
        return np.sqrt((atom.x - cluster_pos.x) ** 2 + (atom.y - cluster_pos.y) ** 2)

    r = np.fromiter(map(radius, carbon), float)
    return np.array([[sim_num, len(carbon), r.mean(), r.max()]])


def plot_cluser_xyz(path: Path):
    name = path.stem
    path_xz = path.with_name(f"{name}_xz").with_suffix(".png")
    path_xy = path.with_name(f"{name}_xy").with_suffix(".png")
    data = np.loadtxt(path)
    plt.scatter(data[:, 1], data[:, 3])
    plt.savefig(path_xz)
    plt.close()
    plt.scatter(data[:, 1], data[:, 2])
    plt.savefig(path_xy)
    plt.close()


def process_args() -> tuple[RunVars, Path, int, list[str]]:
    args = Arguments()

    out_dir: Path = Path(args.results_dir)
    if not out_dir.exists():
        out_dir.mkdir()

    lammps_util.setup_root_logger(out_dir / "run.log")

    run_vars: RunVars
    if args.input_vars is not None:
        with open(args.input_vars, mode="r") as f:
            run_vars = RunVars.model_construct(json.load(f))
    else:
        run_vars = RunVars.model_construct(
            zero_lvl=82.7813, input_file=Path(args.input_file)
        )

    run_vars.cluster_file = Path(args.cluster_file)
    run_vars.elstop_table = Path(args.elstop_table)
    run_vars.temperature = args.temperature
    run_vars.energy = args.energy
    run_vars.angle1 = args.angle1
    run_vars.angle2 = args.angle2
    run_vars.run_time = args.run_time

    accelerator_cmds: list[str] = []
    if args.omp_threads > 0:
        accelerator_cmds += [f"package omp {args.omp_threads}", "suffix omp"]
    else:
        accelerator_cmds += ["package gpu 0 neigh no", "suffix gpu"]

    return run_vars, out_dir, args.runs, accelerator_cmds


def setup_tables(run_vars: RunVars, out_dir: Path) -> Tables:
    tables = Tables(
        clusters=out_dir / "clusters_table.txt",
        sputter_count=out_dir / "sputter_count_table.txt",
        carbon=out_dir / "carbon_table.txt",
        carbon_dist=out_dir / "carbon_dist.txt",
        rim=out_dir / "rim_table.txt",
        surface=out_dir / "surface_table.txt",
        coord_num=out_dir / "coord_num_table.txt",
    )

    if run_vars.run_i == 1:
        write_header("sim_num N_Si N_C mass Px Py Pz Ek angle", tables.clusters)
        write_header("sim_num N r_mean r_max", tables.carbon)
        write_header("z count", tables.carbon_dist)
        write_header("sim_num N r_mean r_max z_mean z_max", tables.rim)
        write_header("sim_num sigma", tables.surface)
        write_header("sim_num N_Si N_C N_Sum", tables.coord_num)

    return tables


def set_lmp_run_vars(lmp: LammpsMPI, run_vars: RunVars):
    vars = run_vars.model_dump()
    for key in vars.keys():
        value = getattr(run_vars, key)
        if isinstance(value, int) or isinstance(value, int):
            lmp.command(f"variable {key} equal {value}")
        elif isinstance(value, Vector3D):
            for i in "xyz":
                lmp.command(f"variable {key}_{i} equal {getattr(value, i)}")
        elif value is not None:
            lmp.command(f"variable {key} string {value}")


def main(lmp: LammpsMPI) -> None:
    try:
        run_vars, out_dir, n_runs, accelerator_cmds = process_args()
    except SystemExit as e:
        print(e)
        return

    tmp_dir: Path = out_dir / "tmp"
    if not tmp_dir.exists():
        tmp_dir.mkdir()

    tables = setup_tables(run_vars, out_dir)

    cos_1 = np.cos(np.radians(run_vars.angle1))
    sin_1 = np.sin(np.radians(run_vars.angle1))
    cos_2 = np.cos(np.radians(run_vars.angle2))
    sin_2 = np.sin(np.radians(run_vars.angle2))
    for run_num in range(run_vars.run_i, n_runs + 1):
        run_dir: Path = out_dir / f"run_{run_num}"
        if not run_dir.exists():
            run_dir.mkdir()

        def rnd_coord(offset: float) -> float:
            return offset + (np.random.rand() * 2 - 1) * run_vars.lattice * C60_WIDTH

        def check_run_vars_field(field_name: str) -> bool:
            return (not hasattr(run_vars, field_name)) or run_num != run_vars.run_i

        run_vars.dump_during = run_dir / "dump.during"
        run_vars.dump_initial = run_dir / "dump.initial"
        run_vars.dump_final = run_dir / "dump.final"
        run_vars.cluster_xyz_file = run_dir / "cluster_xyz.txt"
        run_vars.output_file = tmp_dir / "tmp.input.data"

        if check_run_vars_field("cluster_position"):
            # run_vars.cluster_position.x = rnd_coord(run_vars.cluster_position.x)
            # run_vars.cluster_position.y = rnd_coord(run_vars.cluster_position.y)
            run_vars.cluster_position = Vector3D(
                x=run_vars.cluster_offset.x,
                y=run_vars.cluster_offset.y,
                z=run_vars.cluster_offset.z,
            )
            run_vars.cluster_position.x += run_vars.cluster_position.z * sin_1 * cos_2
            run_vars.cluster_position.y += run_vars.cluster_position.z * sin_1 * sin_2
            run_vars.cluster_position.z *= cos_1
            run_vars.cluster_position.z += run_vars.zero_lvl

        if check_run_vars_field("cluster_velocity"):
            run_vars.cluster_velocity = Vector3D()
            vel = (
                -np.sqrt(run_vars.energy * 1000 / CLUSTER_AMASS / CLUSTER_COUNT)
                * 138.842
            )
            run_vars.cluster_velocity.z = vel * cos_1
            run_vars.cluster_velocity.x = vel * sin_1 * cos_2
            run_vars.cluster_velocity.y = vel * sin_1 * sin_2

        if check_run_vars_field("crystal_offset"):
            run_vars.crystal_offset = Vector3D()
            run_vars.crystal_offset.x = rnd_coord(run_vars.crystal_offset.x)
            run_vars.crystal_offset.y = rnd_coord(run_vars.crystal_offset.y)

        backup_input_file: Path = run_dir / "input.data"
        if run_vars.input_file != backup_input_file:
            shutil.copy(run_vars.input_file, backup_input_file)
        run_vars.input_file = backup_input_file

        run_vars = RunVars.model_validate(run_vars)
        with open(run_dir / "vars.json", mode="w") as f:
            json.dump(json.loads(run_vars.model_dump_json()), f, indent=2)

        lmp.command(f"log {run_dir / "log.lammps"}")
        lmp.command("clear")
        lmp.commands_list(accelerator_cmds)
        set_lmp_run_vars(lmp, run_vars)
        lmp.file("in.fall")

        plot_cluser_xyz(run_vars.cluster_xyz_file)

        if IS_MULTIFALL:
            write_file_no_clusters = tmp_dir / "tmp_no_cluster.input.data"
            subprocess.run(
                [
                    "remove_sputtered",
                    str(run_vars.input_file),
                    str(run_vars.dump_final),
                    str(write_file_no_clusters),
                ],
                check=True,
            )
            run_vars.input_file = write_file_no_clusters

    # TODO run rust analyzers
    # lammps_util.clusters_parse(tables.clusters, n_runs)
    # lammps_util.clusters_parse_sum(tables.clusters, n_runs)
    # lammps_util.clusters_parse_angle_dist(tables.clusters, n_runs)
    # lammps_util.carbon_dist_parse(tables.carbon_dist)


if __name__ == "__main__":
    lammps_mpi4py.run(main)
