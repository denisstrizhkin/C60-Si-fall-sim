#!/usr/bin/env python

import json
import shutil
import subprocess
from pathlib import Path
from typing import Annotated, TypeVar

import lammps_mpi4py
import matplotlib.pyplot as plt
import numpy as np
import typer
from pydantic import BaseModel, ConfigDict, Field

SI_ATOM_TYPE: int = 1
C_ATOM_TYPE: int = 2

C60_WIDTH: int = 20
CLUSTER_AMASS: float = 12.011
CLUSTER_COUNT: int = 60


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
    energy_file: Path | None = None
    cluster_xyz_file: Path

    cluster_file: Path
    elstop_table: Path
    graphene_data: str | None = None


class State(BaseModel):
    run_vars: RunVars
    is_multifall: bool


T = TypeVar("T", bound=BaseModel)


def save_model(model: T, path: Path):
    with open(path, mode="w") as f:
        json.dump(json.loads(model.model_dump_json()), f, indent=2)


def load_model(path: Path, model_class: type[T]) -> T:
    with open(path) as f:
        return model_class.model_construct(json.load(f))


cli = typer.Typer()


@cli.command()
def rerun(
    results_dir: Annotated[Path, typer.Option()],
    runs: Annotated[
        int, typer.Option(help="Number of simulations to run.")
    ] = 1,
    is_multifall: Annotated[
        bool,
        typer.Option(),
    ] = False,
    n_run: Annotated[int | None, typer.Option()] = None,
    omp_threads: Annotated[
        int,
        typer.Option(
            help="Set number of OpenMP threads. (if set to 0 use GPU)",
        ),
    ] = 2,
):
    state = load_model(results_dir / "state.json", State)
    app = App(
        results_dir=results_dir,
        run_vars=state.run_vars,
        n_runs=runs,
        is_multifall=state.is_multifall,
        omp_threads=omp_threads,
    )
    lammps_mpi4py.run(app)


@cli.command()
def run(
    results_dir: Annotated[
        Path,
        typer.Option(
            help="Set directory path where to store computational results."
        ),
    ],
    input_file: Annotated[Path, typer.Option(help="Set input file.")],
    cluster_file: Annotated[Path, typer.Option(help="Set cluster file.")],
    elstop_table: Annotated[
        Path, typer.Option(help="Set electron stopping table file.")
    ],
    temperature: Annotated[
        float, typer.Option(help="Set temperature of the simulation. (K)")
    ] = 1e-6,
    energy: Annotated[
        float,
        typer.Option(help="Set fall energy of the simulation. (keV)"),
    ] = 8.0,
    angle1: Annotated[float, typer.Option(help="Vector angle 1")] = 0.0,
    angle2: Annotated[float, typer.Option(help="Vector angle 2")] = 0.0,
    runs: Annotated[
        int, typer.Option(help="Number of simulations to run.")
    ] = 1,
    is_multifall: Annotated[
        bool,
        typer.Option(),
    ] = False,
    run_time: Annotated[
        int,
        typer.Option(help="Run simulation this amount of steps."),
    ] = 1000,
    omp_threads: Annotated[
        int,
        typer.Option(
            help="Set number of OpenMP threads. (if set to 0 use GPU)",
        ),
    ] = 2,
):
    run_vars = RunVars.model_construct(
        zero_lvl=82.7813,
        input_file=input_file,
        cluster_file=cluster_file,
        elstop_table=elstop_table,
        temperature=temperature,
        energy=energy,
        angle1=angle1,
        angle2=angle2,
        run_time=run_time,
    )

    app = App(
        results_dir=results_dir,
        run_vars=run_vars,
        n_runs=runs,
        is_multifall=is_multifall,
        omp_threads=omp_threads,
    )
    lammps_mpi4py.run(app)


def get_accelerator_cmds(omp_threads: int) -> list[str]:
    accelerator_cmds: list[str] = []
    if omp_threads > 0:
        accelerator_cmds += [f"package omp {omp_threads}", "suffix omp"]
    else:
        accelerator_cmds += ["package gpu 0 neigh no", "suffix gpu"]
    return accelerator_cmds


class App:
    def __init__(
        self,
        results_dir: Path,
        run_vars: RunVars,
        n_runs: int,
        is_multifall: bool,
        omp_threads: int,
    ):
        self._out_dir = results_dir
        if not self._out_dir.exists():
            self._out_dir.mkdir()
        self._tmp_dir = self._out_dir / "tmp"
        if not self._tmp_dir.exists():
            self._tmp_dir.mkdir()
        self._run_vars = run_vars
        self._n_runs = n_runs
        self._is_mutifall = is_multifall
        self._accelerator_cmds = get_accelerator_cmds(omp_threads)

    def __call__(self, lmp: lammps_mpi4py.LammpsMPI):
        while self._run_vars.run_i < self._n_runs + 1:
            self._run(lmp)
            self._run_vars.run_i += 1

            state = State(
                run_vars=self._run_vars, is_multifall=self._is_mutifall
            )
            save_model(state, self._out_dir / "state.json")

    def _run(self, lmp: lammps_mpi4py.LammpsMPI):
        cos_1 = np.cos(np.radians(self._run_vars.angle1))
        sin_1 = np.sin(np.radians(self._run_vars.angle1))
        cos_2 = np.cos(np.radians(self._run_vars.angle2))
        sin_2 = np.sin(np.radians(self._run_vars.angle2))

        run_dir: Path = self._out_dir / f"run_{self._run_vars.run_i}"
        if not run_dir.exists():
            run_dir.mkdir()

        def rnd_coord(offset: float) -> float:
            return (
                offset
                + (np.random.rand() * 2 - 1)
                * self._run_vars.lattice
                * C60_WIDTH
            )

        def check_run_vars_field(field_name: str) -> bool:
            return not hasattr(self._run_vars, field_name)

        self._run_vars.dump_during = run_dir / "dump.during"
        self._run_vars.dump_initial = run_dir / "dump.initial"
        self._run_vars.dump_final = run_dir / "dump.final"
        self._run_vars.cluster_xyz_file = run_dir / "cluster_xyz.txt"
        self._run_vars.output_file = self._tmp_dir / "tmp.input.data"

        if check_run_vars_field("cluster_position"):
            # self._run_vars.cluster_position.x = rnd_coord(self._run_vars.cluster_position.x)
            # self._run_vars.cluster_position.y = rnd_coord(self._run_vars.cluster_position.y)
            self._run_vars.cluster_position = Vector3D(
                x=self._run_vars.cluster_offset.x,
                y=self._run_vars.cluster_offset.y,
                z=self._run_vars.cluster_offset.z,
            )
            self._run_vars.cluster_position.x += (
                self._run_vars.cluster_position.z * sin_1 * cos_2
            )
            self._run_vars.cluster_position.y += (
                self._run_vars.cluster_position.z * sin_1 * sin_2
            )
            self._run_vars.cluster_position.z *= cos_1
            self._run_vars.cluster_position.z += self._run_vars.zero_lvl

        if check_run_vars_field("cluster_velocity"):
            self._run_vars.cluster_velocity = Vector3D()
            vel = (
                -np.sqrt(
                    self._run_vars.energy
                    * 1000
                    / CLUSTER_AMASS
                    / CLUSTER_COUNT
                )
                * 138.842
            )
            self._run_vars.cluster_velocity.z = vel * cos_1
            self._run_vars.cluster_velocity.x = vel * sin_1 * cos_2
            self._run_vars.cluster_velocity.y = vel * sin_1 * sin_2

        if check_run_vars_field("crystal_offset"):
            self._run_vars.crystal_offset = Vector3D()
            self._run_vars.crystal_offset.x = rnd_coord(
                self._run_vars.crystal_offset.x
            )
            self._run_vars.crystal_offset.y = rnd_coord(
                self._run_vars.crystal_offset.y
            )

        backup_input_file: Path = run_dir / "input.data"
        if self._run_vars.input_file != backup_input_file:
            shutil.copy(self._run_vars.input_file, backup_input_file)
        self._run_vars.input_file = backup_input_file

        run_vars = RunVars.model_validate(self._run_vars)
        save_model(run_vars, run_dir / "vars.json")

        lmp.command(f"log {run_dir / 'log.lammps'}")
        lmp.command("clear")
        lmp.commands_list(self._accelerator_cmds)
        set_lmp_run_vars(lmp, run_vars)
        lmp.file("in.fall")
        plot_cluser_xyz(self._run_vars.cluster_xyz_file)

        if self._is_mutifall:
            write_file_no_clusters = (
                self._tmp_dir / "tmp_no_cluster.input.data"
            )
            subprocess.run(
                [
                    shutil.which("remove-sputtered"),
                    self._run_vars.output_file,
                    self._run_vars.dump_final,
                    write_file_no_clusters,
                ],
                check=True,
            )
            self._run_vars.input_file = write_file_no_clusters


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


def set_lmp_run_vars(lmp: lammps_mpi4py.LammpsMPI, run_vars: RunVars):
    vars = run_vars.model_dump()
    for key in vars:
        value = getattr(run_vars, key)
        if isinstance(value, int | float):
            lmp.command(f"variable {key} equal {value}")
        elif isinstance(value, Vector3D):
            for i in "xyz":
                lmp.command(f"variable {key}_{i} equal {getattr(value, i)}")
        elif value is not None:
            lmp.command(f"variable {key} string {value}")


if __name__ == "__main__":
    cli()
