from pathlib import Path

import numpy as np
import lammps_mpi4py
from lammps_mpi4py import LammpsMPI
import typer
from pydantic import BaseModel
from typing_extensions import Annotated


class Params(BaseModel):
    n_x: int
    n_y: int
    n_z: int
    x_offset: float
    y_offset: float
    z_offset: float
    distance: float
    crystal_file: Path
    cluster_file: Path
    out_file: Path


def thermo_script() -> str:
    return """
thermo 100
thermo_style custom step pe ke etotal temp dt time
"""


def setup_cript(crystal_file: Path) -> str:
    lattice = 5.43
    return f"""
### gpu ###
package gpu 0
suffix gpu

### init ###
units metal
boundary p p m
atom_modify map yes
read_data {crystal_file}

### variables ###
variable fixed_zhi equal 'zlo + {0.5 * lattice}'
variable thermostat_zhi equal 'v_fixed_zhi + {lattice}'

### regions ###
lattice diamond {lattice} orient x 1 0 0 orient y 0 1 0 orient z 0 0 1
region fixed block EDGE EDGE EDGE EDGE EDGE ${{fixed_zhi}} units box

### potentials ###
pair_style tersoff/zbl
pair_coeff * * SiC.tersoff.zbl Si C
neigh_modify every 1 delay 0 check no
neigh_modify binsize 0.0
"""


def create_crystal_script(length: float, width: float, depth: float, temp: float) -> str:
    nvt_steps = 1000
    nve_steps = 1000
    half_width = width / 2
    half_length = length / 2
    return f"""
region symbox block {-half_length} {half_length} {-half_width} {half_width} {-depth} 200 units lattice
create_box 2 symbox

region si  block EDGE EDGE EDGE EDGE EDGE 0 units lattice
region fixed block EDGE EDGE EDGE EDGE EDGE {-depth + 0.5} units lattice

# CREATE ATOMS
create_atoms 1 region si
mass 1 28.08553

# GROUPS
group si type 1
group fixed region fixed

# MINIMIZE
minimize 1e-20 1e-20 1000 1000

# MINIMAL ENERGY
velocity nve create {temp} 4928459 dist gaussian

# FIXES
fix nve all nve
fix fixed fixed setforce 0 0 0

# THERMO
timestep 0.001
reset_timestep 0

{thermo_script()}

# RUN NVT
run {nvt_steps}

# FIXES
unfix nvt
fix nve all nve

# RUN NVE
run {nve_steps}
"""


def generate_straight_coords(count: int, distance: float) -> list[tuple[float, float]]:
    d = distance
    n = count * 2 + 1
    coords = [(0.0, 0.0)] * n * n
    for i in range(0, n):
        for j in range(0, n):
            xa = i - count
            ya = j - count
            x = xa * d
            y = ya * d
            coords[i * n + j] = (x, y)
    return coords


def generate_hexagonal_coords(
    counts: tuple[int, int, int], distance: float
) -> list[tuple[float, float, float]]:
    n_x = counts[0]
    n_y = counts[1]
    n_z = counts[2]
    d = distance
    coords = [(0.0, 0.0, 0.0)] * n_x * n_y * n_z
    for x_i in range(0, n_x):
        for y_i in range(0, n_y):
            for z_i in range(0, n_z):
                x = (x_i * np.sqrt(3) / 2 + (z_i % 2) * np.sqrt(3) / 6) * d
                y = (y_i + (x_i % 2) / 2 + (z_i % 2) / 2) * d
                z = (z_i * np.sqrt(6) / 3) * d
                coords[x_i * n_y * n_z + y_i * n_z + z_i] = (x, y, z)
    return coords


def place_c60_script(
    cluster_file: Path,
    counts: tuple[int, int, int],
    offsets: tuple[float, float, float],
    distance: float,
) -> str:
    zero_lvl = 82.7813
    coords = generate_hexagonal_coords(counts, distance)
    s = [
        f"read_data {cluster_file} add append shift {offsets[0]+x} {offsets[1]+y} {zero_lvl+offsets[2]+z}"
        for x, y, z in coords
    ]
    return "\n".join(s)


def run_script(out_file: Path) -> str:
    step = 1e-3
    steps = 1 * 1000
    return f"""
balance 1.0 shift xyz 10 1.0

### groups ###
group Si type 1
group C type 2
group fixed region fixed

### thermo ###
reset_timestep 0
timestep {step}
thermo 1
thermo_style custom step pe ke etotal temp dt time

### fixes ###
fix balance all balance 100 1.0 shift xyz 10 1.0
fix nve all nve
fix fixed fixed setforce 0 0 0

### dumps ###
dump during all custom 1000 dump_during id type x y z vx vy vz

# MINIMIZE
minimize 1e-20 1e-20 1000 1000

thermo 20
run {steps}

# write_data {out_file}
"""


def run(
    lmp: LammpsMPI,
    crystal_file: Path,
    cluster_file: Path,
    counts: tuple[int, int, int],
    offsets: tuple[float, float, float],
    distance: float,
) -> float:
    lmp.command("clear")
    lmp.commands_string(setup_cript(crystal_file))
    lmp.commands_string(place_c60_script(cluster_file, counts, offsets, distance))
    lmp.commands_string(run_script(Path("/dev/null")))
    ke = lmp.get_thermo("ke")
    if ke is None:
        return float("-inf")
    return ke


class App:
    def __init__(self, params: Params):
        self._params = params

    def __call__(self, lmp: lammps_mpi4py.LammpsMPI):
        # (8.95, 186.22331291560013)
        # (9.0, 126.10238214552007)
        # (9.049999999999999, 166.43218504771124)
        # (9.1, 147.101564814656)
        # (9.149999999999999, 144.47704863146134)
        # (9.2, 165.07213369945526)
        # (9.25, 169.40853292435327)
        start = 9.0
        stop = 9.0
        step = 0.05
        count = int((stop - start) / step) + 1
        distances = [start + i * step for i in range(0, count)]
        energies: list[float] = list()
        for d in distances:
            energy = run(
                lmp,
                self._params.crystal_file,
                self._params.cluster_file,
                (self._params.n_x, self._params.n_y, self._params.n_z),
                (self._params.x_offset, self._params.y_offset, self._params.z_offset),
                self._params.distance,
            )
            energies.append(energy)
        for p in zip(distances, energies):
            print(p)


def main(
    crystal_file: Annotated[Path, typer.Option(help="crytal file at")],
    cluster_file: Annotated[Path, typer.Option(help="cluster file at")],
    out_file: Annotated[Path, typer.Option(help="output file at")],
    n_x: Annotated[int, typer.Option(help="")] = 5,
    n_y: Annotated[int, typer.Option(help="")] = 5,
    n_z: Annotated[int, typer.Option(help="")] = 5,
    x_offset: Annotated[float, typer.Option(help="")] = 0.0,
    y_offset: Annotated[float, typer.Option(help="")] = 0.0,
    z_offset: Annotated[float, typer.Option(help="")] = 5.0,
    distance: Annotated[float, typer.Option(help="")] = 9.0,
):
    """
    Create input file with fullerite layer
    """
    params = Params(
        crystal_file=crystal_file,
        cluster_file=cluster_file,
        out_file=out_file,
        n_x=n_x,
        n_y=n_y,
        n_z=n_z,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
        distance=distance,
    )
    app = App(params)
    lammps_mpi4py.run(app)


if __name__ == "__main__":
    typer.run(main)
