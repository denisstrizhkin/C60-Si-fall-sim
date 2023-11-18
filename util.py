from pathlib import Path
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import sys
import os
import subprocess
import time


class Dump:
    def __init__(self, dump_path: Path):
        self.data = np.loadtxt(dump_path, ndmin=2, skiprows=9)

        with open(dump_path, 'r', encoding='utf-8') as file:
          self.keys = file.readlines()[8].split()
          self.keys = self.keys[2:]
        print(self.keys)

        self.name = str(dump_path)

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
    def __init__(self, clusters: List[Atom], si_atom_type):
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

            if cluster.type == si_atom_type:
                self.count_Si += 1
            else:
                self.count_C += 1

        self.ek = 2 * 5.1875 * 1e-5 * (self.mx ** 2 + self.my ** 2 + self.mz ** 2) / (2 * self.mass)

        self.angle = np.arctan(self.mz / np.sqrt(self.mx ** 2 + self.my ** 2))
        self.angle = 90 - self.angle * 180 / np.pi


def lammps_run(
        in_file: Path,
        vars: List[Tuple[str, str]]=[],
        omp_threads: int=4,
        mpi_cores: int=3,
        log_file: Path=Path('./log.lammps')
    ) -> None:
    mpirun_base = [
        'mpirun', '-np', str(mpi_cores),
        'lmp', '-in', str(in_file)
    ]

    if (omp_threads <= 0):
        args = mpirun_base + [
          '-sf', 'gpu',
          '-pk', 'gpu', '0',
        ]
    else:
        args = mpirun_base + [
          '-sf', 'omp',
          '-pk', 'omp', str(omp_threads),
        ]

    vars_list = [ ]      
    for var in vars:
        vars_list.append('-var')
        vars_list.append(var[0])
        vars_list.append(var[1])
                  
    print('lammps_run:', args)
    run_args = args + vars_list + [ '-log', f'{log_file}' ]
    print('lammps_run:', run_args)

    process = subprocess.Popen(run_args, encoding='utf-8')
    while process.poll() is None:
        time.sleep(1)

    if process.returncode != 0:
        print("lammps_run: FAIL")
        sys.exit()


def calc_surface_values(data: Dump, lattice: float, coeff: float, square: float) -> np.ndarray:
    def get_linspace(left, right):
        return np.linspace(left, right, round((right - left) / square) + 1)

    X = get_linspace(-lattice * coeff, lattice * coeff)
    Y = get_linspace(-lattice * coeff, lattice * coeff)
    Z = np.zeros((len(X) - 1, len(Y) - 1))
    Z[:] = np.nan

    for i in range(len(X) - 1):
        for j in range(len(Y) - 1):
            Z_vals = data["z"][np.where(
                (data['x'] >= X[i]) &
                (data['x'] < X[i + 1]) &
                (data['y'] >= Y[j]) &
                (data['y'] < Y[j + 1])
            )]
            if len(Z_vals) != 0:
                Z[i, j] = Z_vals.max()


    print(f'calc_surface: - NaN: {np.count_nonzero(np.isnan(Z))}')
    def check_value(i, j):
        if i < 0 or j < 0 or i >= len(X) - 1 or j >= len(Y) - 1:
            return np.nan
        return Z[i, j]

    for i in range(len(X) - 1):
        for j in range(len(Y) - 1):
            if Z[i, j] == 0 or Z[i, j] == np.nan:
                neighs = [
                    check_value(i - 1, j - 1),
                    check_value(i - 1, j    ),
                    check_value(i - 1, j + 1),
                    check_value(i + 1, j - 1),
                    check_value(i + 1, j    ),
                    check_value(i + 1, j + 1),
                    check_value(i    , j - 1),
                    check_value(i    , j + 1)
                ]
                Z[i, j] = np.nanmean(neighs)

    return Z


def calc_zero_lvl(input_file: Path, in_path: Path) -> float:
  dump_path = 'dump.temp'
  dump_str = 'x y z'

  base_dir = Path('../')

  lammps_run(
    in_path,
    [
      ('input_file', str(input_file)),
      ('dump_path', str(dump_path)),
      ('dump_str', dump_str)
    ],
  )

  dump = Dump(base_dir / dump_path)
  
  return dump['z'][:].max()


def calc_surface(data: Dump, run_dir: Path, lattice: float, zero_lvl: float):
    SQUARE = lattice / 2
    COEFF = 5
    VMIN = -20
    VMAX = 10


    def plotting(square, run_dir): 
        fig, ax = plt.subplots()
        
        width = len(square) + 1
        x = np.linspace(0, COEFF * lattice * 2, width)
        y = np.linspace(0, COEFF * lattice * 2, width)
        x, y = np.meshgrid(x, y)
        
        ax.set_aspect('equal')
        plt.pcolor(x, y, square, vmin=VMIN, vmax=VMAX, cmap=cm.viridis)
        plt.colorbar()
        plt.savefig(f"{run_dir / 'surface_2d.png'}")
    
    
    def histogram(data, run_dir):
        data = data.flatten()
        desired_bin_size = 5
        num_bins = compute_histogram_bins(data, desired_bin_size)
        fig, ax = plt.subplots()
        n, bins, patches = plt.hist(data, num_bins, facecolor='green', alpha=1)
        plt.xlabel('Z coordinate (Å)')
        plt.ylabel('Count')
        plt.title('Surface atoms depth distribution')
        plt.grid(True)
        plt.savefig(f"{run_dir / 'surface_hist.png'}")
    
    
    def compute_histogram_bins(data, desired_bin_size):
        min_val = np.min(data)
        max_val = np.max(data)
        min_boundary = min_val - min_val % desired_bin_size
        max_boundary = max_val - max_val % desired_bin_size + desired_bin_size
        n_bins = int((max_boundary - min_boundary) / desired_bin_size) + 1
        num_bins = np.linspace(min_boundary, max_boundary, n_bins)
        return num_bins

    Z = calc_surface_values(data, lattice, COEFF, SQUARE) - zero_lvl
          
    n_X = Z.shape[0]
    X = np.linspace(0, n_X - 1, n_X, dtype=int)

    n_Y = Z.shape[1]
    Y = np.linspace(0, n_Y - 1, n_Y, dtype=int)
    print(X.shape)

    def f_Z(i, j):
       return Z[i,j]

    z_all = Z.flatten()
    sigma = np.std(z_all)
    print(f'calc_surface: - D: {sigma}')
    #print(z_data)

    plotting(Z, run_dir)
    histogram(Z, run_dir)

    Xs, Ys = np.meshgrid(X, Y)
    Z = f_Z(Xs, Ys)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Xs * SQUARE, Ys * SQUARE, Z, vmin=VMIN, vmax=VMAX, cmap=cm.viridis)
    SCALE = 2
    ax.set_zlim3d(-60,15)
    plt.savefig(f"{run_dir / 'surface_3d.png'}")

    return sigma


def save_table(filename, table, header="", dtype="f", precision=5, mode='w'):
    fmt_str = ""

    if dtype == "d":
        fmt_str = "%d"
    elif dtype == "f":
        fmt_str = f"%.{precision}f"

    with open(filename, f"{mode}b") as file:
        np.savetxt(file, table, delimiter="\t", fmt=fmt_str, header=header)
