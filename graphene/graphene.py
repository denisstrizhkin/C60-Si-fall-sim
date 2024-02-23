from pathlib import Path
import numpy as np

# INPUT PARAMETERS:
a = 1.45  # interatomic distance
nx = 45  # number of repetitions in the x direction
ny = 77  # number of repetitions in the y direction
filename = Path("./graphene.dat")


# Size of the unit cell
A = 3 * a
B = np.sqrt(3) * a

# Coordinates of the 4 atoms in the unit cell
base = np.array(
    [[0.0, 0.0, 0.0], [a / 2, B / 2, 0.0], [A / 2, B / 2, 0.0], [2 * a, 0.0, 0.0]]
)

# Total number of atoms
N = len(base) * nx * ny

# Calculate the coordinates of the atoms in the layer
coords = np.zeros((N, 3))
id = 0
for ix in range(0, nx):
    for iy in range(0, ny):
        for iatom in range(0, len(base)):
            coords[id] = base[iatom] + np.array([ix * A, iy * B, 0])
            id += 1


with open(filename, mode="w") as f:
    f.write(f"graphene a={a}\n")
    f.write(f"{N} atoms\n\n")
    f.write(f"1 atom types\n\n")
    f.write(f"-10 {A*nx + 10} xlo xhi\n")
    f.write(f"-10 {B*ny + 10} ylo yhi\n")
    f.write(f"-10 10 zlo zhi\n\n")
    f.write(f"Masses\n\n")
    f.write(f"1 12.011\n\n")
    f.write(f"Atoms\n\n")
    for i in range(0, N):
        s = np.array2string(coords[i], precision=10, floatmode="fixed").strip("[]")
        f.write(f"{i+1} 1 {s} 0 0 0\n")
