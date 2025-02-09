from mpi4py import MPI
from lammps_mpi4py import LammpsMPI

CLUSTER_AMASS: float = 39.95
CLUSTER_COUNT: int = 500
WIDTH: float = 40


def init() -> str:
    return """
units metal
dimension 3
boundary p p p
atom_style atomic
atom_modify map yes
    """


def box() -> str:
    h = WIDTH / 2
    return f"""
region box block {-h} {h} {-h} {h} {-h} {h}
create_box 1 box
    """


def atoms() -> str:
    r = WIDTH / 2
    return f"""
    region sphere sphere 0.0 0.0 0.0 {r}
    create_atoms 1 random {CLUSTER_COUNT} 123456 sphere
    """


def potential() -> str:
    return f"""
mass 1 {CLUSTER_AMASS}
pair_style  table linear 3000
pair_coeff  1 1 ../../potentials/Ar_potentials/atsim_pot/ArAr39.lmptab Ar-Ar
neigh_modify every 1 delay 0 check no
neigh_modify binsize 0.0
neigh_modify one 4000
"""


def main(lmp: LammpsMPI):
    lmp.commands_string(init())
    lmp.commands_string(box())
    lmp.commands_string(atoms())
    lmp.commands_string(potential())
    s = f"""
minimize 1e-10 1e-10 1000 1000

fix nve all nve
dump during all custom 100 dump.during id x y z

write_data data.pre_Ar_{CLUSTER_COUNT}

run 10000

write_data data.Ar_{CLUSTER_COUNT}
    """
    lmp.commands_string(s)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    lmp = LammpsMPI(comm, 0)
    if comm.Get_rank() == 0:
        main(lmp)
        print("*** FINISHED COMPLETELY ***")
        lmp.close()
    else:
        lmp.listen()
