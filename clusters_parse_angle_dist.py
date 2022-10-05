#!/usr/bin/env python3
import sys
from os import path
import numpy as np


def write_header(header_str, table_path):
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# " + header_str + "\n")


def append_table(filename, table, header=""):
    with open(filename, "ab") as file:
        np.savetxt(file, table, delimiter="\t", fmt="%.5f", header=header)


def write_table(filename, table, header):
    write_header(header, filename)
    append_table(filename, table)


def main():
    file_path = sys.argv[1]
    clusters = np.loadtxt(file_path, skiprows=1)

    clusters_simNum_N = clusters[:, :2]
    clusters_simNum_N[:, 1] = clusters[:, 1] + clusters[:, 2]

    clusters_enrg_ang = clusters[:, -2:]
    clusters_enrg_ang[:, 0] /= clusters_simNum_N[:, 1]

    num_bins = (85 - 5) // 10 + 1
    num_sims = 50 + 1

    number_table = np.zeros((num_bins, num_sims))
    energy_table = np.zeros((num_bins, num_sims))

    number_table[:, 0] = np.linspace(5, 85, 9)
    energy_table[:, 0] = np.linspace(5, 85, 9)

    for i in range(0, len(clusters)):
        angle_index = int(np.floor(clusters_enrg_ang[i, 1])) // 10
        sim_index = int(clusters_simNum_N[i, 0])

        if angle_index >= num_bins:
            continue

        number_table[angle_index, sim_index] += clusters_simNum_N[i, 1]
        energy_table[angle_index, sim_index] += clusters_enrg_ang[i, 1]

    header_str_number = "angle N1 N2 N3 ... N50"
    output_path_number = (
        path.splitext(file_path)[0]
        + "_parsed_number_dist"
        + path.splitext(file_path)[1]
    )
    write_table(output_path_number, number_table, header_str_number)

    header_str_energy = "angle E1 E2 E3 ... E50"
    output_path_energy = (
        path.splitext(file_path)[0]
        + "_parsed_energy_dist"
        + path.splitext(file_path)[1]
    )
    write_table(output_path_energy, energy_table, header_str_energy)


if __name__ == "__main__":
    main()
