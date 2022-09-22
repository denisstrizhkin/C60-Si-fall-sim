#!/usr/bin/env python3
import sys
from os import path
import numpy as np


def write_header(header_str, table_path):
    with open(table_path, "w", encoding="utf-8") as f:
        f.write("# " + header_str + "\n")


def append_table(filename, table, header=""):
    with open(filename, "ab") as file:
        np.savetxt(file, table, delimiter="\t", fmt="%d", header=header)


def main():
    file_path = sys.argv[1]
    clusters = np.loadtxt(file_path, skiprows=1)
    clusters = clusters[:, :3]

    clusters_dic = {}
    for cluster in clusters:
        cluster_str = "Si" + str(int(cluster[1])) + "C" + str(int(cluster[2]))
        if not cluster_str in clusters_dic.keys():
            clusters_dic[cluster_str] = {}

        sim_num = int(cluster[0])
        if not cluster[0] in clusters_dic[cluster_str]:
            clusters_dic[cluster_str][sim_num] = 0

        clusters_dic[cluster_str][sim_num] += 1

    total_sims = len(np.unique(clusters[:, 0]))
    total_clusters = len(clusters_dic.keys())

    table = np.zeros((total_sims, total_clusters + 1))
    cluster_index = 0
    for key in clusters_dic.keys():
        for sim_num in clusters_dic[key].keys():
            table[sim_num - 1][cluster_index + 1] = clusters_dic[key][sim_num]
            table[sim_num - 1, 0] = sim_num
        cluster_index += 1

    header_str = "simN\t" + "\t".join(clusters_dic.keys())
    output_path = path.splitext(file_path)[0] + "_parsed" + path.splitext(file_path)[1]

    write_header(header_str, output_path)
    append_table(output_path, table)


if __name__ == "__main__":
    main()
