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
        sim_num = int(cluster[0])
        if not sim_num in clusters_dic.keys():
            clusters_dic[sim_num] = {}
            clusters_dic[sim_num]["Si"] = 0
            clusters_dic[sim_num]["C"] = 0
        clusters_dic[sim_num]["Si"] += cluster[1]
        clusters_dic[sim_num]["C"] += cluster[2]

    total_sims = len(clusters_dic.keys())
    table = np.zeros((total_sims, 3))

    keys = list(clusters_dic.keys())
    for i in range(0,len(keys)):
        sim_num = keys[i]
        table[i][0] = keys[i]
        table[i][1] = clusters_dic[sim_num]["Si"]
        table[i][2] = clusters_dic[sim_num]["C"]

    header_str = "simN Si C"
    output_path = path.splitext(file_path)[0] + "_parsed_sum" + path.splitext(file_path)[1]

    write_header(header_str, output_path)
    append_table(output_path, table)


if __name__ == "__main__":
    main()
