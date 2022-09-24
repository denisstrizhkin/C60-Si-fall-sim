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

    with open(file_path, "r") as f:
        lines = f.readlines()

    lines_dic = {}
    for i in range(1, len(lines)):
        if lines[i][0] == "#":
            sim_num = int(lines[i].strip().split()[1])
            lines_dic[sim_num] = []
        else:
            lines_dic[sim_num].append(list(map(float, lines[i].strip().split())))

    z_min = 0
    z_max = 0
    for key in lines_dic.keys():
        z_min = min(lines_dic[key][0][0], z_min)
        z_max = max(lines_dic[key][len(lines_dic[key]) - 1][0], z_max)

    bins = np.linspace(z_min, z_max, int(z_max - z_min) + 1)
    table = np.zeros((len(lines_dic), len(bins) + 1))

    sim_nums = list(lines_dic.keys())
    for i in range(0, len(sim_nums)):
        table[i][0] = sim_nums[i]
        for pair in lines_dic[sim_nums[i]]:
            index = int(pair[0] - z_min)
            table[i][index + 1] = pair[1]

    header_str = "simN " + " ".join(list(map(str, bins)))

    output_path = path.splitext(file_path)[0] + "_parsed" + path.splitext(file_path)[1]
    write_header(header_str, output_path)
    append_table(output_path, table)


if __name__ == "__main__":
    main()
