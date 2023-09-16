#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt


def plot(data, name):
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(data[:, 0], data[:, 1])
    fig.savefig(name + ".png")
    plt.grid()
    plt.close(fig)


def main():
    #data1 = np.loadtxt("./graph1.txt")
    #plot(data1, "graph1")

    data2 = np.loadtxt("./graph2.txt")
    plot(data2, "graph2")


if __name__ == "__main__":
    main()
