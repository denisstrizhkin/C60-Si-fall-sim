#!/usr/bin/env python

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline


def main():
  data_C_path = Path('./rdf_C.txt')
  data_Si_path = Path('./rdf_Si.txt')

  data_C = np.loadtxt(data_C_path, skiprows=4)[:,1:]
  data_Si = np.loadtxt(data_Si_path, skiprows=4)[:,1:]

  x = data_C[:,0]
  gr_C = data_C[:,1]
  gr_Si = data_Si[:,1]

  #xnew = np.linspace(x.min(), x.max(), 1000)
  #spl = make_interp_spline(x, gr, k=3)
  #gr_smooth = spl(xnew)

  plt.figure()
  plt.plot(x, gr_C)
  plt.grid()
  plt.savefig('./gr_C.pdf')

  plt.figure()
  plt.plot(x, gr_Si)
  plt.grid()
  plt.savefig('./gr_Si.pdf')


if __name__ == '__main__':
  main()

