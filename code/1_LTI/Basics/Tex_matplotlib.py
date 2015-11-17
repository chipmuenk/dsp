# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:15:52 2012

Plot mit TeX-Formatierung der Labels
(LaTeX muss auf dem Rechner installiert sein)
"""

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

rc('text', usetex=True)
plt.figure(1)
ax = plt.axes([0.1, 0.1, 0.8, 0.7])
t = np.arange(0.0, 1.0+0.01, 0.01)
s = np.cos(2*2*np.pi*t)+2
plt.plot(t, s)

plt.xlabel(r'\textbf{Time (s)}')
plt.ylabel(r'\textit{Voltage} (mV)',fontsize=16)
plt.title(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
      fontsize=16, color='r')
plt.grid(True)
plt.savefig('tex_demo')

plt.show()