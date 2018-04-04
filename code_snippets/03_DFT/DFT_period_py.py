# -*- coding: utf-8 -*-
"""
==== DFT_Delta_f_py.py ====================================================

Python Aufgabenstellung zu "DFT periodischer Signale mit Python / Matlab"

(c) 2014-Mar-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

f_S = 51e3; T_S = 1. / f_S
N_FFT = 510; t_max = N_FFT * T_S
f_a = 1e3; f_b = 1.1e3
t = arange(0, t_max, T_S)
y = 1 + 0.5* sin(2*pi*t*f_a) + 0.2 * cos(2*pi*t*f_b)
Sy = fft(y,N_FFT)/N_FFT
f = arange(N_FFT)
figure(1); clf()
stem(t, y)
figure(2); clf()
stem(f, abs(Sy)); plt.show()