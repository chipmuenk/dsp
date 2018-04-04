# -*- coding: utf-8 -*-
"""
==== DFT_wideband_py.py ===================================================

Python Code zu "DFT von Breitbandsignalen"

Interpretieren Sie das Ergebnis / die Leistungen in Zeit- und Frequenz-
ebene

(c) 2014-Mar-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
============================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

f_S = 5e3; T_S = 1. / f_S
N_FFT = 2000; t_max = N_FFT * T_S
f_a = 1e3; f_b = 1e2; 
A_a = 5; A_b = 1; NQ = 0.01
t = arange(0, t_max, T_S)
y = 1 + A_a * sin(2*pi*t*f_a) + A_b * cos(2*pi*t*f_b)
n = np.sqrt(NQ) * rnd.randn(len(t))
yn = y + n
Syn = fft(yn,N_FFT)
f = arange(N_FFT)
figure(1); clf()
plot(t, yn)
figure(2); clf()
plot(f,abs(Syn)); plt.show()