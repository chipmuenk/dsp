#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#===========================================================================
# DFT_Delta_f_py.py
#
# Python Aufgabenstellung zu "DFT periodischer Signale mit Python / Matlab"
#
#
#
#
#
#
# (c) 2014-Mar-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

#import dsp_fpga_lib as dsp
#------------------------------------------------------------------ v3line30
# ... Ende der Import-Anweisungen
f_S = 51e3; T_S = 1. / f_S
N_FFT = 128; t_max = N_FFT * T_S
f_a = 1.1e3
t = arange(0, t_max, T_S)
y = sin(2*pi*t*f_a + pi/2)
Sy = fft(y,N_FFT)
f = arange(N_FFT)
figure(1); clf()
plot(t, y)
figure(2); clf()
plot(f, Sy); plt.show()