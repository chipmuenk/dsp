#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#===========================================================================
# interpolate_noise.py
#
# Demonstrate interpolation in scipy.
# 
# - plotting with a different gridding
# - removing of noise
# 
#
#
# (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
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

#------------------------------------------------------------------------
# Ende der gemeinsamen Import-Anweisungen   
time_i = linspace(0,1,50) # 50 Zeitpunkte im Intervall [0, 1[
time_o = linspace(time_i[0], time_i[-1], 500) # 500 Punkte im gleichen Zeitraum
data_i = np.sin(2 * pi * 100 * time_i) # generiere Sinussignal
data_inoi = data_i + rnd.randn(50) * 0.2 # füge Störung hinzu

f = intp.UnivariateSpline(time_i, data_i, k = 3, s = 0.2)
# k: spline order, must be <= 5
# s: smoothing factor: s = 0 -> spline interpolates through all data points

data_o = f(time_o)

figure(1)
plot(time_i, data_inoi, 'ro', markersize = 12, label = 'Daten (gestört)')
plot(time_i, data_i, 'r--', label = 'Daten (ideal)')
plot(time_o, data_o, 'o', linestyle = ':', label = 'Daten (interpoliert)',
             color = (0.,0.,1,0.5), markerfacecolor=(0.,0.,1,0.5))
xlabel(r'Zeit in s $\rightarrow$')
ylabel(r'Spannung in V $\rightarrow$')
plt.legend()
             
plt.show()

