# -*- coding: utf-8 -*-
"""
=== DFT_basics_ML_py.py =====================================================
 

 Python Musterlösung zu "Fourierreihe und synchrone DFT"

 Berechnung und Darstellung der DFT in Python
 
 

 
 (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

N_FFT = 3; 
f_a = 1e3; T_mess = 1. / f_a
t = linspace(0,T_mess,N_FFT)
xn = 1 + 1 * cos(2*pi*t*f_a)
# calculate DFT and scale it with 1/N:
Xn = fft(xn)/len(xn) 
Xn = fftshift(Xn) # center DFT around f = 0
 # create f-Vector, centered around f = 0:
f = fftshift(fftfreq(len(xn),d=1.0/len(xn)))
# set phase = 0 for very small magnitudes:
for i in range(len(xn)): 
    if abs(Xn[i]/max(abs(Xn))) < 1.0e-10:
        Xn[i] = 0
figure(1)
subplot(211)
stem(f,abs(Xn))
subplot(212)
stem(f,angle(Xn)/pi)
plt.tight_layout(); plt.show()
