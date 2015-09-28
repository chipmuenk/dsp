#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# LTI_sensor_py.py ====================================================
#
#
# Einfaches Code-Beispiel zum Kapitel "LTI-Systeme im Zeitbereich"
#
# Thema: Abtastung und Filterung eines Sensorsignals
#
# Python-Musterlösung zur Übungsaufgabe "Filterung abgetasteter Signale"
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

#import dsp_fpga_lib as dsp
#------------------------------------------------------------------------
# ... Ende der gem. import-Anweisungen
# ------------ Define variables ----------------
Ts = 1/240     # sampling period
f1 = 50.0        # signal frequency
phi0  = 0        # signal initial phase
tstep = 1e-3     # time step for "analog" signal
Tmax = max(6.0/f1, 10*Ts) # timespan for 6 signal periods or 10 Ts
N_Ts = Tmax / Ts # number of samples in Tmax
# -- Calculate time-vectors and input signals ---
t = arange(0,Tmax,tstep)  # (start,stop,step)
n = arange(0,round(N_Ts)) # sample n, step = 1 
xt = 1.5 + 0.5*cos(2.0*pi*f1*t + phi0) # x(t).
xn = 1.5 + 0.5*cos(2.0*pi*f1*n*Ts + phi0) # x[n]
#xn = zeros(len(xn)); xn[0] = 1 # Dirac-Stoß
# ----- Plot "analog" and sampled signals -----
figure(1); grid(True)    # Turn on grid
xlabel(r'$t$ / s $\rightarrow$')
ylabel(r'$y$ / V $\rightarrow$')
title('$x(t) = 1.5 + \
0.5 \cos(2 \pi t \cdot 50 \mathrm{Hz})$\n\
$x[n] = 1.5 + 0.5 \cos[2 \pi n \cdot 50 / %.1f]$'\
%(1./Ts))

plot(t, xt, 'b-') # x(t) with blue line
stem(n*Ts, xn, linefmt='r-') # x[n], red stems
ylim(-0.1, 2.2)   # set y-limits to ymin, ymax
# horizontal line at y = 1.5
plt.axhline(1.5, linestyle='--') 
plt.subplots_adjust(top=0.88,right=0.95)
# ------- Impulse response ------------------
figure(2); grid(True)
# v hier die Koeeffizienten eintragen
h = [1, 1, 1, 1] # impulse response MA-filter
#h = np.convolve([1,1,1],[1,1,1]) # cascaded filt.
#h = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125] # ~IIR
stem(range(len(h)), h, 'b-') # plot h[n]
xlabel(r'$n \rightarrow$'); 
ylabel(r'$h[n] \rightarrow$')
title(r'Impulsantwort $h[n]$')
# ------- Filtered signal -------------------
figure(3); grid(True)
#yn = np.convolve(xn,h) # convolve & scale
yn = sig.lfilter([1,0],[1, 0.5],xn) # IIR filter
stem(range(len(yn)), yn, 'b') # y[n]
xlabel(r'$n\;\rightarrow$')
ylabel(r'$y[n]\;\rightarrow$')
title('Gefiltertes Signal')
# ------- Print signal and filtered signal ------
print('  n :', end="")
for i in range(10): print('%6d' %(i), end="")
print('\nx[n]:', end="")
for i in range(10): print('%6.2f' %(xn[i]), end="")
print('\ny[n]:', end="")
for i in range(10): print('%6.2f' %(yn[i]), end="")
plt.show()        # draw and show the plots