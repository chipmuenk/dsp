# -*- coding: utf-8 -*-
"""
===== DFT_period_ML_py.py ==================================================

 Python Musterlösung zu "DFT periodischer Signale mit Python / Matlab"

 Berechnung und Darstellung der DFT in Python

 (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

f_S = 1e4; T_S = 1/f_S
N_FFT = 100; T_mess = N_FFT * T_S
f_a = 1e3; f_b = 1.1e3; DC = 1.
t = arange(0, T_mess, T_S) # start / stop / step
y = DC + 0.5 * sin(2 * pi * t * f_a)\
      + 0.2 * cos(2 * pi * t * f_b)
print ('P = ', np.sum(y**2) * T_S / T_mess)
figure(1) # two-sided spectrum
subplot(212)
Sy = fft(y, N_FFT) / N_FFT # calculate DFT at
# f = [0 ... f_S[ = [0... f_S/2[, [-f_S/2 ... 0[
f = fftfreq(N_FFT, T_S)
# freq. points at [0... f_S/2[, [-f_S/2 ... 0[
stem(f,abs(Sy)); grid(True)
plt.xlim(-2000, 2000); plt.ylim(-0.1, 1.1)
xlabel('f [Hz]->'); plt.ylabel('Y(f) ->');
title('Zweiseitige DFT')
print ('P = ', np.dot(Sy,Sy.conj().T))
subplot(222) # one-sided spectrum [0 ... f_S/2[
Sy = 2 * fft(y, N_FFT) / N_FFT # ... needs x2
Sy[0] = Sy[0] / 2. # adjust DC scaling
f = linspace(0, f_S, N_FFT) # f = 0 ... f_S[
stem(f[0:N_FFT//2],abs(Sy[0:N_FFT//2])) #.. f_S/2[
xlabel('f [Hz]->'); plt.ylabel('Y(f) ->')
plt.axis([-100,2000,-0.1, 1.1])
title('Einseitige DFT')
plt.tight_layout(); grid(True); plt.show()
