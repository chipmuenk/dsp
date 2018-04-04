# -*- coding: utf-8 -*-
"""
LTF_ML_Notchfilter_py.py ==========================================

Kapitel "LTI-Systeme im Frequenzbereich"
Musterloesung zur Aufgabe "Notchfilter" 
 
(c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
=======================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange, 
                   linspace, array, zeros, ones)
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import sys
sys.path.append('..')
import dsp_fpga_lib as dsp

# Definiere Nullstellen auf EK:
N = np.asarray([np.exp(1j*0.3*2*pi), 
                np.exp(-1j*0.3*2*pi)])
# Pole: gleicher Winkel, kleinerer Radius:
P = 0.95 * N
# "Ausmultiplizieren" von P/N ergibt Koeff. 
b = np.poly(N); a = np.poly(P)
figure(1)
dsp.zplane(N,P,zpk = True)
figure(2)
subplot(211)
# Frequenzgang an 2048 Punkten:
[W,H] = sig.freqz(b,a,2048) 
F = W / (2* pi)
plot(F, 20*log10(abs(H))); grid(True)
subplot(212) 
plot(F,angle(H))
xlabel('F ->'); grid(True)
plt.tight_layout()
# Testfreq. (normierte Kreisfreq.):
W_test = array([0, 0.29, 0.3, 0.31, 0.5])*2*pi
# Frequenzgang bei Testfrequenzen:
[H_test, W_test]=sig.freqz(b,a,W_test)
print('H_test =', H_test)
print('|H_test| = ', abs(H_test))
print(20*log10(abs(H_test)))
plt.show()