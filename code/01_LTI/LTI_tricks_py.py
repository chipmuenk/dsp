# -*- coding: utf-8 -*-
"""
LTI_tricks_py.py ===========================================================

Tricks zu Grafik und LTI-Systemen:
- Anlegen von Subplots
- Definition eines IIR-Filters über seine Koeffizienten
- Impulsantwort impz(), linear und logarithmisch
- filtere (= falte) Eingangssequenz mit Impulsantwort des IIR-Filters
  (unendlich ausgedehnt!) mit scipy.signal.lfilter()
- interpoliere Sequenz mit scipy.interpolate.interp1

(c) 2013-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
============================================================================
"""
from __future__ import division, print_function, unicode_literals

from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, zeros, ones)

import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import sys
sys.path.append('..')
import dsp_fpga_lib as dsp

# -- Impulse response (lin / log) ---
f1 = 50; Ts = 5e-3 
n = arange(0, 50) # sample n
t = arange(0, 49., 0.1) # feiner aufgelöst
xn = 1.5 + 0.5*cos(2.0*pi*f1*n*Ts) # x[n]
b = [0.1, 0]; a = [1, -0.9] # Koeffizienten
#   H(z) = (0.1 z + 0) / (z - 0.9)
[h, k] = dsp.impz(b, a, N = 30) # -> h[k]
figure(1)
subplot(211)
stem(k, h, 'r') # x[n], red stems
ylabel(r'$h[k] \rightarrow$'); grid(True)
title(r'Impulsantwort $h[n]$')
subplot(212)
stem(k, 20*log10(abs(h)), 'r') 
xlabel(r'$k \rightarrow$'); grid(True)
ylabel(r'$20 \log\, h[k] \rightarrow$')
# ------- Filtered signal -------------
figure(2); 
yn = sig.lfilter(b,a,xn) #filter xn with h
f = intp.interp1d(n, yn, kind = 'cubic')
yt = f(t) # y(t), interpolated
plot(t, yt, color='#cc0000', linewidth=3) 
stem(n, yn, 'b') # y[n]
xlabel(r'$n \rightarrow$'); grid(True)
ylabel(r'$y[n] \rightarrow$')
title('Filtered Signal')
plt.show()       # draw and show the plots