# -*- coding: utf-8 -*-
"""
==== LTF_IIR_allgemein_py.py =========================================

Code zu Übung "Allgemeine IIR-Struktur": 
   Betragsgang, Impulsantwort und P/N-Diagramm eines IIR-Systems 
      erster Ordnung

(c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
======================================================================
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

alpha = 0.9; f_S = 1 
b = [1, 0] # z + 0
# b = [1 0 0] # z^2 + 0
a = [1, -alpha] # z - 0.9; Add., 1 Verzögerung
#a = [1 +alpha] # z + 0.9; Subtr., 1 Verz.
#a = [1 0 -alpha] # z^2 - 0.9; Add., 2 Verz.
#a = [1 0 +alpha] # z^2 - 0.9; Subtr., 2 Verz.

dsp.zplane(b,a)#, plt_ax = ax)  # Plotte P/N Diagramm
# H(f) entlang der oberen Hälfte des EK:
[W,H] = sig.freqz(b,a,1024, f_S)
fig2 = figure(2)
plot(W/(2*pi),abs(H),linewidth = 2) # H(F)
xlabel(r'$F$  bzw. $\Omega / 2 \pi$') 
ylabel(r'$|H(F)| \; \rightarrow$')
# Berechne 20 Werte der Impulsantwort:
[himp,t] = dsp.impz(b,a,20,f_S) 
figure(3)
(ml, sl, bl) = stem(t,himp) # Impulsantwort
plt.setp(ml,'markerfacecolor','r',
	'markersize',8)
plt.setp(sl,'linewidth',2)
xlabel('$n$'); ylabel(r'$h[n]$')
plt.show()