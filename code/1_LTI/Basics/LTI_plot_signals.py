#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# LTI_faltung_py.py ====================================================
# 
#
# Einfaches Code-Beispiel zum Kapitel "LTI-Systeme im Zeitbereich"
#
# Thema: Zeitdiskrete Faltung
#
#
#
# 
# (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals, absolute_import # v3line15

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, sqrt, exp, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

#import dsp_fpga_lib as dsp
#-------- ----------------------------------------------------------------
# ... Ende der gem. import-Anweisungen
h = [0.25, 0.5, 0.25] # Impulsantwort
x = [1, 1, 1, 1, 1] # Eingangssignal
#y = np.convolve(x, h)
y = [0, 0.5, 1, 0, 0]
n = arange(len(y))-1 # n = 0 ... len(y)-1
xticklabels = n
fig1 = figure(num=1, figsize=(4,2))
ax1 = fig1.add_subplot(111)
stem(n, y, 'r')
grid(False)
markerline, stemlines, baseline = ax1.stem(n, y, label = '$y[n]$')
plt.setp(markerline, 'markerfacecolor', 'r', 'markersize', 10)
plt.setp(stemlines, 'color','r', 'linewidth', 2)
plt.setp(baseline, 'linewidth', 0) # turn off baseline
#
ax1.set_xlabel(r'$n \rightarrow$')
ax1.set_ylabel(r'$h[n] \rightarrow$')
#
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.spines['left'].set_position(('data',0))
ax1.spines['left'].set_linewidth(2)
ax1.xaxis.set_ticks_position('bottom')
ax1.spines['bottom'].set_position(('data',0))
ax1.spines['bottom'].set_linewidth(2)
#ax1.set_xticklabels(xticklabels, rotation=0, ha='left', minor=False)
#
plt.margins(0.05) # setting xmargin / ymargin individually doesnt work

#ax1.spines['left'].set_smart_bounds(True)
#ax1.spines['bottom'].set_smart_bounds(True)
#ax1.set_title(r'Faltung $y[n] = x[n] \star \{1; 1; 1; 1; 1\}$')

#plt.ticklabel_format(useOffset=False, axis='y') # disable using offset print
fig1.savefig('D:/Daten/test_%s.pdf' %int(len(h)))

plt.show()