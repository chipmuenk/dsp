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

import pyfda_lib as dsp
#-------- ----------------------------------------------------------------
# ... Ende der gem. import-Anweisungen
def resadjust(ax, xres=None, yres=None):
    """
    Send in an axis and I fix the resolution as desired.
    """

    if xres:
        start, stop = ax.get_xlim()
        ticks = np.arange(start, stop + xres, xres)
        ax.set_xticks(ticks)
    if yres:
        start, stop = ax.get_ylim()
        ticks = np.arange(start, stop + yres, yres)
        ax.set_yticks(ticks)
h = [0.5, 1] # Impulsantwort
x = [1, 1, 1, 1, 1] # Eingangssignal
x = [0, 1, 0, -2, 1, 1, 0]
x = [0, 1, 2, -1, 0]
y = x
#y = np.convolve(x,h)
#y = [0, 0.5, 1, 0, 0]
n = arange(len(y))-2 # n = 0 ... len(y)-1
xticklabels = n
fig1 = figure(num=1, figsize=(6,3))
ax1 = fig1.add_subplot(111)
stem(n, y, 'r')
grid(False)
#
markerline, stemlines, baseline = ax1.stem(n, y, label = '$y[n]$')
plt.setp(markerline, 'markerfacecolor', 'r', 'markersize', 10)
plt.setp(stemlines, 'color','r', 'linewidth', 2)
plt.setp(baseline, 'linewidth', 0) # turn off baseline
#
ax1.set_xlabel(r'$n \rightarrow$')
ax1.xaxis.set_label_coords(2, 0.5, transform=ax1.transData)
ax1.set_ylabel(r'$x[n] \rightarrow$')
ax1.yaxis.set_label_coords(-0.3, 2.3, transform=ax1.transData)
ax1.set_ylim([-2,3])
#ax1.set_ylim([-2,2.5])
#

#
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.yaxis.set_ticks_position('left')
ax1.spines['left'].set_position(('data',0))
ax1.spines['left'].set_linewidth(2)
ax1.xaxis.set_ticks_position('bottom')
ax1.spines['bottom'].set_position(('data',0))
ax1.spines['bottom'].set_linewidth(2)
resadjust(ax1, yres = 1)
for label in ax1.get_xticklabels():
       label.set_horizontalalignment('left')


#for label in ax1.get_yticklabels():
#       label.set_verticalalignment('bottom')
#ax1.set_xticklabels(xticklabels, rotation=0, ha='left', minor=False)
#
plt.margins(0.05) # setting xmargin / ymargin individually doesnt work

#ax1.spines['left'].set_smart_bounds(True)
#ax1.spines['bottom'].set_smart_bounds(True)
#ax1.set_title(r'Faltung $y[n] = x[n] \star \{1; 1; 1; 1; 1\}$')

#plt.ticklabel_format(useOffset=False, axis='y') # disable using offset print
fig1.savefig('D:/Daten/HM/dsvFPGA/Uebungen/HM/2016/img/LTF-IIR_Filter_2nd-xn.pdf')

fig2 = figure(num=2, figsize=(6,6))
ax21 = fig2.add_subplot(211)

W, H = sig.freqz(h)
F = W /(2 * pi)

plot(F, abs(H))
ax21.set_ylabel(r'$|H(\mathrm{e}^{\mathrm{j} 2 \pi F})| \rightarrow$')
#ax2.xaxis.set_label_coords(2, 0.5, transform=ax1.transData)

#ax2.yaxis.set_label_coords(-0.3, 2.3, transform=ax1.transData)
#ax2.set_ylim([-2,3])
ax22 = fig2.add_subplot(212)
plot(F, angle(H) /pi *180)
ax22.set_ylabel(r'$\angle H(\mathrm{e}^{\mathrm{j} 2 \pi F})\; \mathrm {in} \; \deg \rightarrow$')
ax22.set_xlabel(r'$F \rightarrow$')
fig2.savefig('D:/Daten/HM/dsvFPGA/Uebungen/HM/2016/img/LTF-FIR_filter_H.pdf')

fig3 = figure(num=3)
ax3 = fig3.add_subplot(111)
dsp.zplane(h, zpk=False, plt_ax = ax3)
ax3.set_xlabel(r'reell $\rightarrow$')
ax3.set_ylabel(r'imaginär $ \rightarrow$')
fig3.savefig('D:/Daten/HM/dsvFPGA/Uebungen/HM/2016/img/LTF-FIR_filter_PN.pdf')

ax3.set_ylim([-1.1,1.1])
ax3.set_xlim([-2.5,1])

plt.show()