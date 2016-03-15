#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# DFT_plot_signals.py ====================================================
# 
#
# Einfache Plots zum Kapitel "DFT"
#
# Thema: Zeitdiskrete Faltung
#
#
#
# 
# (c) 2016-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
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

n = arange(32) # n = 0 ... len(y)-1
y = np.sin(2* pi* n/20)*np.cos(pi* n/(2*len(n)))#*np.exp(-n/(len(n)))
y = np.concatenate((np.zeros(len(n)), y, np.zeros(len(n))))
#y = np.tile(y,3)
n = arange(32*3)
#xticklabels = n

plt.xkcd(scale=1, length=200, randomness=20) # Aktiviere XKCD - Modus
fig1 = plt.figure(num=1, figsize=(12,3))
ax1 = fig1.add_subplot(111)
#stem(n, y, 'r')
grid(False)
#

#
#ax1.set_xlabel(r'$n \rightarrow$')
#ax1.xaxis.set_label_coords(2, 0.5, transform=ax1.transData)
#ax1.set_ylabel(r'$x[n] \rightarrow$')
#ax1.yaxis.set_label_coords(-0.3, 2.3, transform=ax1.transData)
#ax1.set_ylim([-2,3])
#ax1.set_ylim([-2,2.5])
#

#
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.get_yaxis().set_ticks([])
#ax1.yaxis.set_ticks_position('none')
#ax1.spines['left'].set_position(('data',0))
#ax1.spines['left'].set_linewidth(2)
ax1.xaxis.set_ticks_position('bottom')
ax1.set_xticklabels([])
ax1.spines['bottom'].set_position(('data',0))
ax1.spines['bottom'].set_linewidth(2)
ax1.spines['bottom'].set_zorder(0)
#[line.set_zorder(1) for line in ax1.lines]
#resadjust(ax1, yres = 1)
#for label in ax1.get_xticklabels():
#       label.set_horizontalalignment('left')

if False:
    markerline, stemlines, baseline = ax1.stem(n, y)
    plt.setp(markerline, 'markerfacecolor', 'r', 'markersize', 8, 'marker', 'o')
    plt.setp(stemlines, 'color','r', 'linewidth', 1)
    plt.setp(baseline, 'linewidth', 0) # turn off baseline
else:
    plot(n,y,'r',linewidth=3)
#for label in ax1.get_yticklabels():
#       label.set_verticalalignment('bottom')
#ax1.set_xticklabels(xticklabels, rotation=0, ha='left', minor=False)
#
plt.margins(0.05) # setting xmargin / ymargin individually doesnt work

#ax1.spines['left'].set_smart_bounds(True)
#ax1.spines['bottom'].set_smart_bounds(True)
#ax1.set_title(r'Faltung $y[n] = x[n] \star \{1; 1; 1; 1; 1\}$')

#plt.ticklabel_format(useOffset=False, axis='y') # disable using offset print
#fig1.savefig('D:/Daten/HM/dsvFPGA/Uebungen/HM/2016/img/LTF-IIR_Filter_2nd-xn.pdf')

fig2 = figure(num=2, figsize=(6,6))
ax21 = fig2.add_subplot(211)

Y = fft(y)
F = fftfreq(len(n))
plot(F, abs(Y))
ax21.set_ylabel(r'$|H(\mathrm{e}^{\mathrm{j} 2 \pi F})| \rightarrow$')
#ax2.xaxis.set_label_coords(2, 0.5, transform=ax1.transData)

#ax2.yaxis.set_label_coords(-0.3, 2.3, transform=ax1.transData)
#ax2.set_ylim([-2,3])
ax22 = fig2.add_subplot(212)
plot(F, angle(Y) /pi *180)
ax22.set_ylabel(r'$\angle H(\mathrm{e}^{\mathrm{j} 2 \pi F})\; \mathrm {in} \; \deg \rightarrow$')
ax22.set_xlabel(r'$F \rightarrow$')
#fig2.savefig('D:/Daten/HM/dsvFPGA/Uebungen/HM/2016/img/LTF-FIR_filter_H.pdf')



plt.show()