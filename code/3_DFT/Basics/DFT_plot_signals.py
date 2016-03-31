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

SINGLE = False # single or multiple pulses
NREP = 3 # number of repetitions
DISCRETE = False # discrete time or continous time
PHASE = False
ZEROPAD = False
NFFT = 16

NPAD = NFFT * 99 # amount of zero padding

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

n = arange(NFFT) # n = 0 ... len(y)-1
#y = np.sin(2* pi* n/(NFFT/2)) + np.sin(2* pi* n/(NFFT/4))# *np.cos(pi* n/(2*len(n)))#*np.exp(-n/(len(n)))
y = sig.waveforms.sawtooth(n/NFFT * 2 * pi)
print(y)
if SINGLE:
    y = np.concatenate((np.zeros(len(n)), y, np.zeros(len(n))))
    n = arange(NFFT*3)
else:
    y = np.tile(y,NREP)
    n = arange(NFFT*NREP)
#xticklabels = n

#plt.xkcd(scale=1, length=200, randomness=20) # Aktiviere XKCD - Modus
fig1 = plt.figure(num=1, figsize=(12,3))
ax1 = fig1.add_subplot(111)
#stem(n, y, 'r')f
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
ax1.tick_params(axis='x', direction='inout', width=2, length=7,
                       color='k', labelsize=20, pad=5) # 'inout' 'in'
#[tick.set_zorder(0) for tick in ax1.xaxis.ticklabels]
#[line.set_zorder(1) for line in ax1.lines]
#resadjust(ax1, yres = 1)
#for label in ax1.get_xticklabels():
#       label.set_horizontalalignment('left')

if DISCRETE:
    markerline, stemlines, baseline = ax1.stem(n, y)
    plt.setp(markerline, 'markerfacecolor', 'r', 'markersize', 8, 'marker', 'o')
    plt.setp(stemlines, 'color','b', 'linewidth', 1)
    plt.setp(baseline, 'linewidth', 0) # turn off baseline
else:
    plot(n,y,'r',linewidth=3)
#for label in ax1.get_yticklabels():
#       label.set_verticalalignment('bottom')
#ax1.set_xticklabels(xticklabels, rotation=0, ha='left', minor=False)
#
#plt.margins(0.00) # setting xmargin / ymargin individually doesnt work
plt.tight_layout()

#ax1.spines['left'].set_smart_bounds(True)
#ax1.spines['bottom'].set_smart_bounds(True)
#ax1.set_title(r'Faltung $y[n] = x[n] \star \{1; 1; 1; 1; 1\}$')

#plt.ticklabel_format(useOffset=False, axis='y') # disable using offset print
fig1.savefig('D:/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/Kap03_DFT_DFT.png')

fig2 = figure(num=2, figsize=(6,6))
if PHASE:
    ax21 = fig2.add_subplot(211)
else:
    ax21 = fig2.add_subplot(111)

ax21.spines['top'].set_visible(False)
#ax21.spines['right'].set_visible(False)
ax21.spines['left'].set_visible(False)
ax21.get_yaxis().set_ticks([])
ax21.spines['right'].set_position(('data',0))
#ax1.yaxis.set_ticks_position('none')
#ax1.spines['left'].set_position(('data',0))
#ax1.spines['left'].set_linewidth(2)

ax21.xaxis.set_ticks_position('bottom')
ax21.set_xticklabels([])
ax21.spines['bottom'].set_position(('data',0))
ax21.spines['bottom'].set_linewidth(2)
ax21.spines['bottom'].set_zorder(0)
ax21.tick_params(axis='x', direction='inout', width=2, length=7,
                       color='k', labelsize=20, pad=5) # 'inout' 'in'


if ZEROPAD:
    print(NPAD)
    print(np.zeros(NPAD))
    yn = np.concatenate((y, np.zeros(NPAD)))
else:
    yn = y
    
#Y = fft(yn)[0:len(yn)/2]/len(y)
Y = fftshift(fft(yn)/len(y))
F = fftshift(fftfreq(len(yn)))
#F = arange(len(Y))
#F = fftfreq(len(y))
plot(F, abs(Y))
ax21.set_ylabel(r'$|H(\mathrm{e}^{\mathrm{j} 2 \pi F})| \rightarrow$')
#ax2.xaxis.set_label_coords(2, 0.5, transform=ax1.transData)

#ax2.set_ylim([-2,3])
if PHASE:
    ax22 = fig2.add_subplot(212)
    plot(F, angle(Y) /pi *180)
    ax22.set_ylabel(r'$\angle H(\mathrm{e}^{\mathrm{j} 2 \pi F})\; \mathrm {in} \; \deg \rightarrow$')
    ax22.set_xlabel(r'$F \rightarrow$')
#fig2.savefig('D:/Daten/HM/dsvFPGA/Uebungen/HM/2016/img/LTF-FIR_filter_H.pdf')



plt.show()