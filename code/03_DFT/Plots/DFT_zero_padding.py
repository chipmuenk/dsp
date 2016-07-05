# -*- coding: utf-8 -*-
"""
=== DFT_zero_padding.py ===================================================

 Plots zum Kapitel "DFT": Effekt von Zero-Padding auf die DFT

 (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, sqrt, exp, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

from matplotlib.patches import FancyArrow
import matplotlib.gridspec as gridspec


#mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rc('xtick', labelsize='small', direction='in')#, major.size = 4)
mpl.rc('xtick.major', size = 4)
mpl.rc('ytick', labelsize='small', direction='in')
mpl.rc('ytick.major', size = 4)
mpl.rc('lines', markersize = 6)

mpl.rcParams['ytick.labelsize'] = 'small'

EXPORT =  False
#BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
BASE_DIR = "D:/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
FILENAME = "DFT_zero_pad_" # "DFT" #
FMT = ".svg"

NFFT = 32
OSR = 20 # Oversampling ratio for analog curves
ZEROPAD = 0


# generate time arrays for one signal period
n = arange(NFFT) # discrete samples 0 ... NFFT
t = linspace(0,NFFT,num=NFFT*OSR) # "analog" time 0 ... NFFT in NFFT*OSR steps
#x = sig.gausspulse(n-NFFT/2, fc = 20/NFFT, bw = 0.2)# *np.cos(pi* n/(2*len(n)))#*np.exp(-n/(len(n)))
x = np.sinc((n - NFFT/2)/(NFFT/10))*sin(2*pi*n/4)*sig.windows.tukey(NFFT)
#x = zeros(NFFT)
xt = zeros(NFFT*OSR)
#x = ((n - 2) > 0) * exp(-(n-1)/4) * (-1)
#xt = exp(-((t -2)  > 0)*(t-3)/5)

x = np.concatenate((x,np.zeros(ZEROPAD)))
n = arange(NFFT + ZEROPAD)

fig1 = figure(figsize=(4,3), num = 1)
ax1 = fig1.add_subplot(111)
ml_1, sl_1, bl_1 = ax1.stem(n,x)
plt.setp(ml_1, 'markerfacecolor', 'k', 'markersize', 8, 'marker', 's')
plt.setp(sl_1, 'color','b', 'linewidth', 1)
plt.setp(bl_1, 'linewidth',0)

#ax1.plot(t, xt)
plt.axhline(y = 0, color='k')

ax1.set_xlabel(r'$n \; \rightarrow$')
ax1.set_ylabel(r'$x[n] \; \rightarrow$')
#ax1.set_xlim([-1, NFFT])
#ax1.set_ylim([-1, NFFT+1])

fig1.tight_layout(pad = 0.1)
if EXPORT:
    fig1.savefig(BASE_DIR + FILENAME + str(NFFT+ZEROPAD) + '_xn' + FMT)

X = fft(x)/NFFT
Xt = fft(xt)/NFFT
f = arange(0, NFFT, 1/OSR)

k = fftfreq(NFFT+ZEROPAD)

fig2 =figure(figsize=(4,3), num = 2)
ax_1 = fig2.add_subplot(111)

bbox_props = dict(boxstyle="Round, pad=0.3", fc="white", ec="k", lw=0.5)
A = 1.1

mlm, slm, blm = ax_1.stem(k[0:(NFFT+ZEROPAD)//2], np.abs(X[0:(NFFT+ZEROPAD)//2]))
plt.setp(mlm, 'markerfacecolor', 'k', 'markersize', 6, 'marker', 's')
plt.setp(slm, 'color','b', 'linewidth', 0.5)
plt.setp(blm, 'linewidth', 0, 'color', 'k') # turn off baseline
plot(k[0:(NFFT+ZEROPAD)//2], np.abs(X[0:(NFFT+ZEROPAD)//2]), lw = 3)
ax_1.set_xlabel(r'$F \; \rightarrow$')
ax_1.set_ylabel(r'$| X(F) |   \; \rightarrow$')
#ax_1.plot(f, np.abs(Xt))
plt.axhline(y = 0, color='k')
#ax_1.set_ylim([-0.1, A])
fig2.tight_layout(pad = 0.1)

if EXPORT:
 fig2.savefig(BASE_DIR + FILENAME + str(NFFT+ZEROPAD) + '_Xf' + FMT)

plt.show()