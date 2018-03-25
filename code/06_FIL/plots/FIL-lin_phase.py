# -*- coding: utf-8 -*-
"""
=== FIL-lin_phase ====================================================

(c) 2016-Apr-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"

Demostriere linearphasige Systeme Typ 1 ... 4:
- Impulsantwort
- Betragsgang
- Phasengang
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, sqrt, exp, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

#mpl.rcParams['xtick.labelsize'] = 'small'
mpl.rc('xtick', labelsize='small', direction='in')#, major.size = 4)
mpl.rc('xtick.major', size = 4)
mpl.rc('ytick', labelsize='small', direction='in')
mpl.rc('ytick.major', size = 4)
mpl.rc('lines', markersize = 6)

mpl.rcParams['ytick.labelsize'] = 'small'

EXPORT = False
#BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
BASE_DIR = "D:/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
FILENAME = "FIL_lin_phase_type_4" # "DFT" #
FMT = ".svg"

SYM = False
ASYM = False

NFFT = 32
OSR = 20 # Oversampling ratio for analog curves
ZEROPAD = 200
Ntot = NFFT + ZEROPAD
#HLen = 5 # Typ 1 + 3
HLen = 4 # Typ 2 + 4
# generate time arrays for one signal period
n = arange(NFFT) # discrete samples 0 ... NFFT
t = linspace(0,NFFT,num=NFFT*OSR) # "analog" time 0 ... NFFT in NFFT*OSR steps

n = arange(Ntot)-1
x = np.zeros(Ntot)
#x[1:HLen + 1] = np.ones(HLen) # Typ 1 + 2
for i in range(HLen): x[1+i] = -(i - (HLen-1)/2) # Typ 3 + 4

#x = zeros(NFFT)
xt = zeros(NFFT*OSR)
#x = ((n - 2) > 0) * exp(-(n-1)/4) * (-1)
#xt = exp(-((t -2)  > 0)*(t-3)/5)

#x = np.concatenate((x,np.zeros(ZEROPAD)))

bbox_props = dict(boxstyle="Round, pad=0.3", fc="white", ec="k", lw=1)

fig1 = figure(figsize=(5,8), num = 1)
ax1 = fig1.add_subplot(311)
ml_1, sl_1, bl_1 = ax1.stem(n,x)
plt.setp(ml_1, 'markerfacecolor', 'k', 'markersize', 10, 'marker', 'o')
plt.setp(sl_1, 'color','k', 'linewidth', 2)
plt.setp(bl_1, 'linewidth',0)

#ax1.plot(t, xt)
plt.axhline(y = 0, color='k', zorder = 0)
plt.axvline(x = 0, color='k', zorder = 0)

ax1.set_xlabel(r'$n \; \rightarrow$')
ax1.set_ylabel(r'$h[n] \; \rightarrow$')
ax1.set_xlim([-1.1, HLen + 0.1])
ax1.set_ylim([min(x)-0.5, max(x)*1.5])
#ax1.annotate(r'$ L = %d$'%(HLen), xy = (0.8, 0.8), xycoords='axes fraction',
#              xytext = (HLen,0), size=16,
#                textcoords='axes fraction', va = 'top', ha = 'center',
#                bbox = bbox_props)
#ax1.text((HLen-1)/2,-0.3, "Sym",
#        ha='center', va='center', fontsize=16, color='k',
#        transform=ax1.transData, bbox = bbox_props)

if SYM:
    plt.axvline(x = (HLen-1)/2, color='k', linestyle = '--', zorder = 0, lw = 2)


if ASYM:
    ax1.plot((HLen-1)/2, 0, markerfacecolor='yellow', alpha = 0.8, linestyle = '-', marker = 'o',
             markeredgecolor = 'k',  markeredgewidth = 2, markersize = 20, zorder = 0)

ax1.text(HLen-1, 0.5, r'$ N = %d$'%(HLen-1),
        horizontalalignment='center',
        verticalalignment='bottom',
        fontsize=16, color='k',
        transform=ax1.transData,
        bbox = bbox_props)

ax2 = fig1.add_subplot(312)
X = fft(x)/HLen
Xt = fft(xt)/NFFT
f = arange(0, NFFT, 1/OSR)

k = linspace(0, 1, Ntot)

ax2.plot(k, np.abs(X), lw = 2, color = 'k')
ax2.set_ylabel(r'$| H(F) |   \; \rightarrow$')
#ax_1.plot(f, np.abs(Xt))
plt.axhline(y = 0, color='k')
plt.axvline(x = 0.5, color='k', linestyle = '-.', zorder = 0, lw = 2)
#ax_1.set_ylim([-0.1, A])

ax3 = fig1.add_subplot(313)
ax3.plot(k, np.unwrap(np.angle(X))/pi, lw = 2, color = 'k')
ax3.set_xlabel(r'$F = \Omega/2 \pi \; \rightarrow$')
ax3.set_ylabel(r'$ \angle H(F)/\pi  \; \rightarrow$')

fig1.tight_layout(pad = 0.1)
if EXPORT:
    fig1.savefig(BASE_DIR + FILENAME + FMT)

plt.show()