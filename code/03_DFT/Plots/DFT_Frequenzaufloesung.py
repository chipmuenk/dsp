# -*- coding: utf-8 -*-
"""
=== DFT_plot_signals.py ===================================================

 Plots zum Kapitel "DFT": Frequenzauflösung in Abhängigkeit von der DFT - Länge

 (c) 2016-Apr-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals, absolute_import

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

EXPORT = False
#BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
BASE_DIR = "D:/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
FILENAME = "DFT_Frequenzaufloesung_" # "DFT" #
FMT = ".svg"

NFFT = 256
OSR = 20 # Oversampling ratio for analog curves
DEL = 0


# generate time arrays for one signal period
n = arange(NFFT) # discrete samples 0 ... NFFT
t = linspace(0,NFFT,num=NFFT*OSR) # "analog" time 0 ... NFFT in NFFT*OSR steps
#y = np.sin(2* pi* n/(NFFT/2)) + np.sin(2* pi* n/(NFFT/4))# *np.cos(pi* n/(2*len(n)))#*np.exp(-n/(len(n)))

x = (sin(2*pi*n/5) + sin(2*pi*n/6)) * 2
xt = zeros(NFFT*OSR)
#x = ((n - 2) > 0) * exp(-(n-1)/4) * (-1)
#xt = exp(-((t -2)  > 0)*(t-3)/5)


X = fft(sig.windows.hann(NFFT)*x)/NFFT * 2
Xt = fft(xt)/NFFT
f = arange(0, NFFT, 1/OSR)

fig1 =figure(figsize=(5,4), num = 1)

bbox_props = dict(boxstyle="Round, pad=0.3", fc="white", ec="k", lw=0.5)
A = 1.1

ax_1 = fig1.add_subplot(111)
mlm, slm, blm = ax_1.stem(n[0:NFFT//2], np.abs(X[0:NFFT//2]))
plt.setp(mlm, 'markerfacecolor', 'k', 'markersize', 6, 'marker', 's')
plt.setp(slm, 'color','b', 'linewidth', 0.5)
plt.setp(blm, 'linewidth', 0, 'color', 'k') # turn off baseline
ax_1.set_xlabel(r'$k \; \rightarrow$')
ax_1.set_ylabel(r'$| X[k] |   \; \rightarrow$')
ax_1.annotate(r'$ N_{FFT} = %d$'%(NFFT), xy = (0.8, 0.8), xycoords='axes fraction',
                    xytext = (0.94,0.94), size=16,
                    textcoords='axes fraction', va = 'top', ha = 'right',
                    bbox = bbox_props)
#ax_1.plot(f, np.abs(Xt))
plt.axhline(y = 0, color='k')
ax_1.set_ylim([-0.1, A])
ax_1.set_xlim([-0.1, NFFT/2])

if EXPORT:
    fig1.savefig(BASE_DIR + FILENAME + 'mag_' + str(NFFT) + FMT)

plt.show()