# -*- coding: utf-8 -*-
"""
=== DFT_dirac.py ==========================================================

 Plots zum Kapitel "DFT": Betrags- und Phasengang des Diracstoßes mit unter-
  schiedlichen Verzögerungen

 (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals, absolute_import # v3line15

import numpy as np
from numpy import (pi, log10, sqrt, exp, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

from matplotlib.patches import FancyArrow
import matplotlib.gridspec as gridspec

#import dsp_fpga_lib as dsp
#-------- ----------------------------------------------------------------
# ... Ende der gem. import-Anweisungen

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
FILENAME = "DFT_dirac_" # "DFT" #
FMT = ".svg"

NFFT = 16
OSR = 20 # Oversampling ratio for analog curves
DEL = 0


# generate time arrays for one signal period
n = arange(NFFT) # discrete samples 0 ... NFFT
t = linspace(0,NFFT,num=NFFT*OSR) # "analog" time 0 ... NFFT in NFFT*OSR steps
#y = np.sin(2* pi* n/(NFFT/2)) + np.sin(2* pi* n/(NFFT/4))# *np.cos(pi* n/(2*len(n)))#*np.exp(-n/(len(n)))

x = zeros(NFFT)
xt = zeros(NFFT*OSR)
#x = ((n - 2) > 0) * exp(-(n-1)/4) * (-1)
x[DEL] = NFFT
xt[DEL] = NFFT
#xt = exp(-((t -2)  > 0)*(t-3)/5)


fig1 = figure(figsize=(3,3), num = 1)
ax1 = fig1.add_subplot(111)
ml_1, sl_1, bl_1 = ax1.stem(n,x)
plt.setp(ml_1, 'markerfacecolor', 'k', 'markersize', 8, 'marker', 's')
plt.setp(sl_1, 'color','b', 'linewidth', 1)
plt.setp(bl_1, 'linewidth',0)

#ax1.plot(t, xt)
plt.axhline(y = 0, color='k')

ax1.set_xlabel(r'$n \; \rightarrow$')
ax1.set_ylabel(r'$x[n] \; \rightarrow$')
ax1.set_xlim([-1, NFFT])
ax1.set_ylim([-1, NFFT+1])

fig1.tight_layout(pad = 0.1)
if EXPORT:
    fig1.savefig(BASE_DIR + FILENAME + '_'+ str(DEL) + FMT )

X = fft(x)/NFFT
Xt = fft(xt)/NFFT
f = arange(0, NFFT, 1/OSR)

#fig1, axes = plt.subplots(nrows=4, ncols=4)

gs33 = gridspec.GridSpec(1,2)
gs33.update(left = 0.12, wspace=0.45, hspace = 0.1, right = 0.98, top = 0.98,
            bottom = 0.18)
fig2 =figure(figsize=(6,3), num = 2)

bbox_props = dict(boxstyle="Round, pad=0.3", fc="white", ec="k", lw=0.5)
A = 1.1

ax_1 = fig2.add_subplot(gs33[0,0])
mlm, slm, blm = ax_1.stem(n, np.abs(X[n]))
plt.setp(mlm, 'markerfacecolor', 'k', 'markersize', 6, 'marker', 's')
plt.setp(slm, 'color','b', 'linewidth', 0.5)
plt.setp(blm, 'linewidth', 0, 'color', 'k') # turn off baseline
ax_1.set_xlabel(r'$k \; \rightarrow$')
ax_1.set_ylabel(r'$| X[k] |   \; \rightarrow$')
#ax_1.plot(f, np.abs(Xt))
plt.axhline(y = 0, color='k')
ax_1.set_ylim([-0.1, A])


ax_2 = fig2.add_subplot(gs33[0,1])
mlp, slp, blp = ax_2.stem(n, np.angle(X[n])/pi)
plt.setp(mlp, 'markerfacecolor', 'k', 'markersize', 6, 'marker', 's')
plt.setp(slp, 'color','b', 'linewidth', 0.5)
plt.setp(blp, 'linewidth', 0, 'color', 'k') # turn off baseline
#ax_2.plot(f, np.angle(Xt)/pi, lw = 0.5)
ax_2.set_xlabel(r'$k \; \rightarrow$')
ax_2.set_ylabel(r'$\angle X[k] / \pi  \; \rightarrow$')
plt.axhline(y = 0, color='k')
ax_2.set_ylim([-A, A])

if EXPORT:
    fig2.savefig(BASE_DIR + FILENAME + 'mag_phi_' + str(DEL) + FMT)

plt.show()