#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# DFT_plot_signals.py ====================================================
# 
#
# Plots zum Kapitel "DFT": 3 x 3 Plot für die ersten 8 Schwingungen einer DFT
#
# 
#
#
#
# 
# (c) 2016-Apr-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals, absolute_import # v3line15

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, sqrt, exp, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

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


BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
#BASE_DIR = "D:/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
FILENAME = "DFT_NxN" # "DFT" #
FMT = ".svg"

NFFT = 16
OSR = 20 # Oversampling ratio for analog curves


ZEROPAD = False
NPAD = NFFT * 99 # amount of zero padding

# generate time arrays for one signal period
n = arange(NFFT) # discrete samples 0 ... NFFT
t = linspace(0,NFFT,num=NFFT*OSR) # "analog" time 0 ... NFFT in NFFT*OSR steps
#y = np.sin(2* pi* n/(NFFT/2)) + np.sin(2* pi* n/(NFFT/4))# *np.cos(pi* n/(2*len(n)))#*np.exp(-n/(len(n)))

x = ((n - 2) > 0) * exp(-(n-1)/4) * (-1)
x[2] = 1
xt = exp(-((t -2)  > 0)*(t-3)/5)


fig1 = figure(figsize=(4,3), num = 1)
ax1 = fig1.add_subplot(111)
ml_1, sl_1, bl_1 = ax1.stem(n,x)
plt.setp(ml_1, 'markerfacecolor', 'k', 'markersize', 8, 'marker', 's')
plt.setp(sl_1, 'color','b', 'linewidth', 2)
plt.setp(bl_1, 'linewidth', 2) 
ax1.set_xlabel(r'$n \; \rightarrow$')
ax1.set_ylabel(r'$x[n] \; \rightarrow$')
fig1.tight_layout()


X = fft(x)/NFFT

#fig1, axes = plt.subplots(nrows=4, ncols=4)

gs33 = gridspec.GridSpec(3,3)
gs33.update(left = 0.15, wspace=0.1, hspace = 0.1, right = 0.99, top = 0.99)

fig2 =figure(figsize=(8,6), num = 2)

bbox_props = dict(boxstyle="Round, pad=0.3", fc="white", ec="k", lw=0.5)
A = 0.11
COS = True
for i in range(3):
    for j in range(3):
      k = 3*i+j 
      ax_i = fig2.add_subplot(gs33[i,j])
      if not ax_i.is_last_row(): ax_i.set_xticklabels([])
      if not ax_i.is_first_col():ax_i.set_yticklabels([])
      if k == 3: 
          if COS:
              ax_i.set_ylabel(r'$\Re \{X[k]\}\, \cos[2 \pi k n / N_{FFT}] \; \rightarrow$')
          else:
              ax_i.set_ylabel(r'$\Im \{X[k]\}\, \sin[2 \pi k n / N_{FFT}] \; \rightarrow$')

      if k == 7: ax_i.set_xlabel(r'$n \; \rightarrow$')

      ax_i.annotate(r'$ k = %d$'%(k), xy = (0.8, 0.8), xycoords='axes fraction',
                    xytext = (0.94,0.94), size=16, 
                    textcoords='axes fraction', va = 'top', ha = 'right',
                    bbox = bbox_props)
      if COS:
          ml, sl, bl = ax_i.stem(n, X[k].real*cos(2*pi*n/NFFT * k))
      else:
          ml, sl, bl = ax_i.stem(n, X[k].imag*sin(2*pi*n/NFFT * k))
      plt.setp(ml, 'markerfacecolor', 'k', 'markersize', 6, 'marker', 's')
      plt.setp(sl, 'color','b', 'linewidth', 0.5)
      plt.setp(bl, 'linewidth', 0, 'color', 'k') # turn off baseline

      if COS:
          ax_i.plot(t, X[k].real*cos(2*pi*t/NFFT * k), 'r', lw = 1)
      else:
          ax_i.plot(t, X[k].imag*sin(2*pi*t/NFFT * k), 'r', lw = 1)
      ax_i.yaxis.set_major_locator(plt.MaxNLocator(5))
#      ax_i.xaxis.set_major_locator(plt.MaxNLocator(8))
      plt.axhline(y = 0, color='k')
      ax_i.set_ylim([-A, A])

fig2.savefig(BASE_DIR + FILENAME + FMT)

plt.show()