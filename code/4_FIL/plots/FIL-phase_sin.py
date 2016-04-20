#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# DFT_plot_signals.py ====================================================
# 
#
# Plots zum Kapitel "FIL":
#    Darstellung der Phase eines Sinussignals in Abhängigkeit der Verzögerung
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

from matplotlib.patches import FancyArrow, Circle
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

def scale_axis(val, scale = 0.1):
    delta = abs(max(val) - min(val))
    return [min(val) - delta * scale, max(val) + delta * scale]

PRINT = True            
#BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
BASE_DIR = "D:/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
FILENAME = "FIL_phase_sin_" 
FMT = ".svg"

fsig = 1 # signal frequency in kHz
Tmax = 2#e-3 # max. simulation time

TDel = 1/16 # delay time in ms
N = 1000 # plot points
NPlots = 5 # number of plots

# generate time array

t = linspace(0,Tmax,num=N) # "analog" time 0 ... NFFT in NFFT*OSR steps

bbox_props = dict(boxstyle="Round, pad=0.3", fc="white", ec="k", lw=1)

fig1 = figure(figsize=(4,6), num = 1)
ax1 = fig1.add_subplot(211)
ax1.set_title(r"$\sin \left(2 \pi \cdot %d \mathrm{kHz} (t - \Delta T)\right)$" %(fsig))
for i in arange(NPlots):
    x = sin(2*pi*fsig*(t-i*TDel))
    ax1.plot(t,x, alpha = (NPlots - i)/NPlots, color = 'r',
             label = r"$%.3f$ ms" %(i*TDel))

ax1.legend(fontsize = 14, framealpha = 0.8, title = "$\Delta T$")
plt.axhline(y = 0, color='k', zorder = 0)
plt.axvline(x = 0, color='k', zorder = 0)

ax1.set_xlabel(r'$t / \mathrm{ms} \; \rightarrow$')
ax1.set_ylabel(r'$h(t) \; \rightarrow$')

ax1.set_ylim([-1.1, 1.1])
    
#ax1.text(TDel, 0.5, r'$ N = %d$'%(TDel),
#        horizontalalignment='center',
#        verticalalignment='bottom',
#        fontsize=16, color='k',
#        transform=ax1.transData,
#        bbox = bbox_props)

ax2 = fig1.add_subplot(212)

delta_t = arange(NPlots) * TDel
delta_phi = -delta_t * fsig * 2

ax2.set_xlabel(r'$\Delta T / \mathrm{ms}  \; \rightarrow$')
ax2.set_ylabel(r'$\Delta \phi/ \pi   \; \rightarrow$')

ax2.plot(delta_t, delta_phi, 'ro', linestyle = '-.')

plt.axhline(y = 0, color='k')
plt.axvline(x = 0, color='k', linestyle = '-', zorder = 0, lw = 2)

ax2.set_xlim(scale_axis(delta_t))
ax2.set_ylim(scale_axis(delta_phi))

fig1.tight_layout(pad = 0.1)

if PRINT:
    fig1.savefig(BASE_DIR + FILENAME + str(int(fsig)) + "_kHz" + FMT)
###########################################################################
TDel = 0.25
fsig = 1
f = linspace(1, 2, num = NPlots) * fsig

fig2 = figure(figsize=(4,6), num = 2)
ax21 = fig2.add_subplot(211)
ax21.set_title(r"$\sin \left(f_{sig}  (t - \Delta T)\right)$" )
for i in arange(NPlots):
    x = sin(2 * pi* f[i]*(t-TDel))
    ax21.plot(t,x, alpha = (NPlots - i)/NPlots, color = 'b',
             label = r"$%.2f$ kHz" %(f[i]))

ax21.legend(fontsize = 14, framealpha = 0.8, title = "$f_{sig}$")
plt.axhline(y = 0, color='k', zorder = 0)
plt.axvline(x = 0, color='k', zorder = 0)

ax21.set_xlabel(r'$t / \mathrm{ms} \; \rightarrow$')
ax21.set_ylabel(r'$h(t) \; \rightarrow$')

ax21.set_ylim([-1.1, 1.1])
    


ax22 = fig2.add_subplot(212)

#f = arange(1,NPlots) * fsig
delta_phi = -TDel * f * 2

ax22.set_xlabel(r'$f_{sig} / \mathrm{kHz}  \; \rightarrow$')
ax22.set_ylabel(r'$\Delta \phi/ \pi   \; \rightarrow$')

ax22.plot(f, delta_phi, 'bo', linestyle = '-.')

plt.axhline(y = 0, color='k')
plt.axvline(x = 0, color='k', linestyle = '-', zorder = 0, lw = 2)

ax22.set_xlim(scale_axis(f))
ax22.set_ylim(scale_axis(delta_phi))

ax22.text(max(f), max(delta_phi), r'$ \Delta T = %0.2f$ ms'%(TDel),
        horizontalalignment='right',
        verticalalignment='top',
        fontsize=16, color='k',
#        transform=ax2.transAxes,
        bbox = bbox_props)

fig2.tight_layout(pad = 0.1)
if PRINT:
    fig2.savefig(BASE_DIR + FILENAME + str(int(TDel*1000)) + "_us" + FMT)

plt.show()