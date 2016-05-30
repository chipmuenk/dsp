# -*- coding: utf-8 -*-
# INP_plot_si.py ====================================================
# 
#
# Einfache Plots zum Kapitel "INP": Verlauf der si-Funktion
#
# 
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

#import dsp_fpga_lib as dsp
#-------- ----------------------------------------------------------------
# ... Ende der gem. import-Anweisungen
from scipy.special import diric
EXPORT =  True
BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
#BASE_DIR = "D:/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
FILENAME = "INP_si" # "DFT" #
FMT = ".svg"
FIGSIZE = (11,4)
DIRAC = False
VIEW = False
GRID = True
dB = True


N = 1000

def y(f, dB = False):
#    y = np.abs(diric(t*pi/4,8))
    y = np.abs(np.sinc(f))
    if dB:
        return 20*log10(y)
    else:
        return y
# generate time array
f = linspace(0, 3.5, num=N) # "analog" time in N steps


fig1 = plt.figure(num=1, figsize=FIGSIZE, facecolor = 'white')
ax1 = fig1.add_subplot(111)

ax1.grid(GRID)

ax1.plot(f, y(f, dB = dB) ,color = 'b', lw = 3)   

f_max = 1.5 - 1./(1.5 * pi**2) # approximation for max. of first sidelobe
y_max = y(f_max, dB = dB)
y_05 = y(0.5, dB = dB)


ax1.plot(0.5, y(0.5, dB = dB), 'o', markerfacecolor=(0.5,1,0.5,0.7), markersize = 15,
      markeredgecolor = 'darkgreen')# mark the edge of the baseband
ax1.plot(f_max, y_max, 'o', markerfacecolor='yellow', alpha = 0.7,  markersize = 15,
      markeredgecolor = 'darkgreen')# mark max. sidelobe

    
ax1.annotate(r'%.1f dB $ \equiv $ %.1f' %(y(0.5, dB = True), y(0.5, dB = False)*100) + '%',
        (0.5, y_05),(0.7, y_05), xycoords='data', ha="left", va="center", size=18,
        arrowprops=dict(arrowstyle="-", facecolor = 'red', edgecolor='black' ))
ax1.annotate(r'%.1f dB $ \equiv $ %.1f' %(y(f_max, dB = True), y(f_max, dB = False)*100) + '%',
        (f_max, y_max),(f_max + 0.4, y_max+5), xycoords='data', ha="left", va="bottom", size=18,
        arrowprops=dict(arrowstyle="-", facecolor = 'red', edgecolor='black' ))
ax1.axvspan(0, 0.5, facecolor='0.8', edgecolor = 'none')# baseband in grey
ax1.axvline(1/2, color='g', linestyle='--', lw = 1.5) # and baseband border
ax1.axhline(0, color = 'k', lw = 1)


if dB:
    ax1.set_ylim(-40, 5)    
else:
    ax1.set_ylim(-0.3, 1.2)   

ax1.set_xlabel(r'$f/f_S \; \rightarrow$', size = 20, ha ="right")
ax1.set_ylabel(r'$|H_{ZOH}(f)| \; \rightarrow$', size = 20)

fig1.tight_layout()

if EXPORT:
    fig1.savefig(BASE_DIR + FILENAME + FMT, dpi =300)



plt.show()