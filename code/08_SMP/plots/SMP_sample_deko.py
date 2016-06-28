# -*- coding: utf-8 -*-
# SMP_sampled_sine.py ================================================
# Plotte mehrere Sinusfunktionen unterschiedlicher Frequenz und Phase,
# die alle die gleiche abgetastete Sequenz liefern (-> Aliasing!)
# TODO: DFT der ursprünglichen und abgetasteten Signale
# (c) 2013-Dez-2 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#%%=======================================================================
from __future__ import division, print_function

import numpy as np
import numpy.random as rnd
from numpy import sin, cos, tan, angle, pi, array, arange, log10, zeros, \
  linspace, ones, sqrt
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, stem, grid, xlabel, ylabel, \
    subplot, title, clf, xlim, ylim

import dsp_fpga_lib as dsp
#------------------------------------------------------------------------
# ... Ende der Import-Anweisungen
f_1 = 500;    f_2 = 1000;    f_3 = 2500
phi_1 = pi/4; phi_2 = -pi/4; phi_3 = -pi/4
A1 = 1.; A2 = 1.; A3 = 1.
Np = 1.5 # Plotte Np Perioden mit f_1:
OSR = 1.5  # Oversampling Ratio in Bezug auf f_1   

t_min = 0; t_max = t_min + Np / f_1 
N = 120; # Anzahl Datenpunkte pro Periode von f_1

NMax = int(Np*N)

# Erzeuge Vektor mit Np*N aequidistanten Zeitpunkten (Np Perioden von f_1):
t  = linspace(t_min, t_max, NMax+1)
#

NS = int(np.floor(N / (2 * OSR))) # Abtastung alle NS Zeitpunkte
t_S = t[0:NMax:NS] # Vektor mit Sampling-Zeitpunkten
f_S = 2 * f_1 * OSR # Abtastfrequenz
#t_P =  t(1 : NS/8 : 2*N); 
x1 = A1 * cos(f_1*2*pi*t + phi_1)
x2 = A2 * cos(f_2*2*pi*t + phi_2)
x3 = A3 * cos(f_3*2*pi*t + phi_3)
# Abtasten der drei Funktionen

x1_S = x1[0:NMax:NS]
x2_S = x2[0:NMax:NS]
x3_S = x3[0:NMax:NS]
#%% Figure 1: Zeitbereich
fig1 = plt.figure(1) # neue Grafik
plt.clf()
ax1 = fig1.add_subplot(111) 
ax1.plot(t, x1, linewidth = 3, color = 'r', label='$f_1$')
ax1.plot(t, x2, color = (0, 0.4, 0), lw = 1., linestyle='--',label='$f_2$')
ax1.plot(t, x3, color = 'b', linewidth = 1., linestyle='-', label='$f_3$')
plt.legend()

[ml,sl,bl] = ax1.stem(t_S,x1_S)
plt.setp(ml, 'markerfacecolor', 'white', alpha = 0.5, 
         markeredgecolor = 'k', markeredgewidth=3.0, markersize = 12)
plt.setp(sl, linewidth = 3, color = 'k')
plt.setp(bl, 'color', 'white') 

title_string = '$f_1 = %d \mathrm{ Hz},\, f_2 = %d \mathrm{ Hz},\, f_3 = %d \mathrm{ Hz}$ \n \
$\phi_1 = %g \pi,\, \phi_2 = %g \pi,\, \phi_3 = %g \pi$' \
                 % (f_1,f_2,f_3,phi_1 / pi, phi_2 / pi, phi_3 / pi)
#plt.title('Analoge Signale mit gleicher Abgetasteten\n' + title_string, fontsize = 18)
plt.title(title_string, fontsize = 18, ha = 'center', va = 'bottom')
ax1.set_xlabel(r'$t \, \mathrm{/ \, s\,} \rightarrow $', fontsize = 16)
ax1.set_ylabel(r'$x \, \mathrm{/ \, V\,} \rightarrow $', fontsize = 16)
ax1.text(.03, 0.97, r'$f_S = %.1f \mathrm{Hz}$' %(f_S), fontsize=16,
         ha="left", va="top",linespacing=1.5, transform=ax1.transAxes,
         bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='0.9')) 
grid('on')
plt.axhline(0., xmin = 0., xmax = 1, linewidth=2, color='k')
ax1.axis([t_min, t_max, -1.2, 1.2])
plt.tight_layout()
#plt.savefig('D:/Daten/ueb-SMP-Spectra_ML_t.png')

plt.show()