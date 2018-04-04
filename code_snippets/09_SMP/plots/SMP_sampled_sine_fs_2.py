# -*- coding: utf-8 -*-
# SMP_sampled_sine_fs_2.py ================================================
# Plotte mehrere Sinusfunktionen gleicher Frequenz und unterschiedlicher Phase
# und Amplitude, die bei fs = f_1/2 alle die gleiche abgetastete Sequenz bzw. 0
#  liefern -> Aliasing!
#
# (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#%%=======================================================================
from __future__ import division, print_function

import numpy as np
import numpy.random as rnd
from numpy import sin, cos, tan, angle, pi, array, arange, log10, zeros, \
  linspace, ones, sqrt
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, stem, grid, xlabel, ylabel, \
    subplot, title, clf, xlim, ylim

import dsp_fpga_lib as dsp
#------------------------------------------------------------------------

#f_1 = 1000;  f_2 = 1000;         f_3 = 1000
#phi_1 = 0;   phi_2 = -pi/3;      phi_3 = +pi/4
#A_1 = 1.;    A_2 = 1/cos(phi_2); A_3 = 1/cos(phi_3)
#Np = 4 # Plotte Np Perioden mit f_1:
#OSR = 1 # Oversampling Ratio

f_1 = 1000;  f_2 = 1000;         f_3 = 1000
phi_1 = pi/2;   phi_2 = -pi/2;      phi_3 = pi/2
A_1 = 1.;    A_2 = 1; A_3 = 2
Np = 4 # Plotte Np Perioden mit f_1:
OSR = 1 # Oversampling Ratio    

t_min = 0; t_max = t_min + Np / f_1 
N = 120; # Anzahl Datenpunkte pro Periode von f_1
# Erzeuge Vektor mit Np*N aequidistanten Zeitpunkten (Np Perioden von f_1):
t  = linspace(t_min, t_max, Np*N+1)
#

NS = np.floor(N / (2 * OSR)) # Abtastung alle NS Zeitpunkte
t_S = t[0:Np*N:NS] # Vektor mit Sampling-Zeitpunkten
f_S = 2 * f_1 * OSR # Abtastfrequenz
#t_P =  t(1 : NS/8 : 2*N); 
x1 = A_1 * cos(f_1*2*pi*t + phi_1)
x2 = A_2 * cos(f_2*2*pi*t + phi_2)
x3 = A_3 * cos(f_3*2*pi*t + phi_3)
# Abtasten der drei Funktionen
x1_S = x1[0:Np*N:NS]
x2_S = x2[0:Np*N:NS]
x3_S = x3[0:Np*N:NS]
#%% Figure 1: Zeitbereich
fig1 = plt.figure(1) # neue Grafik
plt.clf()
ax1 = fig1.add_subplot(111) 
ax1.plot(t, x1, linewidth = 2, color = 'r', label='$\phi_1$')
ax1.plot(t, x2, color = (0, 0.4, 0), lw = 1.5, linestyle='-',label='$\phi_2$')
ax1.plot(t, x3, color = 'b', linewidth = 1.5, linestyle='-', label='$\phi_3$')
plt.legend()

[ml,sl,bl] = ax1.stem(t_S,x1_S)
plt.setp(ml, 'markerfacecolor', 'white', alpha = 0.5, 
         markeredgecolor = 'k', markeredgewidth=3.0, markersize = 12)
plt.setp(sl, linewidth = 3, color = 'k')
plt.setp(bl, 'color', 'white') 

title_string = '$A_1 = %.3g \mathrm{\,V},\, A_2 = %.3g \mathrm{\,V},\,\
                 A_3 = %.3g \mathrm{\,V}, \
                 \phi_1 = %.3g \pi,\, \phi_2 = %.3g \pi,\, \phi_3 = %.3g \pi$' \
                 % (A_1,A_2,A_3,phi_1 / pi, phi_2 / pi, phi_3 / pi)
#plt.title('Analoge Signale mit gleicher Abgetasteten\n' + title_string, fontsize = 18)
plt.title(title_string, fontsize = 16, ha = 'center', va = 'bottom')
ax1.set_xlabel(r'$t \, \mathrm{/ \, s\,} \rightarrow $', fontsize = 16)
ax1.set_ylabel(r'$x \, \mathrm{/ \, V\,} \rightarrow $', fontsize = 16)
ax1.text(.03, 0.97, r'$f_S \;\;= %d \mathrm{\,Hz}$'%(f_S) + '\n' + \
         r'$f_{sig} = %d \mathrm{\,Hz}$' %(f_1), fontsize=16,
         ha="left", va="top",linespacing=1.5, transform=ax1.transAxes,
         bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='0.9')) 
grid('on')
plt.axhline(0., xmin = 0., xmax = 1, linewidth=2, color='k')
ax1.axis([t_min, t_max, -2, 2])
plt.tight_layout()
#plt.savefig('D:/Daten/ueb-SMP-Spectra_ML_t.png')

plt.show()