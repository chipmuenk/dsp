# -*- coding: utf-8 -*-
# SMP_sampled_sine_alias.py ================================================
# Plotte Sinusfunktion und abgetastete Sequenz um Aliasing zu demonstrieren
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

f_1 = 50
phi_1 = pi/4
A_1 = 1.
Np = 30 # Plotte Np Punkte mit f_S:
f_S = 1000 # Abtastfrequenz  

t_min = 0; t_max = t_min + Np / f_S 
N = 120; # Anzahl Datenpunkte pro Periode von f_1
# Erzeuge Vektor mit Np*N aequidistanten Zeitpunkten (Np Perioden von f_1):
t  = linspace(t_min, t_max, Np*N+1)
#
t_S = arange(t_min, t_max, 1/f_S)  # Vektor mit Sampling-Zeitpunkten

#t_P =  t(1 : NS/8 : 2*N); 
x1 = A_1 * cos(f_1*2*pi*t + phi_1)
# Abtasten der drei Funktionen
x1_S = A_1 * cos(f_1*2*pi*t_S + phi_1)

#%% Figure 1: Zeitbereich
fig1 = plt.figure(1) # neue Grafik
plt.clf()
ax1 = fig1.add_subplot(111) 
ax1.plot(t, x1, linewidth = 3, color = 'r', label='$\phi_1$')

#plt.legend()

[ml,sl,bl] = ax1.stem(t_S,x1_S)
plt.setp(ml, 'markerfacecolor', 'k', marker ='o',
         markeredgecolor = 'k', markeredgewidth=1.0, markersize = 10)
plt.setp(sl, linewidth = 1, color = 'k')
plt.setp(bl, 'color', 'white') 

title_string = '$f_{sig} = %.3g \mathrm{\,Hz},\, f_{S} = %.4g \mathrm{\,Hz}$' \
                 % (f_1,f_S)
#plt.title('Analoge Signale mit gleicher Abgetasteten\n' + title_string, fontsize = 18)
#plt.title(title_string, fontsize = 16, ha = 'center', va = 'bottom')
ax1.set_xlabel(r'$t \, \mathrm{/ \, s\,} \rightarrow $', fontsize = 16)
ax1.set_ylabel(r'$x \, \mathrm{/ \, V\,} \rightarrow $', fontsize = 16)
ax1.text(.03, 0.97, r'$F_{sig} = f_{sig} / f_S =  %.2g$'%(f_1 / f_S), fontsize=20,
         ha="left", va="top",linespacing=1.5, transform=ax1.transAxes,
         bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='0.9')) 
grid('on')
plt.axhline(0., xmin = 0., xmax = 1, linewidth=2, color='k')
ax1.axis([t_min, t_max, -1.2, 1.2])
plt.tight_layout()
#plt.savefig('D:/Daten/ueb-SMP-Spectra_ML_t.png')

plt.show()