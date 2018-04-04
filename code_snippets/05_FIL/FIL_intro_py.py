# -*- coding: utf-8 -*-
"""
=== FIL_intro_py.py =======================================================
 

 Demonstrate different filter design methods and compare results
 to specifications
 
 
 
 

 (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import sys
sys.path.append('..')
import dsp_fpga_lib as dsp

f_S = 400 # Samplingfrequenz
f_DB = 40 # Grenzfreq. Durchlassband (DB)
f_SB = 50 # Grenzfrequenz Sperrband (SB)
F_DB = f_DB/(f_S/2) # Frequenzen bezogen
F_SB = f_SB/(f_S/2) # auf HALBE Abtastfreq.
#
A_DB = 0.1 # max. Ripple im DB in dB
A_DB_lin = (10**(A_DB/20.0)-1) / \
  (10**(A_DB/20.0)+1)*2 # und linear
A_SB = 60 # min. Dämpfung im SB in dB
A_SB_lin = 10**(-A_SB/20.0) # und linear
#
L = 44 # Manuelle Vorgabe Filterordnung
############## FIR-Filterentwurf ##########
a = [1] # Nennerpolynom = 1 bei FIR-Filtern
#=== Windowed FIR / Least Square =========
F_c  = f_DB / (f_S/2)  # -6dB - Frequenz
b = sig.firwin(L, F_c) # Hamming-Window
#=== Frequency Sampling ==================
b = sig.firwin2(L, [0, F_DB, F_SB, 1],
           [1, 1, 0, 0])
#=== REMEZ / Parks-McClellan / Equiripple
W_DB = 1;  W_SB = 1 # manuelle Ordnung:
b = sig.remez(L, [0, F_DB, F_SB, 1],
            [0, 1], [W_DB, W_SB], Hz = 2)
# minimale Ordnung:
(L_min,F,A,W) = dsp.remezord([F_DB, F_SB],
    [1, 0], [A_DB_lin, A_SB_lin], Hz = 2)
b = sig.remez(L_min, F, A, W )
##############  IIR-Filterentwurf  ########
#=== Butterworth Filter===================
[Lb,F_b] = sig.buttord(F_DB,F_SB,A_DB,A_SB)
#[b, a] = sig.butter(Lb, F_b)
#=== IIR-Wrapper (nur Python) =====
#[b, a] = sig.iirdesign(F_DB, F_SB,
#              A_DB, A_SB, ftype='ellip')
############################################
print('Filterkoeffizienten:')
print('a = ', a); print('b = ', b)
## Calculate H(w), w = 0 ... pi, 1024 Pts.
[w, H] = sig.freqz(b, a, 1024)
# Translate w to physical frequencies:
f = w / (2 * pi) * f_S
############## Plot the Results #########
## Pol/Nullstellenplan
fig1 = figure(1); ax1 = fig1.add_subplot(111)
[z, p, k] = dsp.zplane(b, a, plt_ax = ax1)
## ----- Impulsantwort -----
figure(2); grid('on')
[h, td] = dsp.impz(b, a, f_S)  #Impulsantwort / Koeffizienten
[ml, sl, bl] = stem(td, h)
plt.setp(ml, 'markerfacecolor', 'r', 'markersize', 8)
title(r'Impulsantwort h[n]')
## ----- Linear frequency plot -----
figure(3); grid('on')
plot(f, abs(H))
title(r'Betragsfrequenzgang')
## Log. Frequenzgang mit Spezifikationen
figure(5)
subplot (211)
plot(f,20 * log10(abs(H)), 'r'); plt.grid('on')
plot([0, f_DB],[-A_DB, -A_DB],'b--') # untere Spec-Grenze
plot([f_DB, f_DB], [ -A_DB, -A_DB-10], 'b--') #@ F_DB
if len(a) == 1:
    plot([0, f_DB],[A_DB, A_DB], 'b--') # obere Spec-Grenze
    plt.axis([0, f_DB * 1.1, -A_DB*1.1, A_DB * 1.1])
else:
    plot([0, f_DB], [0, 0], 'b--') # obere Spec-Grenze
    plt.axis([0, f_DB * 1.1, -A_DB * 1.1, A_DB * 0.1])
title(r'Betragsfrequenzgang in dB')
#
subplot(212)
plot(f,20 * log10(abs(H)), 'r'); plt.grid('on')
plot([0,  f_DB],[-A_DB, -A_DB],'b--') # untere Grenze DB
if len(a) == 1:
    plot([0,  f_DB], [A_DB, A_DB],'b--') # obere Grenze DB
else:
    plot([0, f_DB], [0, 0], 'b--') # obere Grenze DB
plot([f_SB, f_S/2.], [-A_SB, -A_SB], 'b--') # obere Grenze SB
plot([f_DB, f_DB], [-A_DB, -A_DB-10], 'b--') # @ F_DB
plot([f_SB, f_SB],[1, -A_SB],'b--') # @ F_SB
plt.tight_layout() # pad=1.2, h_pad=None, w_pad=None
#=========================================
## Phasengang
figure(6); grid('on')
plot(f,np.unwrap(np.angle(H))/pi)
# Ohne unwrap wird Phase auf +/- pi umgebrochen
title(r'Phasengang (normiert auf Vielfache von $\pi$)')
## Groupdelay
plt.figure(7)
[tau_g, w] = dsp.grpdelay(b, a, Fs = f_S)
plot(w, tau_g); plt.grid('on')
plt.ylim(max(min(tau_g)-0.5,0), (max(tau_g) + 0.5))
title(r'Group Delay $ \tau_g$') # (r: raw string)

plt.show()