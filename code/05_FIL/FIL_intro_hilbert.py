#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#===========================================================================
# ueb_FIL_intro_py.py
#
# Demonstrate filter design methods and specifications for 
# differentiators and hilbert filter 
#
#
# 
#
#
# (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import dsp_fpga_lib as dsp
#------------------------------------------------------------------------
# Ende der gemeinsamen Import-Anweisungen       
f_S = 200 # Samplingfrequenz 
f_DB = 40 # Grenzfreq. Durchlassband (DB)
f_SB = 50 # Grenzfrequenz Sperrband (SB)
F_DB = f_DB/(f_S/2) # Frequenzen bezogen
F_SB = f_SB/(f_S/2) # auf HALBE Abtastfreq.
#
A_DB = 0.1 # max. Ripple im DB in dB
A_DB_lin = (10**(A_DB/20.0)-1) / \
  (10**(A_DB/20.0)+1)*2 # und linear
A_SB = 40 # min. Dämpfung im SB in dB
A_SB_lin = 10**(-A_SB/20.0) # und linear
#
L = 31 # Manuelle Vorgabe der Tap-Zahl
############## FIR-Filterentwurf ##########
a = [1] # Nennerpolynom = 1 bei FIR-Filtern
#=== Windowed FIR / Least Square =========

#=== Differentiator: needs to have zero at f = 0 
# -> antisymmetric, even numtaps -> type IV
#Frequency Sampling =============
#b = sig.firwin2(L, [0, 1], [0, 1], antisymmetric = True)
#=== REMEZ / Parks-McClellan / Equiripple
#b = sig.remez(L, [0, 0.5], [1], type = 'differentiator')

#=======================================================1
# Hilbert-Transformer: zero at f = 0  and f = fS/2
# -> antisymmetric, odd numtaps
#b = sig.firwin2(L, [0,0.01, 0.5, 0.99, 1], [0,1, 1, 1,0], antisymmetric = True)
b = sig.remez(L, [0, 0.1, 0.11, 0.4, 0.41, 0.5], [0,1,0], [0.1,10,0.1],type = 'hilbert')

b = b / sum(abs(b))
print (b)
[w, H] = sig.freqz(b, a, 1024) 
# Translate w to physical frequencies:                   
f = w / (2 * pi) * f_S          
############## Plot the Results #########
## Pol/Nullstellenplan
figure(1)
[z, p, k] = dsp.zplane(b, a)
## ----- Impulsantwort -----
figure(2)
[h, td] = dsp.impz(b, a, f_S)  #Impulsantwort / Koeffizienten
[ml, sl, bl] = stem(td, h) 
plt.setp(ml, 'markerfacecolor', 'r', 'markersize', 8)
title(r'Impulsantwort h[n]')
## ----- Linear frequency plot -----
figure(3)
plot(f, abs(H))
title(r'Betragsfrequenzgang')       
## Log. Frequenzgang mit Spezifikationen
figure(4)
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
figure(5)
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