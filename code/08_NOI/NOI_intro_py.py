# -*- coding: utf-8 -*-
"""
=== NOI_intro_py.py ==========================================================

Demonstration des Einflusses von Quantisierung im Frequenzbereich:

Erzeuge ein Sinussignal und quantisiere es auf eine vorgegebene Anzahl von
Vor- und Nachkommabits mit verschiedenen Quantisierungs- und Overflowmethoden.

Überlegen / überprüfen Sie jeweils:
Passt die Formel SQNR = (6.02 w + 1.76) dB ?
Wie kann man den angezeigten mittleren Rauschpegel umrechnen in die Rauschleistung?

(c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import sys
sys.path.append('..')
import dsp_fpga_fix_lib as fx

NOISE_LABEL = True # add second axis for noise power density
NFFT = 2000; f_a = 2; N_per = 7
T_mess = N_per / f_a; T_S = T_mess / NFFT
t = linspace(0, T_mess, NFFT, endpoint=False)
a = 1.1 * sin(2 * pi * f_a * t)
#
q_obj = {'QI':1, 'QF': 3, 'quant':'round', 'ovfl': 'wrap'} # versuchen Sie auch 'floor' / 'fix' und 'sat'
fx_a = fx.Fixed(q_obj)
aq = fx_a.fix(a)

print('Anzahl der Überläufe = ', fx_a.N_over)
#
figure(1)
title('Quantisiertes Sinussignal und Quantisierungsfehler')
plot(t, a, label = r'$a(t)$')
plt.step(t, aq, where = 'post', label = r'$a_Q(t)$')
plot(t, a-aq, label = r'$a(t) - a_Q(t)$')
plt.legend(fontsize = 14)
xlabel(r'$t \; \mathrm{/ \; s} \; \rightarrow$'); ylabel(r'$a \rightarrow$')
A_max = 2**q_obj['QI'] - 2**-q_obj['QF']
plt.axhline(y = A_max, linestyle = '--', color = 'k')
plt.axhline(y = -A_max, linestyle = '--', color = 'k')
#
Amin = -100 # Unteres Limit in dB für die Darstellung
A =  abs(2 / sqrt(2) * fft(a) / NFFT)[0:NFFT // 2 - 1]  # einseitiges Spektrum,
AQ = abs(2 / sqrt(2) * fft(aq) / NFFT)[0:NFFT // 2 - 1] #    Effektivwert !
A[0] = A[0] * sqrt(2)/2; AQ[0] = AQ[0] * sqrt(2)/2  # korrigiere DC-Wert zurück
f = fftfreq(NFFT, T_S)[0:NFFT//2 - 1]      # Frequenzen f. einseitiges Spektrum
#
fig2 = figure(2)
title('Spektrum von Eingangs- und quantisiertem Signal')
ax2 = fig2.add_subplot(111)
ax2.stem(f, 20 * log10(AQ), bottom = Amin, label = r'$A_Q(f)$')
ml, sl, bl = stem(f,  20 * log10(A), bottom = Amin, label = r'$A(f)$')
plt.setp(ml, 'markerfacecolor', 'r', 'markersize', 10 ,'alpha', 0.4) # Marker
plt.setp(sl, 'color','r', 'linewidth', 5, 'alpha', 0.4)    # Stemline
AQ[N_per] = abs(AQ[N_per] - A[N_per]) # Subtrahiere Signal
S = 10*log10(A[N_per]**2.)       # Signalleistung, gemessen
N_PSD = 10 * log10(np.average(AQ*AQ))   # mittlere Rauschleistungsdichte
NQ = N_PSD + 10*log10(NFFT/2)
ENOB = ((S-NQ) - 1.76)/6.02
# print(S, NQ, S - NQ)
plt.axhline(y = N_PSD) # plotte horiz. Linie mit mittlerer Rauschleist.dichte
ax2.set_xlabel(r'$f \; \mathrm{/ \; Hz} \; \rightarrow$')
ax2.set_ylabel(r'$A_{eff(,q)}(f), \;S_{(q)}(f)\; \mathbf{[dB]} \;\rightarrow$')
ax2.set_ylim(Amin,10)
plt.text(1/(4*T_S), Amin,
         r'$S = %.2f \, \mathrm{dB},\, N_Q = %.2f \, \mathrm{dB}$, '
         r'$SQNR = %.2f \, \mathrm{dB}$'%(S, NQ, S-NQ) + '\n' +
         r'$ENOB = %.2f \, \mathrm{bits},\, N_{FFT} = %d,\, f_S = %.2f\,'
         r'\mathrm{Hz}$'%(ENOB, NFFT, 1./T_S),
        ha='center', va='bottom', bbox=dict(facecolor='0.8', alpha=0.8))
     #
plt.legend(fontsize = 14)
if NOISE_LABEL:
    ax2b = ax2.twinx()
    ax2b.set_ylim(ax2.get_ylim()+ 10*log10(NFFT/2))
    ax2b.set_ylabel(r'$N_q(f)\;  \mathbf{[dBW / Hz]} \; \rightarrow$')
plt.show()