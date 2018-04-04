# -*- coding: utf-8 -*-
"""
 === LTF_sensor_py.py =================================================

 Kapitel "LTF", "Abgetastete Signale im Frequenzbereich" (LTF)
 Abgetastetes und gefiltertes "Sensorsignal" im Frequenzbereich

 (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
=======================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import sys
sys.path.append("..")
import dsp_fpga_lib as dsp

Ts = 1/200.0     # sampling period
f1 = 50.0      # signal frequency
phi0  = 0        # signal initial phase
Tmax = 6.0/f1  # time for 6 signal periods
N_Ts = Tmax / Ts # No. of samples in Tmax
# -- Sampled input signal and filter coeffs.
n = arange(0,round(N_Ts)) # sample n
xn = 1.5 + 0.5*cos(2.0*pi*f1*n*Ts + phi0)
b = np.ones(3)/3; a = 1 # MA-filter, N = 5
b = np.convolve([1,1,1],[1,1,1]); a = 1
b = [1, 0]; a = [1, -0.9] # lossy integr.
b = [1,2,1]; a = [1]
#p = [1 + 1j, 1 + 1j, 1]
#b = np.poly(p); a = 1
#
# ----- P/Z-Plot -----
figure(1); grid(True)
title('Pole/Zero-Plan')
xlabel(r'real $\rightarrow$')
ylabel(r'imag $\rightarrow$')
dsp.zplane(b,a)
# ----- frequency response -----
figure(2); grid(True)
[W, H] = sig.freqz(b, a, whole=0);
f = W  / (Ts * 2 * pi)
(w,Asig) = sig.freqz(b,a, f1*Ts*2*pi)
H_mx = np.max(abs(H))
H = H / H_mx; Asig = abs(Asig)/H_mx
#
subplot(311); grid(True)
plot(f,abs(H))
ylabel(r'$|H(e^{j \Omega})| \rightarrow$')
title('Frequency Response')
plt.annotate('Attenuation @ %.1f Hz \n \
 = %1.3f (%3.1f dB)'%(f1,Asig,20*log10(Asig)),\
(f1, Asig),(0.5,0.5),textcoords='axes fraction',\
arrowprops=dict(arrowstyle="->"))
subplot(312)
ylabel(r'$\angle H(e^{j\Omega}) / \pi \rightarrow $')
plot(f, angle(H)/pi)
subplot(313)
tau, w = dsp.grpdelay(b,a, nfft = 2048, Fs = 200, whole=0)
plot(w, tau)
ymin, ymax = ylim(); dy = max(ymax - ymin, 1) * 0.05
ylim(ymin - dy, ymax + dy) # set reasonable range
xlabel(r'$f$ / Hz $\rightarrow$')
ylabel(r'$\tau_g(e^{j \Omega}) / T_S \rightarrow$')
#plt.tight_layout()

#-------------------------------------------------------
fig3 = figure(3)
plt.suptitle('Filtereigenschaften', fontsize=18)
#
h, t = dsp.impz(b,a) # impulse response
ax31 = fig3.add_subplot(311)
ax31.stem(range(len(h)),h)
ax31.grid('on')
ax31.set_ylabel(r'Impulsantwort' '\n' r'$h[n]\, \rightarrow$')
ax31.yaxis.set_label_coords(-0.1, 0.5)
#
yn = np.convolve(xn,h) # Faltung (Convolution) von x und h
ax32 = fig3.add_subplot(312)
ax32.grid('on')
ax32.stem(range(len(yn)),yn)
ax32.set_xlabel(r'$n \, \rightarrow$')
ax32.set_ylabel(r'Filterausgang' '\n' r'$y[n]\, \rightarrow$')
ax32.yaxis.set_label_coords(-0.1, 0.5)
#
w,h = sig.freqz(h); f =  w / (2*pi*Ts)
ax33 = fig3.add_subplot(313)
ax33.plot(f, 20*log10(abs(h)), 'b')
ax33.set_ylabel(r'Übertragungsfunktion' '\n' r'$20 \log |H(f)|\, \rightarrow$')
ax33.set_xlabel(r'$f$ / Hz $\rightarrow$')
ax33.yaxis.set_label_coords(-0.1, 0.5)
ax33.grid('on')
fig3.subplots_adjust(left=0.18, bottom=None, right=0.95, top=0.92,
                wspace=None, hspace=0.5)
plt.show()       # draw and show the plots