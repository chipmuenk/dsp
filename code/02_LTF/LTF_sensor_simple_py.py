# -*- coding: utf-8 -*-
"""
=== LTF_sensor_simple_py.py ==============================================
 
 Kapitel "LTF", "Abgetastete Signale im Frequenzbereich" (LTF)
 Abgetastetes und gefiltertes "Sensorsignal" im Frequenzbereich

 (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
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
sys.path.append('..')
import dsp_fpga_lib as dsp
#------------------------------------------------------------------------
# ... Ende der Import-Anweisungen
Ts = 1/200.0     # sampling period
f1 = 50.0      # signal frequency
phi0  = 0        # signal initial phase
Tmax = 6.0/f1  # time for 6 signal periods
N_Ts = Tmax / Ts # No. of samples in Tmax
# -- Sampled input signal and filter coeffs.
n = arange(0,round(N_Ts)) # sample n
xn = 1.5 + 0.5*cos(2.0*pi*f1*n*Ts + phi0)
b = np.ones(4)/4; a = 1 # MA-filter, N = 5
#b = np.convolve([1,1,1],[1,1,1]); a = 1 
#b = [1, 0]; a = [1, -0.9] # lossy integr.
#
# ----- P/Z-Plot -----
figure(1)
title('Pole/Zero-Plan')
dsp.zplane(b,a)
# ----- frequency response -----
figure(2)
[W, H] = sig.freqz(b, a, whole=0);
f = W  / (Ts * 2 * pi)
(w,Asig) = sig.freqz(b,a, f1*Ts*2*pi)
H_mx = np.max(abs(H))
H = H / H_mx; Asig = abs(Asig)/H_mx
#
subplot(311)
plot(f,abs(H))
ylabel(r'$|H(e^{j \Omega})| \rightarrow$')
title('Frequency Response')
plt.annotate('Attenuation @ %.1f Hz \n \
 = %1.3f (%3.1f dB)'%(f1,Asig,20*log10(Asig)),\
(f1, Asig),(0.5,0.5),textcoords='axes fraction',\
arrowprops=dict(arrowstyle="->"))
subplot(312)
plot(f, angle(H)/pi)
ylabel(r'$\angle H(e^{j\Omega}) / \pi$ ->')
subplot(313)
tau, w = dsp.grpdelay(b,a, nfft = 2048, Fs = 200, whole=0)
plot(w, tau)
xlabel(r'$f$ / Hz $\rightarrow$')
ylabel(r'$\tau_g(e^{j \Omega})/T_S\rightarrow$')
plt.show()       # draw and show the plots