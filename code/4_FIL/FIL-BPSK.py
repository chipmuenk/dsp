#!/usr/bin/python
# -*- coding: utf-8 -*
# BPSK digital modulation example
# by Ivo Maljevic
#=========================================================================
# Python Musterlösung zu "Periodizität abgetasteter Signale"
# Abtastung und Filterung eines Sensorsignals
# 
# (c) 2013-Jul-17 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#=========================================================================
from __future__ import division, print_function # v2

import numpy as np
import numpy.random as rnd
from numpy import sin, cos, tan, angle, pi, array, arange, log10, zeros, \
  linspace, ones, sqrt
from numpy.fft import fft, fftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, plot, stem, grid, xlabel, ylabel, \
    subplot, title, clf

import dsp_fpga_lib as dsp
#------------------------------------------------------------------------
from scipy.special import erfc
from numpy import sign

SNR_dB      = 1 # dB
SNR         = 10 ** (SNR_dB/10.0)  # linear SNR
OSR         = 8 # Samples per symbol 
                # -> F_max = 1 / (2 * OSR)
OSR_2       = 4 # OSR/2, middle of symbol

R_DB = 0.5  # ripple of passband
R_SB = 60   # ripple of stopband
del_DB = (10**(R_DB/20)-1) / (10**(R_DB/20)+1) # lin. Ripple
del_SB = 10**(-R_SB/20.0) # lin. ripple
F_DB = 0.08  # corner of passband
F_SB = 0.12 # corner of stopband
#L = 32 # manual filter order
N_plt       = 1000 # Number of plotted samples
N_show      = 400  # Number of displayed samples

Pe = 0.5 * erfc(sqrt(SNR)) # theoretical error probability
                           # for BPSK
VEC_SIZE = max(np.ceil(100/Pe),10000) # minimum vector length f(Pe)
                        # for estimating the SNR
#====== Filter ================================
#
b = ones(4); a = 1 # MA filter
# Parks-McClellan / Remez (= Linphase FIR) - Filter:
L, bands, amps, w = dsp.remezord([F_DB, F_SB],[1, 0], [del_DB, del_SB])
#b = sig.remez(L, [0, F_DB, F_SB, 0.5], [1, 0], [1, 1], Hz = 1)
b = sig.remez(L, bands, amps, w)

# IIR-Filter
[b, a] = sig.iirdesign(F_DB*2, F_SB*2, R_DB, R_SB, ftype='ellip')
#[b, a] = sig.iirdesign(F_DB*2, F_SB*2, R_DB, R_SB, ftype='cheby2')
#[b, a] = sig.iirdesign(F_DB*2, F_SB*2, R_DB, R_SB, ftype='butter')
#b = [0.2]; a = [1, -0.8]
#
#================================================
w, H = sig.freqz(b,a,2048)
Hmax = max(abs(H)); H = H / Hmax
tau, w = dsp.grpdelay(b,a, 2048)
f = w / (2* pi)
delay = tau[np.floor(2048 / (2*OSR))] # delay at F = 0.5 / (2 * OSR)
dN = np.floor(delay)
print('Delay @ F= %2.3e = %.1f T_S' %((0.5 / (2 * OSR)), delay))
#====== Symbol & Noise Generation ===============
# random binary (+/-1) symbol sequence
s = 2 * rnd.randint(0,high=2,size=VEC_SIZE)-1
s = s.repeat(OSR) # repeat each symbol OSR times 
# linear power of the noise; average signal power = 1:
No = 1.0/SNR
# random signal with normal distribution and power No
n = sqrt(No/2)*rnd.randn(VEC_SIZE*OSR) # 
# signal + noise
x = s + n
#x = s
xf = sig.lfilter(b,a,x) / Hmax
# decode received signal + noise in the middle of symbol
y = sign(x[OSR_2::OSR]) # use every OSR-th sample
yf = sign(xf[OSR_2+dN::OSR]) # 
# find erroneous symbols in unfiltered signal
err = [y != s[OSR_2::OSR]]
err_pos = (np.array(np.where(y != s[OSR_2::OSR])) * OSR) + OSR_2
error_sum = np.sum(err)
BER = error_sum / VEC_SIZE
# and in filtered signal
err_pos_f = (np.array(np.where(yf != s[OSR_2:-dN:OSR])) * OSR) + OSR_2
err_f = [yf != s[OSR_2:-dN:OSR]]
error_sum_f = np.sum(err_f)
BER_f = error_sum_f / VEC_SIZE
print('SNR (dB) = %4.2f => Theor. BER = %8.3e\n\
=> Sim. Länge = %d  erforderlich' %(SNR_dB, Pe, VEC_SIZE))
print('BPSK-Dekodierung mit gestörtem Signal')
print('Fehler = %6d => BER = %8.3e' %(error_sum, BER))
print('BPSK-Dekodierung mit TP-Filterung')      
print('Fehler = %6d => BER = %8.3e' % (error_sum_f, BER_f))
n = arange(N_plt)
figure(1)
subplot(211)
title(u'BPSK Signalübertragung mit AWGN')
plt.ylim(-2.5,2.5);  plt.xlim(0, N_show)
plot(n, s[0:N_plt], 'r', label = 'sent') 
plot(n, x[0:N_plt], 'b', lw = 1, label = 'received')
stem(n[OSR_2::OSR],y[0:np.ceil(N_plt/OSR)],
         'r', markerfmt = 'bo', label = 'recovered')
if np.shape(err_pos)[1] > 0:
    plt.plot(err_pos, ones(len(err_pos))*(-1.8), '^', markersize = 10)
plt.legend()
subplot(212)
title(u'BPSK Signalübertragung mit AWGN (TP-gefiltert)')
plt.ylim(-2.5,2.5); plt.xlim(0, N_show)
plot(n,s[0:N_plt], 'r', n, xf[dN:N_plt+dN], 'b')
stem(n[OSR_2::OSR],yf[0:np.ceil(N_plt/OSR)],'r', markerfmt = 'bo')
if (np.shape(err_pos_f)[1] > 0) and (np.shape(err_pos_f)[1] < 1000) :
    plot(err_pos_f, ones(len(err_pos_f))*(-1.8), '^', markersize = 10)
xlabel(r'$n \; \rightarrow$')
plt.tight_layout()
#
plt.figure(2)
S = 20*log10(abs(fft(s,4096))); S = S - max(S)
X = 20*log10(abs(fft(x,4096))); X = X - max(X)
Xf = 20*log10(abs(fft(xf,4096))); Xf = Xf - max(Xf)
#
subplot(311); plt.grid('on')
plot(f, S[0:2048],label = r'$|S(F)|$')
plot(f, 20*log10(abs(H)), lw=2, label = r'$|H_{TP}(F)|$')
plt.legend()
#ylabel(r'$|S(F)| \rightarrow $')
#ylabel(r'$|H_TP(F)| \rightarrow$')
plt.ylim(-80, 0)
subplot(312); plt.grid('on')
plot(f, tau)
ylabel(r'$\tau_g  \rightarrow$')
plt.ylim(max(min(tau)-0.5,0), (max(tau) + 0.5))

#==============================================================================
# subplot(513); plt.grid('on')
# plot(f, S[0:2048])
# ylabel(r'$|S(F)| \rightarrow $')
#==============================================================================
subplot(313); plt.grid('on')
plot(f, X[0:2048])
ylabel(r'$|S_N(F)| \rightarrow$')

plot(f, Xf[0:2048])
ylabel(r'$|S_N \cdot H_{TP}(F)| \rightarrow$')
xlabel(r'$F = \Omega / 2\pi \; \rightarrow$ ')
plt.tight_layout(pad=0.2, h_pad= 0)
plt.show()

