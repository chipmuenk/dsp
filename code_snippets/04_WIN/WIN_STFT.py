# -*- coding: utf-8 -*-
"""
=== WIN_STFT.py ===========================================================

 Demonstriere Short-Term Fourier Transform mit verschiedenen Signalen
 
 STFT wird dargestellt mit Spectrogram und Wasserfall-Diagramm

 Testen Sie den Einfluss verschiedener Fensterlängen und Fenstertypen

 (c) 2016 Christian MÃŒnker - Files zur Vorlesung "DSV auf FPGAs"
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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

plt.close('all')

fs = 100e3 # Abtastfrequenz
NFFT = 1000 # FFT Punkte

fsig = 10000 # Signalfrequenz
Asig = 0.01 # Signalamplitude
fsig2 = 8376
Asig2 = 1
fmod = 0.5  # Modulationsfrequenz
Amod = 5000 # Modulation gain in Hz / V
tmax = 5 # Simulationzeit in s

dbmin = -100; dbmax = 0 # Limits fÃŒr log. Darstellung

win = sig.windows.kaiser(NFFT,12, sym = False) # needs NFFT and shape parameter beta 
#win = sig.windows.hann(NFFT, sym = False)
#win = sig.windows.blackman(NFFT, sym = False)
win = sig.windows.boxcar(NFFT) # rectangular window

# Calculate Equivalent Noise Bandwidth + Coherent Gain
ENBW = len(win)*np.sum(win**2)/ np.sum(abs(win))**2
CGain = np.sum(win)/len(win)

n = arange(NFFT/2 + 1)
t_label = r'$t$ in s $\rightarrow$'
f_label = r'$f$ in Hz $\rightarrow$'
H_label = r'$|H(e^{j \Omega})|$ in dB $\rightarrow$'


t = arange(0, tmax, 1/fs)
x = Asig*sig.chirp(t, 100, tmax, 1e5) # Chirp-Impuls
#x = Asig*(sin(t* 2* pi * fsig) + 1) # Reiner Sinus
x = Asig*sin(t* 2* pi * fsig + Amod *sin(t* 2* pi * fmod)) # FM-Signal
#x = Asig *sin(t* 2* pi * (fsig + Amod * np.sign(sin(t * 2 * pi * fmod)))) # FSK-Signal
x += Asig2 * sin(t* 2* pi * fsig2) # ZusÃ€tzlicher Sinuston
#x = A*np.sign(x) # Rechteckfunktion

#figure(1)
#plot(t,x)
######################################################################
# Spectrogram
######################################################################
figure(2)
# scale factor for correct *signal power* of spectrogram:
scale = NFFT * CGain

Pxx, freqs, bins, im = plt.specgram(x / scale, NFFT=NFFT, Fs=fs, 
                            noverlap=NFFT/2, mode = 'magnitude', window = win, 
                            scale = 'dB', vmin = dbmin, vmax = dbmax)
# freqs: DFT frequencies, bins: time steps                         

xlabel(t_label)
ylabel(f_label)
xlim([0,tmax])
ylim([0,fs/2])
plt.colorbar(label = H_label)
plt.tight_layout()

#----------------------------------------------------
figure(3)
time_slot = int(len(bins)/2)
plot(freqs, 20*log10(Pxx[:,time_slot]))

xlabel(f_label)
ylabel(H_label)
title(r'$|H(e^{j 2 \pi f / f_S},\, t)|$ bei $t=%0.1f$ s' %(bins[time_slot]))
ylim([dbmin, dbmax])
xlim([0,fs/2])
grid('on')
plt.tight_layout()

######################################################################
# Waterfall Diagram
######################################################################

fig = plt.figure(4)
ax = fig.gca(projection='3d')

xs = freqs # frequency axis
zs = arange(0,len(bins),5)  # time axis index
verts = []
mycolors = []

for z in zs:
    ys = np.maximum(20*log10(Pxx[:,z]),dbmin)
#    ys = np.random.rand(len(xs))
    ys[0], ys[-1] = dbmin, dbmin # set lower polygon points
    verts.append(list(zip(xs, ys)))
    mycolors.append((z/len(bins),0.3,0.4)) # r,g,b

poly = PolyCollection(verts, facecolors = mycolors)
                                      
poly.set_alpha(0.7) # set transparency
ax.add_collection3d(poly, zs = zs/len(bins)*tmax, zdir='y')

ax.set_xlabel(f_label)
ax.set_xlim3d(0, max(xs)) # frequency
ax.set_ylabel(t_label) 
ax.set_ylim3d(0, tmax) # time
ax.set_zlabel(H_label)
ax.set_zlim3d(dbmin, dbmax)
plt.tight_layout()

plt.show()