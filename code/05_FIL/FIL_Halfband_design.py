# -*- coding: utf-8 -*-
"""
=== FIL_Halfband_design.py ================================================
  
 Generate coefficients for a halfband filter.  A halfband filter
 is a filter where the cutoff frequency is Fs/4 and every
 other coeffecient is zero except for the center tap.
 Due to numeric inaccuracies, coefficients are not exactly zero and are set 
 to 0 when below a certain threshold.
 
 (c) 2016 Christian Münker & Florian Thevissen - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import pi, arange, log10
import scipy.signal as sig

import matplotlib.pyplot as plt

N = 16 # Filter order - must be even for Halfband filter!

# --- Filter Design with Parks-McClellan (Remez) method -----------------------
# Filter symmetric around 0.25 (for remez .5 is pi or F = Fs/2 -)
bands = np.array([0., .22, .28, .5]) # define HALFBAND passband / stopband
#bands = np.array([0., .1, .15, .5]) # define passband / stopband
b_rem = sig.remez(N+1, bands, [1, 0], [1, 1], Hz = 1.0)
b_rem[abs(b_rem) <= 1e-4] = 0 # set very small coefficients = 0 (rounding errors)
# normalize for [H(f=0) + |H(f = fS/2)|] = 1
b_rem = b_rem / (sum(b_rem) + abs(np.polyval(b_rem, -1))) 
(w, H_rem) = sig.freqz(b_rem)


# --- Filter Design with Windowed method --------------------------------------
# set cutoff frequency = f_Nyq/2 = f_S/4
b_win = sig.firwin(N+1, 0.5, nyq = 1.0) # HALFBAND filter
#b_win = sig.firwin(N+1, 0.25, nyq = 1.0)
b_win[abs(b_win) <= 1e-5] = 0
# normalize for [H(f=0) + |H(f = fS/2)|] = 1
b_win = b_win / (sum(b_win) + abs(np.polyval(b_win, -1))) 
(w, H_win) = sig.freqz(b_win)

# --- Filter Design with Least Square method ----------------------------------
#b_ls = sig.firwin(N+1, 0.5, window = 'boxcar')
b_ls = sig.firwin(N+1, 0.5, window = 'boxcar') #Halfband
#b_ls = sig.firwin(N+1, 0.25, width=0.01, window = 'boxcar')

b_ls[abs(b_ls) <= 1e-4] = 0
# normalize for [H(f=0) + |H(f = fS/2)|] = 1
b_ls = b_ls / (sum(b_ls) + abs(np.polyval(b_ls, -1))) 
(w, H_ls) = sig.freqz(b_ls)

F = w / (2 * pi)

b_rem_0 = []
b_win_0 = []
b_ls_0 = []

# Print coefficients for comparison and verification & find zero coefficients
print('          remez      firwin     Least-Square')
print('--------------------------------------------')
for i in range(N+1):
    print(' tap %2d  %+3.6F  %+3.6f  %+3.6f' %(i, b_rem[i], b_win[i], b_ls[i]))
    if b_rem[i] == 0: b_rem_0.append(i) 
    if b_win[i] == 0: b_win_0.append(i)
    if b_ls[i] == 0: b_ls_0.append(i)

#--------------------- Plot impulse responses of filters ----------------------
fig1 = plt.figure(1)
ax11 = fig1.add_subplot(311)
ax11.stem(arange(len(b_ls)), b_ls, label = 'LeastSquare') # impulse response
ax11.scatter(b_ls_0, np.zeros(len(b_ls_0)), s = 300, facecolors=(0.5,1,0.5,0.7),
             edgecolors='darkgreen') # mark zero coefficients
ax11.grid(True)
ax11.legend()
ax11.set_xlim(0, N)
ax11.set_title('Impulse Response')

ax12 = fig1.add_subplot(312)
ax12.stem(arange(len(b_win)), b_win, label = 'Windowed LS')
ax12.scatter(b_win_0, np.zeros(len(b_win_0)), s = 300,
             facecolors=(0.5,1,0.5,0.7), edgecolors='darkgreen')
             
ax12.plot(np.arange(N+1), np.max(b_win) * sig.windows.hamming(N+1), '--')
ax12.grid(True)
ax12.legend()
ax12.set_xlim(0, N)

ax13 = fig1.add_subplot(313)
ax13.stem(arange(len(b_rem)), b_rem, label = 'Remez')
ax13.scatter(b_rem_0, np.zeros(len(b_rem_0)), s = 300,
             facecolors=(0.5,1,0.5,0.7), edgecolors='darkgreen')
ax13.grid(True)
ax13.legend()
ax13.set_xlim(0, N)
plt.tight_layout()

#--------------------- Plot linear amplitude response -------------------------
fig2 = plt.figure(2)
ax21 = fig2.add_subplot(111)
ax21.plot(F, abs(H_ls), label = 'LeastSquare')
ax21.plot(F, abs(H_win), label = 'Windowed LS')
ax21.plot(F, abs(H_rem), label = 'Remez')
ax21.legend()
ax21.axvspan(bands[1], bands[2], facecolor='0.8')# transition band in grey
ax21.plot(1/4, 1/2, 'o', markerfacecolor=(0.5,1,0.5,0.7), markersize = 15,
          markeredgecolor = 'darkgreen')# mark the symmetry point
ax21.axvline(1/4, color='g', linestyle='--')
ax21.axis([0,1/2,0,1.2])
ax21.grid('on')
ax21.set_ylabel('Magnitude')
ax21.set_xlabel('Normalized Frequency')
ax21.set_title('Linear Filter Frequency Response')

#--------------------- Plot log. amplitude response ---------------------------
fig3 = plt.figure(3)
ax31 = fig3.add_subplot(111)
ax31.plot(F, 20*log10(abs(H_ls)), label = 'LeastSquare')
ax31.plot(F, 20*log10(abs(H_win)), label = 'Windowed LS')
ax31.plot(F, 20*log10(abs(H_rem)), label = 'Remez')
ax31.legend()
ax31.axvspan(bands[1], bands[2], facecolor='0.8') # transition band in grey
ax31.plot(1/4, -20*log10(2.), 'o', markerfacecolor=(0.5,1,0.5,0.7), 
          markersize = 15, markeredgecolor = 'k')
ax31.axvline(1/4, color='g', linestyle='--')
ax31.axis([0,1/2,-64,3])
ax31.grid('on')
ax31.set_ylabel('Magnitude (dB)')
ax31.set_xlabel('Normalized Frequency')
ax31.set_title('Log. Filter Frequency Response')

plt.show()

