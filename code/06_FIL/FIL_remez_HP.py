# -*- coding: utf-8 -*-
"""
=== FIL_remez_HP.py ===================================================

# Minimum working example for remez (equiripple) filter designs:
# - low pass (LP) design works as expected, yielding ~ 0.1 dB pass band ripple
#        and 0 dB stop band attenuation 
# - high pass filter design is very strange, yielding ~ 15 db pass band ripple 
#      and 6 dB stop band attenuation?!
#
# Solution: HP needs odd order !!
===========================================================================
"""

from __future__ import division, print_function
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

F_PB = 0.1; F_SB = 0.15  # corner frequencies pass and stopo band
W_PB = 1;   W_SB = 1     # weight factors for pass and stop band 
N = 40 # filter order

# find coefficients for Lowpass and Highpass
b_LP = sig.remez(N, [0, F_PB, F_SB, 0.5], [1, 0], [W_PB, W_SB], Hz = 1) 
b_HP = sig.remez(N, [0, F_PB, F_SB, 0.5], [0, 1], [W_PB, W_SB], Hz = 1) 
# Calculate H(w), w = 0 ... pi, 1024 Pts.
[w, H_LP] = sig.freqz(b_LP, worN = 1024)
[w, H_HP] = sig.freqz(b_HP, worN = 1024)
# Translate w to normalized frequencies F = 0 ... 0.5:                   
F = w / (2 * np.pi)   
plt.figure(1)
plt.plot(F, 20 * np.log10(abs(H_LP)), label = 'LP')
plt.plot(F, 20 * np.log10(abs(H_HP)), label = 'HP')
plt.legend(loc = 'best')
plt.title(r'Magnitude transfer function in dB')
plt.show()