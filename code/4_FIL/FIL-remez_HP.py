#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# Minimum working example for remez (equiripple) filter designs:
# - working low pass (LP), yielding ~ 0.1 dB pass band ripple and 
#       40 dB stop band attenuation 
# - strange high pass filter design, yielding ~ 15 db pass band ripple 
#      and 6 dB stop band attenuation
#
# Solution: HP needs odd order !!
# 
from __future__ import division, print_function
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt
F_PB = 0.1  # corner frequency pass band
F_SB = 0.15 # corner frequency stop band
W_PB = 1    # weight factor pass band 
W_SB = 1    # weight factor stop band
L = 40 # filter order
#b = sig.remez(L, [0, F_PB, F_SB, 0.5], [1, 0], [W_PB, W_SB], Hz = 1) # LP
b = sig.remez(L, [0, F_PB, F_SB, 0.5], [0, 1], [W_PB, W_SB], Hz = 1) # HP
## Calculate H(w), w = 0 ... pi, 1024 Pts.
[w, H] = sig.freqz(b, worN = 1024)
# Translate w to normalized frequencies F = 0 ... 0.5:                   
F = w / (2 * np.pi)   
plt.figure(1)
plt.plot(F, 20 * np.log10(abs(H)))
plt.title(r'Magnitude transfer function in dB')
plt.show()