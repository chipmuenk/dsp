# -*- coding: utf-8 -*-
"""
=== SDM_Basic.py ================================================================

Erzeuge einen Sigma-Delta modulierten Bitstream

TODO: funktioniert noch nicht ...
 

 (c) 2014-Feb-04 Christian MÃŒnker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

t = linspace(0,4*pi, 1000)
x = sig.waveforms.square(t * 40)
y = sin(t * 2 * pi)
#y = np.sign(y)
b = [1,3]; a = [1,1]
my_t = arange(len(t))
z = arange(len(t))
for i in t:
    my_t[i], z[i], xout = sig.lsim2((b,a),y , i)

    q = z > 0
plot(my_t, z, t, y, t, x, t, q)
plt.show()