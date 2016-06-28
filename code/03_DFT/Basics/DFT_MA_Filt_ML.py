# -*- coding: utf-8 -*-
"""
=== DFT_MA_Filt_ML.py ======================================================

Python Musterlösung zu "DFT eines Moving Averaging-Filters"

Berechnung und Darstellung der DFT in Python

(c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

h = [1,1,1,1]
n = arange(len(h))
figure(1);
subplot(211)
stem(n,h)
ylabel(r"$h[n] \; \rightarrow$")
xlabel(r"$n \; \rightarrow$")
subplot(212)
H = abs(fft(h,256))
k = arange(len(H)/2)
stem(k,H[:len(k)])
xlabel(r"$k \; \rightarrow$")
ylabel(r"$|H[k]| \; \rightarrow$")
plt.tight_layout()
plt.show()
