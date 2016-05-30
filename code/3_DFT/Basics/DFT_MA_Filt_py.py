# -*- coding: utf-8 -*-
#===========================================================================
# DFT_MA_Filt_py.py
#
# Python Musterlösung zu "DFT des MA-Filters"
#
# Berechnung und Darstellung der DFT in Python
#
#
#
#
# (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

#import dsp_fpga_lib as dsp
#------------------------------------------------------------------ v3line30
# ... Ende der Import-Anweisungen
h = [1,1,1,1]
n = arange(len(h))
figure(1);
subplot(211)
stem(n,h)
ylabel(r"$h[n] \; \rightarrow$")
xlabel(r"$n \; \rightarrow$")
subplot(212)
H = fft(h,256)
k = arange(len(H))
stem(k,abs(H))

xlabel(r"$k \; \rightarrow$")
ylabel(r"$|H[k]| \; \rightarrow$")
plt.tight_layout()
plt.show()
