#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# LTI_faltung_py.py ====================================================
# 
#
# Einfaches Code-Beispiel zum Kapitel "LTI-Systeme im Zeitbereich"
#
# Thema: Zeitdiskrete Faltung
#
#
#
# 
# (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals, absolute_import # v3line15

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, sqrt, exp, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

#import dsp_fpga_lib as dsp
#-------- ----------------------------------------------------------------
# ... Ende der gem. import-Anweisungen
h = [0.25, 0.5, 0.25] # Impulsantwort
x = [1, 1, 1, 1, 1] # Eingangssignal
y = np.convolve(x, h) 
n = arange(len(y)) # n = 0 ... len(y)-1
figure(1)
stem(n, y, 'r'); grid(True)
xlabel(r'$n \rightarrow$')
ylabel(r'$y[n] \rightarrow$')
title(r'Faltung $y[n] = x[n] \star \{1; 1; 1; 1; 1\}$')
plt.show()