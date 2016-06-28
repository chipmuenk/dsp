# -*- coding: utf-8 -*-
"""
LTI_faltung_py.py ====================================================

Einfaches Code-Beispiel zum Kapitel "LTI-Systeme im Zeitbereich"

Thema: Zeitdiskrete Faltung

 (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
======================================================================
"""

from __future__ import division, print_function, unicode_literals, absolute_import

import numpy as np
from numpy import (pi, log10, sqrt, exp, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

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