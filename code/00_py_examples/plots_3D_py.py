#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# plots_3D_py.py ================================
#
#
# Example for Matplotlib 3D-Plotting, Gradient of |H(z)| is
# displayed as a colormap
# taken from http://stackoverflow.com/questions/6539944/
#     color-matplotlib-plot-surface-command-with-surface-gradient
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

import dsp_fpga_lib as dsp
#------------------------------------------------------------------------
# ... Ende der Import-Anweisungen# -*- coding: utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

fig = figure(1)
#ax = fig.gca(projection='3d')
ax = fig.add_subplot(111, projection='3d') # new syntax?
X = Y = arange(-10, 10, .25)
X, Y = np.meshgrid(X, Y)
R = sqrt(X**2 + Y**2)
Z = sin(R)
Gx, Gy = np.gradient(Z) # gradients with respect to x and y
G = (Gx**2+Gy**2)**.5  # gradient magnitude
N = G/G.max()  # normalize to 0...1 for better colormapping
surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1,
    facecolors=cm.jet(N),
    linewidth=0, antialiased=False, shade=False)
plt.show()