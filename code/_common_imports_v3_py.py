#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#===========================================================================
# _common_imports_v3_py.py
#
# Einfaches Code-Beispiel zum Kapitel "xxx", Übungsaufgabe yyy
#
# Importiere Module zur 
# - Erzeugung von Zufallszahlen:    numpy.rnd
# - (Inverse) FFT:                  numpy.fft
# - Signalverarbeitung              scipy.signal
# - Interpolation                   scipy.interp
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
#------------------------------------------------------------------ v3line30
# Ende der gemeinsamen Import-Anweisungen