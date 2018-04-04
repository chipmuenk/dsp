# -*- coding: utf-8 -*-
"""
== _common_imports_v3_py.py =============================================

 Header mit import Anweisungen für im Kurs benötigte Python-Module

 (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
# Rüste python 3.x features auch für python 2.x nach
from __future__ import division, print_function, unicode_literals

import numpy as np               # schnelle Array-Mathematik
import numpy.random as rnd       # Zufallszahlen/prozesse
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones) # Häufig benötigte Funktionen
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq # FFT-Untermodul

import scipy.signal as sig        # Signalverarbeitung
import scipy.interpolate as intp  # Interpolationsverfahren

import matplotlib.pyplot as plt   # Grafik-Bibliothek
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim) # Häufig benötigte Funktionen
# Lade Stylefiles (Fontgrößen, Linienstärken, Farben) ...
plt.style.use('seaborn-talk')             # ... aus Matplotlib stylelib Directories
plt.style.use('../presentation.mplstyle') # ... sucht stylefile im angegebenen Pfad
print(plt.style.available())     # Zeige verfügbare Matplotlib-Styles

import sys                       # Füge Parent Directory zum Suchpfad hinzu für
sys.path.append('..')            #    eigene Bibliotheken
import dsp_fpga_lib as dsp       # Bibliothek mit zusätzlichen DSP-Funktionen
import dsp_fpga_fix_lib as fx    # Bibliothek mit schneller Fixpoint-Arithmetik