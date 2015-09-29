#!/usr/bin/env python
# -*- coding: utf-8 -*
#===========================================================================
# FIL-Linphase_Filter_annotate_py.py
#
# Demonstriere annotate() und text()
#
# 
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

import dsp_fpga_lib as dsp
#------------------------------------------------------------------ v3line30
# Ende der gemeinsamen Import-Anweisungen
f_S = 1
bb = [-1/8, 0, 5/8, 1, 5/8, 0, -1/8]
aa = 1

z_0 = np.roots(bb) # Wurzeln des Polynoms, 

print(' i |  z_0,i          | (polar)')   
print('------------------------------------------------')
for i in range(len(z_0)):
    print('{0:2d} | {1:15.4f} |{2:6.3f} * e^(j {3:6.4f} pi) '\
    .format(i, z_0[i],abs(z_0[i]),np.angle(z_0[i])/pi))  
#    print('%2d | %15.4f |%6.3f * e^(j {%6.4f} pi)'\
#    %(i, z_0[i],abs(z_0[i]),np.angle(z_0[i])/pi))

for i in range(len(z_0)):
    if z_0[i].real > 0: # Entferne Nullstellen im Durchlassband
        z_0[i] = 0
        
bb_neu = np.poly(z_0) # "Ausmultiplizieren" der Koeffizienten

[w, H_org] = sig.freqz(bb, aa, 1024)
[w, H_neu] = sig.freqz(bb_neu, aa, 1024)
f = w / (2 * pi) * f_S

figure(1)
subplot(211)
plot(f, abs(H_org)); grid(True)
plt.annotate("einfache",
            xy=(0.36, 0.05), xycoords='data',
            xytext=(0.32, 1.0), textcoords='data',
            size=16, va="center", ha="center",
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.2"),
                            )
plt.annotate("doppelte\n Nullstelle",
            xy=(0.5, 0.05), xycoords='data',
            xytext=(0.45, 1.0), textcoords='data',
            size=16, va="center", ha="center",
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.2"),
                            )
title(r'Betragsfrequenzgang $|H(\mathrm{e}^{\mathrm{j} 2 \pi F})|$')
subplot(212)
plot(f,abs(H_neu)); grid(True)
xlabel(r'$F \; \rightarrow$')

plt.text( 0.1,12, "ohne Nullstellen auf pos. reeller Achse (DB)", 
            size=16, va="center", ha="left")
            
plt.tight_layout(); plt.show()