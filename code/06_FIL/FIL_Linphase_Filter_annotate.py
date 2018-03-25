# -*- coding: utf-8 -*-
"""
=== FIL_Linphase_Filter_annotate.py =======================================
  
 Demonstriere Einfluss von einfachen und doppelten Nullstellen im SB und DB,
 mit Anmerkungen im Plot (annotate() und text() )
 
 (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""

from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

f_S = 1
bb = [-1/8, 0, 5/8, 1, 5/8, 0, -1/8]
aa = 1

z_0 = np.roots(bb) # Wurzeln des Polynoms, 

print(' i |  z_0,i          | (polar)')   
print('------------------------------------------------')
for i in range(len(z_0)):
    print('{0:2d} | {1:15.4f} |{2:6.3f} * e^(j {3:6.4f} pi) '\
    .format(i, z_0[i],abs(z_0[i]),np.angle(z_0[i])/pi))  


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

plt.text( 0.1,12, "ohne Nullstellen auf pos. reeller Achse\n (anderer Maßstab!)", 
            size=16, va="center", ha="left")
plt.annotate("einfache",
            xy=(0.36, 0.05), xycoords='data',
            xytext=(0.6, 0.5), textcoords='axes fraction',
            size=16, va="center", ha="center",
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.2"),
                            )
plt.annotate("doppelte\n Nullstelle",
            xy=(0.5, 0.05), xycoords='data',
            xytext=(0.85, 0.5), textcoords='axes fraction',
            size=16, va="center", ha="right",
            arrowprops=dict(arrowstyle="simple",
                            connectionstyle="arc3,rad=-0.2"),
                            )
# try 'figure fraction','axes points', 'offset points', ...             
plt.tight_layout(); plt.show()