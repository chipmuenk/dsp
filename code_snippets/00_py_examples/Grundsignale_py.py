# -*- coding: utf-8 -*-
"""
=== LTI_Grundsignale_py.py =====================================================

 Einfaches Code-Beispiel zum Kapitel "LTI-Systeme im Zeitbereich"

 Thema: Beispiele für Darstellung von einfachen Funktionen in Python
 
 (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange, 
                   linspace, array, zeros, ones)
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

from mpl_toolkits.mplot3d import Axes3D # für 'projection3d'

## Komplexe Exponentialschwingung y = exp( j \omega t)
fig1 = figure(1)
# stelle figure um auf 3D-Projektion
ax1 = fig1.gca(projection='3d') # gca = get current axes
t = arange(0, 2, 0.01)
y = exp(1j * 2 * pi * t)
ax1.plot(y.real, y.imag, t, linewidth=2)
title(r'Komplexe Exponentialschwingung $y = \exp( j \Omega t)$')
xlabel(r'$\Re \{\} $')
ylabel(r'$\Im \{\}$')
#=================================================================
## Rechteckimpuls rect(t/T_0)
figure(2)
T0 = 1
t = arange(-2, 2, 0.01) # start, stop, step
x = (abs(t) < 0.5 * T0)
plot(t,x)
xlabel(r'$t/T_0 \rightarrow$')
ylabel(r'$x(t) \rightarrow$')
title(r'rect(x) - Funktion') 
grid('on')
#axis([-2 2 -0.2 1.2]);
#==================================================================
## sin x / x  - Funktion (sinc - function)
figure(3)
t = arange(-8, 8, 0.01)
T0 = 1
f0 = 1/T0
x = sin(pi*f0*t) / (pi*f0*t) # Elementweise Division
plot(t, x)
grid ('on')
xlabel(r'$t/T_0 \rightarrow$')
ylabel(r'$x(t) \rightarrow$')
title('sin(x)/x - Funktion')
#=========================================================================
## Dirac-Puls (-Kamm)
figure(4)
t = arange(-3,4) # Das letzte Element ist nicht dabei!
x = ones(len(t)) # 1 - Vektor / Matrix der Größe 1 x 7 [ vgl. x=zeros(a,b) ];

stem(t, x) # "stem" = Stamm, Stengel
plt.axis([-3.6, 3.6, -.2, 1.2])
title('Periodische Diracfunktion')
xlabel(r'$t/T_0 \rightarrow$')
ylabel(r'$x(t) \rightarrow$')
# Formatierung:
plt.text(-3.4,0.5,'...',fontsize=16, color='b') # PÃŒnktchen links
plt.text(3.2,0.5,'...',fontsize=16, color='b') # PÃŒnktchen rechts
plt.text(0.15,1.0 ,'(1)',fontsize=16, color='b') # Dirac - Gewicht
#=========================================================================
## Periodische Wellenformen
figure(5)
t = arange(-20,20,0.1)
s1 = sig.waveforms.square(2*t/pi) # T = 10
s2 = sig.waveforms.gausspulse(t,fc = 0.5)
s3 = sig.waveforms.sawtooth(2*t/pi) # T = 10
plot(t,s1,t,s2,t,s3)
ylim([-1.2,1.2])
plt.tight_layout()
plt.show()
