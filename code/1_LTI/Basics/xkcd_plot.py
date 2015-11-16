# -*- coding: utf-8 -*-
"""
Beispiel für einen Plot im XKCD-Modus ("händischer Look")
"""

import matplotlib.pyplot as plt
from numpy import linspace, sin, cos
plt.xkcd(scale=2, length=200, randomness=10) # Aktiviere XKCD - Modus
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['bottom'].set_position('zero')
ax.xaxis.set_ticks_position('bottom')
plt.yticks([])
t = linspace(-1, 10)
plt.plot(t, sin(t), label = 'Sinus')
plt.plot(t, 0.5* cos(t), label = 'Kosinus')
plt.xlabel('Zeit', rotation=5); plt.ylabel('Ein paar Werte',rotation=85)
plt.title('Kurven!', rotation=-2)
plt.legend(loc='best', frameon=False)
plt.show()