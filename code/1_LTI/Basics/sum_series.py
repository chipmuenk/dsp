# -*- coding: utf-8 -*-
"""
Berechne endliche geometrische Reihe für a = 0.5 und N = 10
Drucke Ergebnisse mit Formatierung

Author: Christian Muenker
"""

summe = 0

for i in range(0,10):
    summe +=  0.5**i
    # alte Syntax
    print('i = %2d, Teilsumme: %f' %(i, summe))
    # Python 3 Syntax
    print('i = {0:2d}, Teilsumme: {1:f}'.format(i, summe))

print(summe)