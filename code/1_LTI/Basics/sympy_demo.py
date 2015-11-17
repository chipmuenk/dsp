# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:55:50 2013

@author: Muenker_2
"""

import sympy
from sympy import integrate, sin, cos, pi
from sympy.abc import f,x
# unbestimmte Integration
print integrate(cos(x*f)**2, x)
# umgeformt, multipliziere mit Konstante
print integrate(cos(x*f)**2, x).expand(mul = True)
# bestimmte Integration zwischen 0 und 1/f
print integrate(cos(2*pi*10*f*x)**2, (x, 0, 1/f))