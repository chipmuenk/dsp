#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 13:47:18 2018

@author: muenker
"""

import numpy as np
n = np.arange(10)
x = np.sin(np.pi * n / 5)
X = np.fft.fft(x)

# real_if_close() converts complex numbers to real if imag < 100*eps
# works for scalars or when condition is true for the WHOLE array
# returns an array in any case that cannot be format-printed -> convert to scalar
X3 = [np.asscalar(np.real_if_close(x)) for x in X]

for x in X3:
    print("{:.5g}".format(x))