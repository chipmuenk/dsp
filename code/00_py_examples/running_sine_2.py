# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:42:48 2012

Show an animated sine function and measure frames per second (FPS)
"""
import sys
sys.ps1 = 'Ciao'

import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

plt.axis([0, 10, 0, 1])
plt.ion()

for i in range(10):
    y = np.random.random()
    plt.scatter(i, y)
    plt.pause(0.05)

while True:
    plt.pause(0.05)