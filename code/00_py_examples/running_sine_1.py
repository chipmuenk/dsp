# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:42:48 2012

Show an animated sine function and measure frames per second (FPS)
"""
import sys
sys.ps1 = 'Ciao'
import time
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt

x = np.random.randn(10)
print('ready to plot')
plt.plot(x)
plt.draw()
plt.show(block=False)

print('starting to sleep (or working hard)')
time.sleep(1)
plt.plot(x + 2)
plt.draw()
plt.show(block=False)

print('sleeping again (or more work)')
time.sleep(1)
print('now blocking until the figure is closed')
plt.show(block=True)