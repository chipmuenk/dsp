# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 16:42:48 2012

Show an animated sine function and measure frames per second (FPS)

@author: Muenker_2
"""

import numpy as np
import time
import matplotlib.pyplot as plt

plt.ion() # interactive mode on

tstart = time.time()               # for profiling
x = np.arange(0, 2 * np.pi, 0.01)  # create x-array
line, = plt.plot(x, np.sin(x))
for i in np.arange(1, 200):
    line.set_ydata(np.sin(x + i / 10.0))  # update the data
    plt.draw()                         # redraw the canvas

print('FPS:' , 200 / (time.time() - tstart))