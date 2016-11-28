# -*- coding: utf-8 -*-
"""
Show an animated sine function and measure frames per second (FPS)
Important:
- Create an artist OUTSIDE the loop
- INSIDE the loop, update only the artist (not the background etc.)
- Use plt.pause() to update the current figure (a minimum delay is sufficient)
"""
import numpy as np
import time
import matplotlib.pyplot as plt

#plt.ion() # interactive mode on - not needed with plt.pause()

tstart = time.time()               # for profiling
x = np.arange(0, 2 * np.pi, 0.01)  # create x-array
line, = plt.plot(x, np.sin(x))	   # create an artist with a reference
for i in np.arange(1, 200):
    line.set_ydata(np.sin(x + i / 10.0))  # update the data without re-creating canvas & axes

#  The following two lines are not needed as plt.pause updates the active figure
#    plt.draw()                         # redraw the canvas
#    plt.show(block=False)              # same
    plt.pause(0.001) # this updates the plot

print('FPS:' , 200 / (time.time() - tstart))
plt.show() # without this, the script terminates and the figure closes
