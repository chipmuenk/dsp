# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 17:09:52 2016

http://stackoverflow.com/questions/9401658/matplotlib-animating-a-scatter-plot

"""

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def main():
    numframes = 100
    numpoints = 10
    color_data = np.random.random((numframes, numpoints))
    x, y, c = np.random.random((3, numpoints))

    fig = plt.figure()
    scat = plt.scatter(x, y, c=c, s=100)

    ani = animation.FuncAnimation(fig, update_plot, frames=range(numframes),
                                  fargs=(color_data, scat))
    # use xrange for Python 2
    plt.show()

def update_plot(i, data, scat):
    scat.set_array(data[i])
    return scat,

main()