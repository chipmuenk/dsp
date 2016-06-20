# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:48:25 2016

@author: Christian Muenker

see https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#
#plt.rcParams['animation.ffmpeg_path'] = 'D:/Programme/ffmpeg/bin/ffmpeg.exe'
plt.rcParams['animation.ffmpeg_path'] = '/opt/ffmpeg-3.02/ffmpeg'
#plt.rcParams['savefig.bbox'] = 'tight' # tight - this garbles the video!!!

#movie_file = 'D:/Daten/basic_animation.mp4'
movie_file = '/home/muenker/Daten/basic_animation.mp4'



dpi = 100
fps = 30

ffmpeg_writer = animation.FFMpegWriter(fps = fps, extra_args=['-vcodec', 'libx264'])


#animation.MovieWriterRegistry.list()

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure()
fig.set_size_inches(4, 5, True)
ax = plt.axes(xlim=(0, 2), ylim=(-2, 2))
line, = ax.plot([], [], lw=2)

# initialization function: plot the background for each frame
def init():
    line.set_data([], [])
    return line,

# animation function.  This is called sequentially
def animate(i):
    x = np.linspace(0, 2, 1000)
    y = np.sin(2 * np.pi * (x - 0.01 * i))
    line.set_data(x, y)
    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
# frames: number of frames to be recorded
# interval: time between frames in ms = 1000/fps ; 
# total length = frames x interval
anim = animation.FuncAnimation(fig, animate, init_func=init, frames = 50, blit=True)

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
anim.save(movie_file, dpi=dpi, writer=ffmpeg_writer)

plt.show()

