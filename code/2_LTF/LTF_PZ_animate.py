# -*- coding: utf-8 -*-
"""
Created on Fri May 13 09:48:25 2016

Demonstrate relationship between P/Z and magnitude frequency response

@author: Christian Muenker

see https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
   https://zulko.github.io/blog/2013/09/27/read-and-write-video-frames-in-python-using-ffmpeg/
   http://mbi-wiki.uni-wuppertal.de/python/animation-einer-lininegrafik-mit-python/
   http://stackoverflow.com/questions/9401658/matplotlib-animating-a-scatter-plot#9416663
"""

import numpy as np
from numpy import pi, exp, sin, cos
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import sys
sys.path.append('..')
from dsp_fpga_lib import zplane

#
# 

#plt.rcParams['animation.ffmpeg_path'] = 'D:/Programme/ffmpeg/bin/ffmpeg.exe'
plt.rcParams['animation.ffmpeg_path'] = '/opt/ffmpeg-3.02/ffmpeg'
#plt.rcParams['savefig.bbox'] = 'tight' # tight - this garbles the video!!!

dpi = 100
fps = 30
N = 50 # number of frames
#movie_file = 'D:/Daten/basic_animation.mp4'
movie_file = '/home/muenker/Daten/PZ_animation.mp4'
ffmpeg_writer = animation.FFMpegWriter(fps = fps, extra_args=['-vcodec', 'libx264'])
#animation.MovieWriterRegistry.list()

#P = np.array([0.8* exp(1j * pi * 0.3), 0.8* exp(-1j * pi * 0.3)])
P = [-0.9]
#Z = np.array([exp(1j * pi * 0.6), exp(-1j * pi * 0.6)])


P = np.array(P)
#Z = np.array(Z)

Z = 1/P


# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure(1)
#fig.clf()
fig.set_size_inches(10, 5, True)
ax1 = fig.add_subplot(121, autoscale_on=False, xlim=(-2, 2), ylim=(-2, 2),  aspect=1)
ax1.grid(True)
ax2 = fig.add_subplot(122, xlim=(0, 1), ylim=(0, 4))
ax2.set_xlabel(r"$F \; \rightarrow$")
ax2.set_ylabel(r"$|H \left( \mathrm{e}^{j 2 \pi F}\right) |\; \rightarrow$")
ax2.grid(True)

color_data = ['r', 'g', 'b']
scat = (0,0)

# Now pre-calculate coordinates along the unit circle and the distances 
phi = np.linspace(0, 1, N)
x_uc = cos(2*pi*phi)
y_uc = sin(2*pi*phi)

pl_x = []
pl_y = []
pl = 1
zl_x = []
zl_y = []
zl = 1
mag = []
for k in range(len(P)):
    pl *= np.abs(x_uc + 1j*y_uc - P[k])
for k in range(len(Z)):
    zl *= np.abs(x_uc + 1j*y_uc - Z[k])

mag = zl / pl
print(len(zl), len(mag))


line_p, = ax1.plot([],[], 'r', lw = 0.5) # line between poles and frequency point
line_z, = ax1.plot([],[], 'b', lw = 0.5) # line(s) between zeros and frequency point
point1, = ax1.plot([],[], 'o', ms = 10, mec = 'grey',  mfc = 'yellow', alpha = 0.5)
text = ax1.text(-1.5,-1.5, "", ha = 'left', va = 'center')
line2, = ax2.plot([],[], 'o', ms = 10, mec = 'grey',  mfc = 'green', alpha = 0.5)

# initialization function: plot the background for all frames
# this function is called at the beginning of each animation cycle
def init():
    ax1.plot(x_uc, y_uc, "grey") # plot unit circle
    ax1.scatter(P.real, P.imag, marker = "x", color = 'red', s = 20) # plot poles
    ax1.scatter(Z.real, Z.imag, marker = "o", color = 'blue', s = 20) # plot zeros
    ax2.plot(phi, mag, 'k', lw = 2)
    line_p.set_data([], []) # dlear data for 
    line_z.set_data([], [])
    point1.set_data([], [])
    line2.set_data([], [])
    
    text.set_text('')
    return line_p, line_z, point1, line2,

# This is the actual animation function: The first argument is an integer that
# is incremented with each interation, the other arguments are defined in the
# fargs kw of FuncAnimation
# Arguments passed back are animated
# 
def animate(i, color_data, scat):
#    x2 = np.linspace(0, 1, N)
#    y2 = y2.append
#    y = np.sin(2 * np.pi * (x - 0.01 * i))
#    x = x_uc[i] #cos(2*pi * i / N)
#    y = y_uc[i]# = sin(2*pi * i / N)
    pl_x = []
    pl_y = []
    zl_x = []
    zl_y = []
    for k in range(len(P)):
        pl_x.append([x_uc[i], P[k].real])
        pl_y.append([y_uc[i], P[k].imag])
    for k in range(len(Z)):
        zl_x.append([x_uc[i], Z[k].real])
        zl_y.append([y_uc[i], Z[k].imag]) 
    line_p.set_data(pl_x, pl_y)
    line_z.set_data(zl_x, zl_y)

    print(i, " von ", N)
#    line2.set_data(phi[:i], mag[:i])
    line2.set_data(phi[i], mag[i])
    point1.set_data(x_uc[i],y_uc[i])
    text.set_text(
        str("ZL = %.2f" %(zl[i]) + "\n" + "PL = %.2f" %(pl[i]))  )  
#    print(i,x,y)
    return line_p, line_z, point1, line2, text

# call the animator.  blit=True means only re-draw the parts that have changed.
# frames: number of frames to be recorded
# interval: time between frames in ms = 1000/fps ; 
# total length = frames x interval
anim = animation.FuncAnimation(fig, animate, init_func=init, frames = N, blit=True,
                               fargs=(color_data, scat))

# save the animation as an mp4.  This requires ffmpeg or mencoder to be
# installed.  The extra_args ensure that the x264 codec is used, so that
# the video can be embedded in html5.  You may need to adjust this for
# your system: for more information, see
# http://matplotlib.sourceforge.net/api/animation_api.html
#anim.save(movie_file, dpi=dpi, writer=ffmpeg_writer)
fig.tight_layout()
plt.show()

