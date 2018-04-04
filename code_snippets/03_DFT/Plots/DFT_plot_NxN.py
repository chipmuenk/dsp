# -*- coding: utf-8 -*-
"""
=== DFT_plot_NxN.py =================================================== 

Show DFT base functions

TODO: layout improvement

(c) https://github.com/unpingco/Python-for-Signal-Processing/blob/master/Fourier_Transform.ipynb
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, sqrt, exp, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

from matplotlib.patches import FancyArrow
import matplotlib.gridspec as gridspec

def dftmatrix(Nfft=32,N=None):
    'construct DFT matrix'
    k= np.arange(Nfft)
    if N is None: N = Nfft
    n = arange(N)
    U = np.matrix(exp(1j* 2*pi/Nfft *k*n[:,None])) # use numpy broadcasting to create matrix
    return U/sqrt(Nfft)

Nfft=16
v = ones((16,1))
U = dftmatrix(Nfft=Nfft,N=16)

# --- 
# hardcoded constants to format complicated figure

gs = gridspec.GridSpec(8,12)
gs.update( wspace=1, left=0.01)

fig =figure(figsize=(10,5))
ax0 = subplot(gs[:,:3])
fig.add_subplot(ax0)

ax0.set_aspect(1)
a=2*pi/Nfft*arange(Nfft)

colors = ['k','b','r','m','g','Brown','DarkBlue','Tomato','Violet', 'Tan','Salmon','Pink',
          'SaddleBrown', 'SpringGreen', 'RosyBrown','Silver',]
for j,i in enumerate(a):
  ax0.add_patch( FancyArrow(0,0,cos(i),sin(i),width=0.02,
                            length_includes_head=True,edgecolor=colors[j]))

ax0.text(1,0.1,'0',fontsize=16)
ax0.text(0.1,1,r'$\frac{\pi}{2}$',fontsize=22)
ax0.text(-1,0.1,r'$\pi$',fontsize=18)
ax0.text(0.1,-1.2,r'$\frac{3\pi}{2}$',fontsize=22)
ax0.axis(array([-1,1,-1,1])*1.45)
ax0.set_title('Radial Frequency')
ax0.set_xlabel('Real')
ax0.set_ylabel('Imaginary')

# plots in the middle column
for i in range(8):
  ax=subplot(gs[i,4:8])
  ax.set_xticks([]);  ax.set_yticks([])
  ax.set_ylabel(r'$\Omega_{%d}=%d\times\frac{2\pi}{16}$'%(i,i),fontsize=16,
                rotation='horizontal')
  ax.plot(U.real[:,i],'-o',color=colors[i])
  ax.plot(U.imag[:,i],'--o',color=colors[i],alpha=0.2)
  ax.axis(ymax=4/Nfft*1.1,ymin=-4/Nfft*1.1)
ax.set_xticks(arange(16))
ax.set_xlabel('n')

# plots in the far right column
for i in range(8):
  ax=subplot(gs[i,8:])
  ax.set_xticks([]);  ax.set_yticks([])
  ax.set_ylabel(r'$\Omega_{%d}=%d\times\frac{2\pi}{16}$'%(i+8,i+8),fontsize=16,
                rotation='horizontal')
  ax.plot(U.real[:,i+8],'-o',color=colors[i+8])
  ax.plot(U.imag[:,i+8],'--o',color=colors[i+8],alpha=0.2)
  ax.axis(ymax=4/Nfft*1.1,ymin=-4/Nfft*1.1)    
  ax.yaxis.set_label_position('right')
ax.set_xticks(arange(16))
ax.set_xlabel('n')

plt.show()