# -*- coding: utf-8 -*-
"""
===========================================================================
 FIL-CT-2D3D.py

  2D and 3D plots of various curves of Continous-Time filters

 (c) 2016 Christian Münker - files for the lecture "AACD"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

from mpl_toolkits.mplot3d import Axes3D # needed for 'projection3d'
from matplotlib import cm # Colormap
from  matplotlib import patches
import mpl_toolkits.mplot3d.art3d as art3d
#import mayavi.mlab as mlab
import scipy.special

import sys
sys.path.append('../..')
import dsp_fpga_lib as dsp

EXPORT = False
#BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
BASE_DIR = "D:/Daten/HM/AACD/1_2_Filters/Folien/img/"
FILENAME = "butterworth_filter"
FMT = ".png"

##======== 3D - Plots ========================
OPT_3D_FORCE_ZMAX = False # False: use absolut value zmax for limits
                            # True: use zmax_rel * max(H)
OPT_3D_PLOT_TYPE = 'MESH'#'SURF_MLAB'#'SURF_MLAB' # MESH, MESH_MLAB, SURF, CONTOUR
#OPT_3D_PLOT_TYPE = 'SURF_MLAB'
OPT_3D_ALPHA = 0.9 # transparency from 0 ... 1
ELEV = 20 # elevation
PHI = -30 # angle in x,y - plane

steps = 100;               # number of steps for x, y, z

zmin =  0.0; zmax = 1.0; # zmax-setting is only used when OPT_3D_FORCE_ZMAX = True
zmin_dB = -70
zmax_rel = 3 # Max. displayed z - value relative to max|H(f)|
#
plevel_rel = 1.1; # height of plotted pole position relative to zmax
zlevel_rel = 0.05; # height of plotted zero position relative to zmax
PN_SIZE = 10; # size of P/N symbols
#================================

W_c = 1; A_PB_log = 1; A_SB_log = 40.; L = 4
zeta = sqrt(3)/2 # damping factor for Bessel
zeta = 0.25
#[bb,aa] = sig.bessel(L, W_c, analog=True)
[bb,aa] = sig.butter(L, W_c, analog=True)
#[bb,aa] = sig.cheby1(L, A_PB_log, W_c, analog=True)
#[bb,aa] = sig.cheby2(L, A_SB_log, W_c, analog=True)
#[bb,aa] = sig.ellip(L, A_PB_log, A_SB_log, W_c, analog=True)


# Define system function from polynomial coefficients
# e.g. H(s) =  (b2 s^2 + b1 s + b0) / (a2 s^2 + a1 s + a0)
## Second order systems
#aa = [1, 2 * zeta * W_c, 1] # general 2nd order denominator
#bb = [W_c * W_c] # lowpass
#b =

# 1st order LP: H(s) = 1 / (s RC + 1)
#bb = [1]; aa = [1, 1]
# 1st order HP: H(s) = s RC / (s RC + 1)
#bb = [1, 0]; aa = [1, 1]
#bb = [1./3, 0]; aa = [1/3, 1] # w_c = 3 / tau
# 2nd order HP: H(s) = 0.5 (s RC)^2 / (s RC + 1)(s RC/2 + 1)
#bb = [0.5, 0, 0]; aa = [0.5, 1.5, 1]
#================ Biquad ====================
#
#[bb,aa] = np.real([bb,aa])
aa = np.real(aa)
bb = np.real(bb)

################### Calculate roots #############################
nulls = np.roots(bb) # zeros of H(s)
poles = np.roots(aa) # poles of H(s)

nulls =[-1100]
poles =[-11000]
#nulls = [0,0]
#poles = [-1,-2]
bb = np.poly(nulls)
aa = np.poly(poles)
print("aa, bb =", aa,bb)
print("P, N =", np.roots(aa), np.roots(bb))
print("Angle(P) = ", angle(np.roots(aa))/ pi * 180)


W_max = 2 # normalized circular frequency; W = 2 pi f tau
W = np.linspace(0, W_max, 201) # start, stop, step. endpoint is included
[W,H] = sig.freqs(bb, aa, W) # Calculate H(w) at the frequency points w1
#[w,H]=sig.freqs(bb,aa)  # calculate H(w) at 200 frequencies "around the
                        # interesting parts of the response curve"
f = W
H_abs = abs(H)
H_max = max(H_abs); H_max_dB = 20*log10(H_max)
W_max = W[np.argmax(H_abs)] # frequency where |H(Omega)| is max.
H_angle = np.unwrap(angle(H))

#====================================================
figure(1)
dsp.zplane(bb, aa, analog=True, style = 'square', anaCircleRad=W_c,
           mzc = (0,0.5,0) )
axzp = plt.gca()
axzp.set_xlabel(r'$\sigma \, / \,\omega_n \; \rightarrow$',fontsize=18)
axzp.set_ylabel(r'$j \omega \,  / \,\omega_n \;  \rightarrow$',fontsize=18)
plt.tight_layout()

#====================================================
fig2 = figure(2)
ax21 = fig2.add_subplot(111)
ax22 = ax21.twinx()

l1, = ax21.plot(W,abs(H))
l2, = ax22.plot(W,H_angle/pi,'g')
ax21.set_xlabel(r'$\omega\, / \,\omega_n \;  \rightarrow$')
ax21.set_ylabel(r'$|H(j \omega)| \; \rightarrow$')
ax22.set_ylabel(r'$\angle H(j \omega)/\pi \; \rightarrow$')
ax21.set_title(r'$\mathrm{Frequency\,Response}\; H(j \omega) $')
ax21.legend((l1,l2), (r'$ \left|{H(j \omega)}\right|$',
                      r'$\angle\{H(j \omega)\}$'), loc=1)
#                        bbox_transform=ax21.transAxes)
plt.tight_layout()

#====================================================
fig3 = figure(3)
ax31 = fig3.add_subplot(111)
ax32 = ax31.twinx()

tau_g, w_g = dsp.grp_delay_ana(bb, aa, W)
#print(np.shape(tau_g), np.shape(w_g))
#tau_g = (H_angle[1:]-H_angle[0:-1])/(w_g[1]-w_g[0])
#tau_g, w_g = dsp.grpdelay(bb, aa, analog = True)
l31, = ax31.plot(W,angle(H)/pi,'g')
l32, = ax32.plot(w_g, tau_g)

ax31.set_xlabel(r'$\omega\, / \,\omega_n \; \rightarrow$')
ax32.set_ylabel(r'$\tau_g  \{H(j \omega)\} \; \rightarrow$')
ax31.set_ylabel(r'$\angle H(j \omega)/\pi \; \rightarrow$')
ax31.set_title(r'$\mathrm{Phase \, and \, Group \, Delay \, of}\, H(j \omega) $')
ax31.legend((l31,l32),(r'$\angle H(j \omega) $', r'$\tau_g \{H(j \omega)\}$'))
plt.tight_layout()

#===============================================================
## Step Response
#===============================================================
figure(4)
sys = sig.lti(bb,aa)
t, y = sig.step2(sys, N=1024)
plot(t, y)
title(r'Step Response $h_{\epsilon}(t)$')
xlabel(r'$t \; \rightarrow$')


#===============================================================
## 3D-Plots
#===============================================================
xmin = -max(f); xmax = 1e-6;  # cartesian range definition
ymin = 0 #-max(f);
ymax = max(f);
#

if OPT_3D_FORCE_ZMAX == True:
    thresh = zmax
else:
    thresh = zmax_rel * H_max # calculate display thresh. from max. of H(f)

plevel = plevel_rel * thresh; # height of displayed pole position
zlevel = zlevel_rel * thresh; # height of displayed zero position
z_scale = 1.0

# Calculate limits etc. for 3D-Plots
x1 = np.linspace(xmin,xmax,steps,endpoint=True) # x-coordinates
y1 = np.linspace(ymin,ymax,steps,endpoint=True) # y-coordinates
zc = np.linspace(zmin,thresh,steps,endpoint=True) # z-coordinates

xm, ym = np.meshgrid(x1,y1); # cartesian grid
x = xm.T
y = ym.T
#x, y = np.mgrid[xmin:xmax:steps*1j, ymin:ymax:steps*1j]

#xc = x[:,0]
yc = y[0,:]

s = x + 1j*y # complex coordinate grid

fig5 = plt.figure(5)
ax12 = fig5.add_subplot(111,projection='3d')

#colormap gray;  #hsv / gray / default / colorcube / bone / summer / autumn
#extents=(-1,1, -1,1, -1,1)
if OPT_3D_PLOT_TYPE == 'MESH':
    g=ax12.plot_wireframe(x,y,dsp.H_mag(bb,aa,s,thresh),rstride=2, cstride=2,
                          linewidth = 1, color = 'gray')
                          #plot 3D-mesh of |H(z)| ; limit at |H(z)| = thresh
elif OPT_3D_PLOT_TYPE == 'MESH_MLAB':
    mlab.mesh(x, y, dsp.H_mag(bb,aa,s,thresh), colormap="bone")
elif OPT_3D_PLOT_TYPE == 'SURF_MLAB':
#    x, y = np.mgrid[-7.:7.05:0.1, -5.:5.05:0.05]
#    s = np.sin(x+y) + np.sin(2*x - y) + np.cos(3*x+4*y)
    fm1 = mlab.figure()

    p1 = mlab.plot3d(zeros(len(yc)), yc,
            dsp.H_mag(bb,aa,1j*yc*2*pi,thresh)*z_scale, color = (0,0,0))
            #extent = (0, 0,ymin, ymax,0,thresh)
    s = mlab.surf(x, y, dsp.H_mag(bb,aa,s,thresh), warp_scale = z_scale)
    #extent = (xmin, xmax,ymin, ymax,0,thresh)
    mlab.axes(z_axis_visibility = False)

    #mlab.outline()
    #mlab.xlabel('x ->')
    mlab.colorbar(object=s, title=None, orientation='vertical', nb_labels=None,
                  nb_colors=None, label_fmt=None)
    mlab.title('3D-Darstellung von |H(s)| |', height = 0.95, size = 0.25)

#plot 3D-surface of |H(z)| ; limit at |H(z)| = thresh:
elif OPT_3D_PLOT_TYPE == 'SURF':
    g=ax12.plot_surface(x,y,dsp.H_mag(bb,aa,s,thresh), alpha = OPT_3D_ALPHA,
                      rstride=2, cstride=2, cmap = cm.jet,
    linewidth=0, antialiased=False, edgecolor = 'k')
    # try cm.hot, gray, hsv

else:
    ax12.contourf3D(x,y,dsp.H_mag(bb,aa,s,thresh),rstride=5, cstride=5)

# Plot xyz-axes
ax12.plot([xmin-0.1, xmax + 0.1],[0,0],[0,0], linewidth=2, color ='k')
ax12.plot([0,0],[ymin-0.1, ymax+0.1],[0,0], linewidth=2, color ='k')
ax12.plot([0,0],[0,0],[zmin-0.1, thresh+0.1], linewidth=2, color ='k')
#ax12.scatter(xmax + 0.1, 0, 0, color = 'k', marker = ">", s = 30)
#ax12.scatter(0, 0, thresh+0.1, color = 'k', marker = "^", s = 40)

# plot -3dB line
ax12.plot([0,0],[1,1],[0,1],'r--', lw = 1)
ax12.plot([0,0],[0,ymax],[1/sqrt(2), 1/sqrt(2)],'r--', lw = 1)

# Plot |H(j omega)| along the j omega axis:
ax12.plot(zeros(len(yc)), yc, dsp.H_mag(bb,aa,1j*yc,thresh),
          linewidth=3, color = 'r');

# Plot the zeros at (x,y,0) with "stems":
for k in range(len(nulls)):
    if xmax >= nulls[k].real >= xmin and ymax >= nulls[k].imag >= ymin:
        ax12.plot([nulls[k].real, nulls[k].real],[nulls[k].imag, nulls[k].imag],
          [0, zlevel],linewidth=1,color='b')
        ax12.plot([nulls[k].real, nulls[k].real],[nulls[k].imag, nulls[k].imag], zlevel,
        'o', markersize = PN_SIZE, mec='blue', mew=2.0, markerfacecolor='none'); # plot nulls

# Plot the poles at |H(s_p)| = plevel with "stems"
for k in range(len(poles)):
    if xmax >= poles[k].real >= xmin and ymax >= poles[k].imag >= ymin:
        ax12.plot([poles[k].real, poles[k].real],[poles[k].imag, poles[k].imag],
                   [0, plevel],linewidth=1,color='r')
        ax12.plot([poles[k].real, poles[k].real],[poles[k].imag, poles[k].imag],
                    plevel, 'x', markersize = PN_SIZE, mew=2.0, mec='red')

i = arange(steps)
uc = exp(1j*pi*(1 - i/(2*steps))) # plot unit circle between pi / 2 and pi
ax12.plot(uc.real, uc.imag, 0, color = 'grey')

#uc = patches.Circle((0,0), radius=1, fill=False, color='grey', ls='solid', zorder=1)
#ax12.add_patch(uc)
#art3d.pathpatch_2d_to_3d(uc, z=0, zdir="z")


#ax12.set_title( r'3D-Darstellung von $|H(s)|$',fontsize=20);
#ax12.xaxis._axinfo['label']['space_factor'] = 50
ax12.xaxis.labelpad=15
ax12.yaxis.labelpad=15
ax12.zaxis.labelpad=-30
ax12.set_xlabel(r'$\sigma \, / \,\omega_n \;  \rightarrow$',fontsize=18)
ax12.set_ylabel(r'$j\omega \, / \,\omega_n \; \rightarrow$',fontsize=18)
#ax12.set_zlabel(r'$|{H(s)|} \; \rightarrow$')
ax12.set_zlabel(r'    $|{A(s = \sigma + j \omega)|} \; \rightarrow$')
#ax12.set_title(r'$|H(s = \sigma + j \omega)| $')
ax12.set_xlim3d(xmin, xmax)
ax12.set_ylim3d(ymin, ymax)
ax12.set_zlim3d(zmin, thresh*plevel_rel)
ax12.locator_params(axis = 'both', tight = True, nbins=4)
ax12.locator_params(axis = 'z', tight = True, nbins=8)
ax12.view_init(ELEV, PHI)

plt.tight_layout()
if EXPORT:
    fig5.savefig(BASE_DIR + FILENAME + '3d' + FMT)
plt.show()
