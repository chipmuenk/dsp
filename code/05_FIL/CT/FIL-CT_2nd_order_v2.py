# -*- coding: utf-8 -*-
"""
=== FIL-CT_2nd_order.py ===================================================
 
Compare different continuous time second-order filters
 
(c) 2016 Christian Münker - Files for the lecture "AACD"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

from mpl_toolkits.mplot3d import Axes3D # needed for 'projection3d'
from matplotlib import cm # Colormap

import scipy.special

import sys
sys.path.append('../..')
import dsp_fpga_lib as dsp


plt.rcParams['lines.linewidth'] = 2

#================================

W_c = 2; A_PB_log = 0.5; A_SB_log = 40.; L = 2
filt_type = 'LP'
zeta = sqrt(3)/2 # damping factor for Bessel
zeta = 0.25
#%omega_n sqrt{ 1 - 2 %zeta^2 + sqrt{4 %zeta^4 -  4 %zeta^2  + 2}}
[bb,aa] = sig.bessel(L, W_c, analog=True)
[bb,aa] = sig.butter(L, W_c, analog=True)
[bb,aa] = sig.cheby1(L, A_PB_log, W_c, analog=True)
#[bb,aa] = sig.cheby2(L, A_SB_log, W_c, analog=True)
#[bb,aa] = sig.ellip(L, A_PB_log, A_SB_log, W_c, analog=True)


# Define system function from polynomial coefficients
# e.g. H(s) =  (b2 s^2 + b1 s + b0) / (a2 s^2 + a1 s + a0)
## Second order systems
aa = [1, 2 * zeta * W_c, 1] # general 2nd order denominator
bb = [W_c * W_c] # lowpass
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

#nulls = [0,0]
#poles = [-1,-2]
#bb = np.poly(nulls
#aa = np.poly(poles)
print("aa, bb =", aa,bb)
print("P, N =", np.roots(aa), np.roots(bb))
print("Angle(P) = ", angle(np.roots(aa))/ pi * 180) 

W_plot = 20 # max. frequency for plot
zeta = np.array([0.5, 0.6, 1/sqrt(2), sqrt(3)/2, 1.0, 1.2])
W_max = np.empty(len(zeta))
H_max = np.empty(len(zeta))
W_3dB = np.empty(len(zeta))
W_cz = np.ones(len(zeta))
for i in range(len(zeta)):
# Calculate the -3dB frequency of the normalized (W_n = 1) |H(W)|:

    if zeta[i] < 1/sqrt(2): # complex roots
        W_3dB_rel = sqrt(1 - 2 * zeta[i]**2 
                                        + 2* zeta[i] * sqrt(1 - zeta[i]**2))
        H_max_rel = 1 / (2 * zeta[i] * sqrt( 1 -  zeta[i]**2 ))
    else: # real roots, no maximum
        W_3dB_rel = sqrt(1 - 2 * zeta[i]**2 + sqrt(4 * zeta[i]**4 
                                                    -  4 * zeta[i]**2 + 2))
        H_max_rel = 1

    # Normalize W_c such that W_3dB = 1 for all filters        
    if filt_type == 'LP': # Lowpass
        W_cz[i] = W_c / W_3dB_rel
        W_3dB[i] = W_c * W_3dB_rel
        if zeta[i] < 1/sqrt(2): # complex roots
            W_max[i] = 1 / W_3dB[i] * sqrt(1 - 2 * zeta[i]**2)
        else: # real roots, no maximum
            W_max[i] = 0
        
    elif filt_type == 'HP': # Highpass
        W_cz[i] = W_c * W_3dB_rel
        W_3dB[i] = W_c / W_3dB_rel 
        if zeta[i] < 1/sqrt(2): # complex roots
            W_max[i] = W_3dB[i] / sqrt(1 - 2 * zeta[i]**2)
        else: # real roots, no maximum
            W_max[i] = W_plot
    else: # Bandpass, normalize to same center frequency
        W_cz[i] = W_c
        W_3dB[i] = W_c / W_3dB_rel 
        if zeta[i] < 1/sqrt(2): # complex roots
            W_max[i] = W_3dB[i] / sqrt(1 - 2 * zeta[i]**2)
        else: # real roots, no maximum
            W_max[i] = W_plot
     
    aa = [1, 2 * zeta[i] * W_cz[i], W_cz[i]**2] # general 2nd order denominator

    if filt_type == 'LP':
        bb = [W_cz[i] * W_cz[i] * 1] # lowpass
    elif filt_type == 'HP':
        bb = [1, 0, 0] # highpass
    else:
        bb = [2*zeta[i] * W_cz[i], 0] # bandpass
    
    zeta_label = r'$\zeta = ${0:.2f}'.format(zeta[i])
    
    # Calculate |H_max(W_max)|
    W_max[i], H_max[i] = np.abs(sig.freqs(bb, aa, [W_max[i]]))
    W = np.linspace(0, W_plot, 1001) # start, stop, step. endpoint is included
    [W,H] = sig.freqs(bb, aa, W) # Calculate H(w) at the frequency points W
    #[w,H]=sig.freqs(bb,aa)  # calculate H(w) at 200 frequencies "around the
                            # interesting parts of the response curve"
    H_abs = abs(H)
#    H_abs = H_abs / abs(H_max[i])
    Hmax = max(H_abs); Hmax_dB = 20*log10(Hmax) 
    Wmax = W[np.argmax(H_abs)] # frequency where |H(Omega)| is max.
    H_angle = np.unwrap(angle(H))
    H_abs = abs(H)/Hmax
    print('--------------------------------')    
    print('zeta[i] = ',zeta[i])
    print('W_3dB[i] = ', W_3dB[i])
    print('W_max, H_max (calc.) = ',W_max[i], H_max[i])
    print('Wmax, Hmax (sim.) = ',Wmax, Hmax)
    print('H_max_rel = ', H_max_rel)
    
    #===============================================================
    ## Poles and zeros
    #===============================================================
    figure(1)
    dsp.zplane(bb, aa, analog=True, style = 'square', anaCircleRad=W_c,
               mpc = (1-i/len(zeta),i/len(zeta),i/len(zeta)), plabel = zeta_label, lw = 3)
    axzp = plt.gca()
    axzp.set_xlabel(r'$\sigma \; \rightarrow$',fontsize=18)
#    axzp.set_xlabel(r'$\sigma \,/ \omega_{3dB}\; \rightarrow$',fontsize=18)
    axzp.set_ylabel(r'$j \omega \; \rightarrow$',fontsize=18)
#    axzp.set_ylabel(r'$j \omega \,/ \omega_{3dB}\; \rightarrow$',fontsize=18)
    axzp.legend(loc = 'best')
    plt.tight_layout()
    
    #===============================================================
    ## Magnitude Response
    #===============================================================
    
    fig2 = figure(2)
    ax21 = fig2.add_subplot(111)   
    l1, = ax21.plot(W,H_abs, label = zeta_label)
    if i == 0: 
        plt.axhline(y=1/sqrt(2), lw = 1, c = 'k', ls = '-.')
        plt.axvline(x=W_c,  lw = 1, c = 'k', ls = '-.')
    ax21.set_xlabel(r'$\omega\,/ \, \omega_{3dB} \; \rightarrow$')
    ax21.set_ylabel(r'$|H(j \omega)| \; \rightarrow$')
    ax21.legend(loc ='best')
    ax21.set_title(r'$\mathrm{Magnitude\,Response}\; |H(j \omega)| $')
    plt.axis([0,2*W_c, 0.1,1.05])
    
    #===============================================================
    ## Log. Magnitude Response
    #===============================================================
    
    fig3 = figure(3)
    ax31 = fig3.add_subplot(111)   
    ax31.set_title(r'$\mathrm{Log.\, Magnitude\,Response}\; 20 \, \log\, |H(j \omega)| $')  
    ax31.semilogx(W,20*log10(H_abs), label = zeta_label)
    if i == 0: 
        plt.axhline(y=-3, lw = 1, c = 'k', ls = '-.')
        plt.axvline(x=W_c,  lw = 1, c = 'k', ls = '-.')
    ax31.set_xlabel(r'$\omega\,/ \, \omega_{3dB} \; \rightarrow$')
    ax31.set_ylabel(r'$20\, \log\,|H(j \omega)| \; \rightarrow$')
    ax31.legend(loc='best')
    plt.axis([0.2,20, -60,1])
    fig3.tight_layout()
  
    #===============================================================
    ## Group Delay
    #===============================================================
    fig4 = figure(4)
    ax41 = fig4.add_subplot(111)
    
    tau_g, w_g = dsp.grp_delay_ana(bb, aa, W)
    tau_g2, w_g2 = dsp.grpdelay(bb, aa, analog = True, Fs = max(W))

    l41, = ax41.plot(w_g, tau_g, label = zeta_label)
    l42, = ax41.plot(w_g2, tau_g2)

    
    ax41.set_xlabel(r'$\omega\,/\, \omega_{3dB} \; \rightarrow$')
    ax41.set_ylabel(r'$\tau_g (j \omega)\; \rightarrow$')
    ax41.set_title(r'$\mathrm{Group \, Delay \, of}\, H(j \omega) $')
    plt.xlim(0,2*W_c)
    ax41.legend(loc='best')    
#    ax31.legend((l31),(r'$\tau_g \{H(j \omega)\}$'))
    plt.tight_layout()
    
    #===============================================================
    ## Step Response
    #===============================================================
    figure(5)
    sys = sig.lti(bb,aa)
    T = arange(0,5,0.05)
    t, y = sig.step2(sys, T = T, N=1024)
    plot(t, y, label = zeta_label)
    title(r'Step Response $h_{\epsilon}(t)$')
    xlabel(r'$t \,/ \, \tau \,\; \rightarrow$')
    plt.legend(loc='best')

plt.show()
