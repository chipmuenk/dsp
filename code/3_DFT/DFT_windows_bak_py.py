#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
#===========================================================================
# ueb_DFT_Basic_py.py
#
# Demonstrate how various parameters of the DFT 
# - window length
# - window type
# - sample rate
# influence the displayed spectrum
# 
#
# (c) 2014-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals # v3line15

import numpy as np
import numpy.random as rnd
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig
import scipy.interpolate as intp

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

#import dsp_fpga_lib as dsp
#------------------------------------------------------------------ v3line30
# ... Ende der Import-Anweisungen

def lim_eps(a,eps):
    """
    Return min / max of an array a, increased by eps*(max(a) - min(a)). 
    Handy for nice looking axes labeling.
    """
    mylim = (min(a) - (max(a)-min(a))*eps, max(a) + (max(a)-min(a))*eps)
    return mylim
       
#------------------------------------------------
#Initialize variables
#------------------------------------------------
fs = 5500.0    # sampling frequency
Ts = 1.0/fs      # sampling period
OSR = 25.       # "Oversampling Ratio" for displaying pseudo-analog signal
M_FFT  = 50    # DFT points
L = 3          # show L DFT windows of time signal

DC   = 0.0    # DC-Offset
fsig1 = 550.0;  fsig2 = 197.0  # analog signal frequencies,
A1   = 1;       A2 = 0          # amplitudes and
phi1 = pi/2;    phi2 = pi/4     # starting phases
Anoi = 0.00; # noise amplitude (uniform distribution)
Xn_dB_min = -80 # minimum displayed dB value
#-------------------------------------------------
tstep = 1.0/(fs*OSR) # time step for plotting "analog" signal

Tmeas = M_FFT * Ts # calculate meas. window of FFT

mk_lg = 8  # markersize large for stemplots
mk_sm = 6  # markersize small for stemplots
#---------------------------------------------
# Window functions - select one!
wn = sig.windows.boxcar(M_FFT * OSR); myWindow ='Rechteck'
#wn = sig.windows.hann(M_FFT * OSR); myWindow ='Hann'
wn = sig.windows.hamming(M_FFT * OSR); myWindow ='Hamming'
#wn = sig.windows.flattop(M_FFT * OSR); myWindow ='Flattop'
#wn = sig.windows.bartlett(M_FFT * OSR); myWindow ='Bartlett'
#
# Normalized Equivalent Noise Bandwidth of window:
NENBW = len(wn)*np.sum(wn**2)/np.sum(wn)**2
print('NENBW = ', NENBW)
#------------------------------------------------
t = arange(0,Tmeas,tstep) # "analog" time 
t2 = arange(0,Tmeas*L,tstep) 
n = arange(0,M_FFT)  # discrete samples
n2 = arange(0,M_FFT*L)
#---------------------------------------------
# Calculate "analog" signals - select one
noise_t = Anoi * (np.random.rand(len(t)) - 0.5)                      
xt = DC + A1 * cos(2.0*pi*fsig1*t + phi1) \
    + A2 * cos(2.0*pi*fsig2*t + phi2) + noise_t
#xt = DC + A1 * sgn(cos(2.0*pi*fsig1*t + phi1)) 
#xt = np.sign(xt)

#---------------------------------------------
# Window the time signal:
xtw = xt * wn

xtw = np.tile(xtw, L) # repeat "oversampled" data L times
xn = xtw[::OSR] # sample data by picking every OSR sample

#==============================================================================
#-------------------------------------------------------------
#    Figure 1: Plot windowed "analog" and sampled signal
#-------------------------------------------------------------
# create new figure(1) if it does not exist, else make it active 
# clear it and return a reference to it. 
fig1 = figure(1); fig1.clf() # 
ax11 = fig1.add_subplot(111) # use object oriented matplotlib API
                            # for more flexibility
ax11.plot(t, xtw[0:M_FFT*OSR], 'b-') # M_FFT samples (T_meas) of x as blue line

ax12 = ax11.twiny() # twiny: make two plots with same y- but different x-axes
markerline, stemlines, baseline = ax12.stem(n[0:M_FFT], xn[0:M_FFT])
plt.setp(markerline, 'markerfacecolor', 'r', 'markersize', mk_lg)
plt.setp(stemlines, 'color','r', 'linewidth', 2)
plt.setp(baseline, 'linewidth',0) # turn off baseline
if L > 1: # show other measurement windows in grey
    ax11.plot(t2[M_FFT*OSR:], xtw[M_FFT*OSR:], color='grey', linestyle='-') 
    ml, sl, bl = ax12.stem(n2[M_FFT:], xn[M_FFT:]) 
    plt.setp(ml, 'markerfacecolor', 'grey', 'markersize', mk_sm)
    plt.setp(sl, 'color','grey', 'linewidth', 1, linestyle='-')
    plt.setp(bl, 'linewidth',0) # turn off baseline
    ax11.set_xlim([0, L * Tmeas])
    ax12.set_xlim([0, L * M_FFT]) # match range for second x-axis with first one
else:
    ax11.set_xlim([0, Tmeas])
    ax12.set_xlim([0, M_FFT]) # match range for second x-axis with first one

ax11.set_xlabel(r'Zeit / s $\rightarrow$')
ax11.set_ylabel(r'Amplitude / V $\rightarrow$')
ax11.grid(axis='y') # plot y-gridlines for both x-axes
ax12.set_xlabel(r'$n \rightarrow$')
ax12.grid('on') # plot x-gridlines for second x-axis
ax12.text(0.05 * M_FFT*L, min(xtw)+(max(xtw)-min(xtw))*0.03,
         r'$f_S = %.1f$ Hz, $f_{sig1} = %.1f$ Hz' %(fs,fsig1), fontsize=18,
         ha="left", va="bottom",linespacing=1.5,
         bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='0.9'))                         
ax12.set_ylim(lim_eps(xtw,0.05))    # set ylim to min/max of xtw
# Draw a horizontal lines at y from xmin to xmax (rel. coordinates):
ax12.axhline(0, xmin = 0, xmax = 1, linewidth=1, color='k')
#
fig1.tight_layout(rect=[0, 0, 1, 0.96]) # make room for title
fig1.text(0.5, 0.99, "Gefensterte Zeitfunktion",
         ha='center', va = 'top',
         fontsize=20, transform = fig1.transFigure) # create title
#fig1.savefig('D:/Daten/pueb_LTIML-Sampling_%sHz.pdf' %int(fs))

#-------------------------------------------------------------
#     Figure 2: Plot Window Function and its DFT
#-------------------------------------------------------------
Wn = fftshift(fft(wn[::OSR], M_FFT*4))/(M_FFT/2)
wf = fftshift(fftfreq(M_FFT*4))
#
fig2 = figure(2); fig2.clf()
ax21 = fig2.add_subplot(311); grid(True)
ax21.set_title('Fensterfunktion im Zeit- und Frequenzbereich')
ax21.stem(n, wn[::OSR])
ax21.text(0.01, 0.9, myWindow, fontsize=16,
         ha="left", va="top", transform = ax21.transAxes,
         bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='0.9'))      
ax21.set_xlabel(r'$n \, \rightarrow $')
ax21.set_ylabel(r'$w[n] \; \rightarrow $')
ax21.set_ylim(-0.1, 1.1)
#
ax22 = fig2.add_subplot(312); plt.grid(True)
ax22.plot(wf, abs(Wn))
ax22.set_xlim(-0.5, 0.5)
ax22.set_xlabel(r'$F \; \rightarrow $')
ax22.set_ylabel(r'$\| W(\mathrm{e}^{\mathrm{j} 2 \pi F})\| \rightarrow $ ')
#
ax23 = fig2.add_subplot(313); plt.grid(True)
ax23.plot(wf, np.maximum(20 * log10(abs(Wn)), Xn_dB_min))
ax23.set_xlim(-0.5, 0.5)
ax23.set_xlabel(r'$F \; \rightarrow $')
ax23.set_ylabel(r'$\| W(\mathrm{e}^{\mathrm{j} 2 \pi F})\| \mathrm{/ dB} \rightarrow $ ')
fig2.tight_layout()
#fig2.tight_layout(rect=[0, 0, 1, 1])

#-------------------------------------------------------------
#    Figure 3: CALCULATE AND PLOT SINGLE-SIDED DFT
#-------------------------------------------------------------
# Calculate two-sided DFT and scale it with 1/M_FFT:
Xn = fft(xn, n=M_FFT)/ M_FFT 
# f = [0 ... f_S[ = [0... f_S/2[, [-f_S/2 ... 0[
xf  = fftfreq(M_FFT, Ts)
xfn = fftfreq(M_FFT, d=1.0/M_FFT) 
# Corresponding freq. points at [0... f_S/2[, [-f_S/2 ... 0[
#
# Calculate Xn in dBs with a fixed minimum
Xn_dB = np.maximum(20*log10(abs(Xn)),Xn_dB_min)

fig3 = plt.figure(3); fig3.clf()
ax31 = fig3.add_subplot(211); ax31.grid('on')
plt.title('Einseitige DFT $S[k]$') # Overall title
ml, sl, bl = ax31.stem(xf[:M_FFT/2],2.*abs(Xn[:M_FFT/2]))
plt.setp(ml, 'markerfacecolor', 'r', 'markersize', mk_lg) # markerline
plt.setp(sl, 'color','r', 'linewidth', 2) # stemline
plt.setp(bl, 'linewidth',0) # turn off baseline
plt.ylabel(r'$|S(f)|$' )
ax31.set_ylim(0,1)
ax31.set_xlim(0,fs/2.)
plt.axhline(0, xmin = 0, xmax = 1, linewidth=1, color='k')
plt.axvline(x=0, ymin = 0, ymax = 1, linewidth=1, color='k')
ax31.set_xlabel(r'$f \; \mathrm{[Hz]} \;\rightarrow $')
#
ax32 = fig3.add_subplot(212)
ml, sl, bl = ax32.stem(xfn[:M_FFT/2],Xn_dB[:M_FFT/2],bottom=Xn_dB_min)
plt.setp(ml, 'markerfacecolor', 'r', 'markersize', mk_lg) # markerline
plt.setp(sl, 'color','r', 'linewidth', 2) # stemline
plt.setp(bl, 'color','k') # black baseline

plt.axhline(0, xmin = 0, xmax = 1, linewidth=1, color='k')
plt.axvline(x=0, ymin = 0, ymax = 1, linewidth=1, color='k')

ax32.set_xlabel(r'$k \; \rightarrow $')
ax32.set_ylabel(r'$20 \log |S[k]| / \mathrm{V} \, \rightarrow$' )

plt.tight_layout()
plt.grid('on')

#-------------------------------------------------------------
#    Figure 4: CALCULATE AND PLOT TWO-SIDED DFT
#-------------------------------------------------------------
# fftshift centers freq. vector to f = [-f_S/2... f_S/2[
Xn = fftshift(Xn)
# set Xn = 0 for very small magnitudes to eliminate 
# numeric errors in phase calculation:
#Xn = np.real((abs(Xn/max(abs(Xn))) > 1.0e-10)) * Xn
# - doesn't work - result is 0j instead of 0!

for i in range(len(Xn)): 
    # set phase = 0 for very small magnitudes (numeric errors)
    if abs(Xn[i]/max(abs(Xn))) < 1.0e-8: Xn[i] = 0

xf = fftshift(xf) # Center frequency vector 
xfn = fftshift(xfn) # Center frequency vector

fig4 = figure(4); fig4.clf() # 
ax41 = fig4.add_subplot(211)
ml, sl, bl = ax41.stem(xf,abs(Xn),'r')
plt.setp(ml, 'markerfacecolor', 'r', 'markersize', mk_lg) # markerline
plt.setp(sl, 'color','r', 'linewidth', 2) # stemline
plt.setp(bl, 'linewidth',0) # turn off baseline

plt.axhline(0, xmin = 0, xmax = 1, linewidth=1, color='k')
plt.axvline(x=0, ymin = 0, ymax = 1, linewidth=1, color='k')

ax41.grid('on')
ax41.text(min(lim_eps(xf,-0.03)), 0.05,'$f_S = %.1f$ Hz, $\Delta f = %.1f$ Hz' %(fs,fs/M_FFT), fontsize=16,
         ha="left", va="bottom",linespacing=1.5,
         bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='0.9'))       
ax41.set_xlim(lim_eps(xf, 0.05))
ax41.set_ylim(lim_eps(abs(Xn), 0.1))
ax41.set_xlabel(r'$f \;\mathrm{[Hz]} \; \rightarrow$')
ax41.set_ylabel(r'$|S[k]| / V \rightarrow$' )
ax41.set_title('Zweiseitige DFT $S[k]$') # Overall title
#
ax42 = fig4.add_subplot(212)
ax42.stem(xf,angle(Xn)/pi)
ax42.set_ylabel(r'$\angle S[k] / \mathrm{rad} /\pi \rightarrow$' )
#ax42.set_ylim(lim_eps(Xn_dB, 0.1))
plt.axhline(0, xmin = 0, xmax = 1, linewidth=1, color='k')
#ax42.set_xlim(lim_eps(xfn, 0.05))
ax42.set_xlabel(r'$n \; \rightarrow$')
plt.axvline(x=0, ymin = 0, ymax = 1, linewidth=1, color='k')
plt.tight_layout()

plt.grid('on')
#plt.savefig('D:/Daten/ueb-DFT_Basics_1-ML_DFT%s.png' %int(M_FFT))
#==============================================================================

plt.show()

#------ not needed here but useful ---------
#ttl = plt.title('Sampling') # print title and get handle
#ttl.set_y(1.1) # increase y-position (rel. coordinates)
#plt.subplots_adjust(top=0.86)
