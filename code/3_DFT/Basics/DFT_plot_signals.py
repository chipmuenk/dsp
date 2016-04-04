#!/usr/bin/env python
# -*- coding: iso-8859-15 -*-
# DFT_plot_signals.py ====================================================
# 
#
# Einfache Plots zum Kapitel "DFT"
#
# Thema: Zeitdiskrete Faltung
#
#
#
# 
# (c) 2016-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
#===========================================================================
from __future__ import division, print_function, unicode_literals, absolute_import # v3line15

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

import dsp_fpga_lib as dsp
#-------- ----------------------------------------------------------------
# ... Ende der gem. import-Anweisungen
BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
SCALE = True
NAME = "FInt"
PERIODIC_T = False # single or repeated pulse
PERIODIC_F = False # single or repeated spectrum
NDISP = 3 # number of displayed repetitions in t and f domain
NPER = 10 # number of periods for FFT window
DISCRETE_T = False # discrete time
DISCRETE_F = True # discrete frequencies
PHASE = False
ZEROPAD = False
NFFT = 64
OSR = 10 # Oversampling ratio for analog display

Ts = 1.0/50.      # sampling period


NPAD = NFFT * 99 # amount of zero padding

# generate time arrays for one signal period
n = arange(NFFT) # discrete samples 0 ... NFFT
t = linspace(0,1,num=NFFT*OSR) # "analog" time 0 ... 1 in NFFT*OSR steps
#y = np.sin(2* pi* n/(NFFT/2)) + np.sin(2* pi* n/(NFFT/4))# *np.cos(pi* n/(2*len(n)))#*np.exp(-n/(len(n)))
yt = sig.waveforms.square(t * 2*pi + pi/2, duty = 0.5) + 1

if not PERIODIC_T:
#    y = np.concatenate((np.zeros(len(n)), y, np.zeros(len(n))))
    yt = np.concatenate((np.zeros(len(t)), yt, np.zeros((NPER - 2) * len(t))))

else:
    yt = np.tile(yt,NPER)

#xticklabels = n
    
y = yt[0:NFFT*OSR*NPER:OSR] # sample discrete time signal from "anlog signal"
b,a = sig.butter(8,0.2)
y = sig.filtfilt(b,a,y)

n = linspace(0, NPER, num = len(y))
t = linspace(0, NPER, num = len(yt))


#print(len(n), len(y))


#plt.xkcd(scale=1, length=200, randomness=20) # Aktiviere XKCD - Modus
fig1 = plt.figure(num=1, figsize=(12,3))
ax1 = fig1.add_subplot(111)
#stem(n, y, 'r')f
grid(False)
#

#
#ax1.set_xlabel(r'$n \rightarrow$')
#ax1.xaxis.set_label_coords(2, 0.5, transform=ax1.transData)
#ax1.set_ylabel(r'$x[n] \rightarrow$')
#ax1.yaxis.set_label_coords(-0.3, 2.3, transform=ax1.transData)
#ax1.set_ylim([-2,3])
#ax1.set_ylim([-2,2.5])
#

if SCALE:
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_yaxis().set_ticks([])
    #ax1.yaxis.set_ticks_position('none')
    #ax1.spines['left'].set_position(('data',0))
    #ax1.spines['left'].set_linewidth(2)
    
    ax1.xaxis.set_ticks_position('bottom')
    ax1.set_xticklabels([])
    ax1.spines['bottom'].set_position(('data',0))
    ax1.spines['bottom'].set_linewidth(2)
    ax1.spines['bottom'].set_zorder(0)
    ax1.tick_params(axis='x', direction='inout', width=2, length=7,
                           color='k', labelsize=20, pad=5) # 'inout' 'in'
#[tick.set_zorder(0) for tick in ax1.xaxis.ticklabels]
#[line.set_zorder(1) for line in ax1.lines]
#resadjust(ax1, yres = 1)
#for label in ax1.get_xticklabels():
#       label.set_horizontalalignment('left')

if DISCRETE_T:
    markerline, stemlines, baseline = ax1.stem(n[0:NDISP*NFFT], y[0:NDISP*NFFT])
    plt.setp(markerline, 'markerfacecolor', 'r', 'markersize', 8, 'marker', 'o')
    plt.setp(stemlines, 'color','b', 'linewidth', 2)
    plt.setp(baseline, 'linewidth', 0) # turn off baseline
#    plot(t, yt,'b')
else:
    plot(n[0:NDISP*NFFT],y[0:NDISP*NFFT],'r',linewidth=3)
#    plot(t, yt, 'b')
#for label in ax1.get_yticklabels():
#       label.set_verticalalignment('bottom')
#ax1.set_xticklabels(xticklabels, rotation=0, ha='left', minor=False)
#
ax1.set_xlabel(r'$t \; \rightarrow$', size = 24)
ax1.xaxis.set_label_coords(0.98, 0.25)# transform=ax1.transData)   
ax1.set_ylim(-0.3, 2.3)
#plt.margins(0.02) # setting xmargin / ymargin individually doesnt work
fig1.tight_layout()

#ax1.spines['left'].set_smart_bounds(True)
#ax1.spines['bottom'].set_smart_bounds(True)
#ax1.set_title(r'Faltung $y[n] = x[n] \star \{1; 1; 1; 1; 1\}$')

#plt.ticklabel_format(useOffset=False, axis='y') # disable using offset print
fig1.savefig(BASE_DIR + NAME + '_t.png')

fig2 = figure(num=2, figsize=(12,3))

ax2 = fig2.add_subplot(111)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.get_yaxis().set_ticks([])
#ax2.spines['right'].set_position(('data',0))
#ax1.yaxis.set_ticks_position('none')
#ax1.spines['left'].set_position(('data',0))
#ax1.spines['left'].set_linewidth(2)

ax2.xaxis.set_ticks_position('bottom')
ax2.set_xticklabels([])
ax2.spines['bottom'].set_position(('data',0))
ax2.spines['bottom'].set_linewidth(2)
ax2.spines['bottom'].set_zorder(0)
ax2.tick_params(axis='x', direction='inout', width=2, length=7,
                       color='k', labelsize=20, pad=5) # 'inout' 'in'


if ZEROPAD:
    print(NPAD)
    print(np.zeros(NPAD))
    yn = np.concatenate((y, np.zeros(NPAD)))
else:
    yn = y
    
#Y = fft(yn)[0:len(yn)/2]/len(y)
NNFFT = len(yn)
Y = fftshift(fft(yn))[220:NNFFT-220]
print(len(Y), len(yn))
YM = np.abs(Y)
F = fftshift(fftfreq(len(YM)))
#F = arange(len(Y))
#F = fftfreq(len(y))

if PERIODIC_F:
    YM = np.tile(YM,NDISP)
else:
    #    y = np.concatenate((np.zeros(len(n)), y, np.zeros(len(n))))
    YM = np.concatenate((np.zeros(len(YM)), YM, np.zeros((NDISP - 2) * len(YM))))
F = fftshift(fftfreq(len(YM)))
    
if DISCRETE_F:
    markerline, stemlines, baseline = ax2.stem(F, YM)
    plt.setp(markerline, 'markerfacecolor', 'r', 'markersize', 8, 'marker', 'o')
    plt.setp(stemlines, 'color','b', 'linewidth', 2)
    plt.setp(baseline, 'linewidth', 0) # turn off baseline
else:
    plot(F, YM,'r',linewidth=3)
#ax2.set_ylabel(r'$|H(\mathrm{e}^{\mathrm{j} 2 \pi F})| \rightarrow$')
ax2.set_xlabel(r'$F \; \rightarrow$', size = 24)
ax2.xaxis.set_label_coords(0.98, 0.15)# transform=ax1.transData)
ax2.set_ylim(0, max(YM)*1.05)
fig2.tight_layout()
fig2.savefig(BASE_DIR + NAME + '_f.png')

#def resadjust(ax, xres=None, yres=None):
#    """
#    Send in an axis and I fix the resolution as desired.
#    """
#
#    if xres:
#        start, stop = ax.get_xlim()
#        ticks = np.arange(start, stop + xres, xres)
#        ax.set_xticks(ticks)
#    if yres:
#        start, stop = ax.get_ylim()
#        ticks = np.arange(start, stop + yres, yres)
#        ax.set_yticks(ticks)
#
#


plt.show()