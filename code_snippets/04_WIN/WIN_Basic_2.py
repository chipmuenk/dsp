# -*- coding: utf-8 -*-
"""
=== WIN_Basic.py =====================================================

Demonstrate how various parameters of the DFT like
 - window length
 - window type
 - sample rate
 influence the displayed spectrum

TODO: Code Cleanup

(c) 2016-Feb-04 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)
import matplotlib.ticker as plticker

mpl.rcParams['xtick.major.size'] = 5
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.major.size'] = 5
mpl.rcParams['ytick.major.width'] = 2

EXPORT =  True # exportiere Grafiken im Format FMT
#BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
BASE_DIR = "D:/Daten/HM/Tests/dsvFPGA/2016ss/"
FILENAME = "DFT_ML_" # "DFT" #
FMT = ".png"

SHOW_CFT = True # show continuous signal
SHOW_SAMPLED = False # show sampled signal
SHOW_ZERO = True # show zeros outside the window
SHOW_DFT = True # show DFT samples
SHOW_REP = True # show repeated measurement windows
SHOW_LABEL = False # show label with plot info in FIG 1 + 6
SHOW_WINDOW = True # window function as a semi-transparent area

SHOW_LOG = True

SHOW_FIG2_WINDOW = False  # Window function and its spectrum
SHOW_FIG3 = False # single-sided DFT
SHOW_FIG4 = False # two-sided DFT
SHOW_FIG4_SLIDE = False #two-sided DFT for slides
SHOW_FIG5_SLIDE = False # Window spectrum for presentation (large plot)
SHOW_FIG6_BW = True # B&W plots for skript, tests, ...

PLOT_EMPTY = False # plot empty coordinate diagram

TWINY = True

#------------------------------------------------
#Initialize sampling / DFT related variables
#------------------------------------------------
#fs = 5500, fsig = 550, NFFT = 50
#fs = 12000, fsig = 1000 + 2000, NFFT = 36
fs = 300000.0   # sampling frequency
Ts = 1.0/fs    # sampling period
OSR = 500      # "Oversampling Ratio" for displaying pseudo-analog signal
M_FFT  = 50    # DFT points
L = 3          # show L DFT windows of time signal

Tmeas = M_FFT * Ts # calculate meas. window of FFT
tstep = 1.0/(fs*OSR) # time step for plotting "analog" signal

#------------------------------------------------
#Initialize signal related variables
#------------------------------------------------
DC   = 0.0    # DC-Offset
fsig1 = 75000.0;  fsig2 = 30000 #24200  # analog signal frequencies,
A1   = 0.01;       A2 = 0.5          # amplitudes and
phi1 = pi/3;    phi2 = 0     # starting phases
Anoi = 0.0  # noise amplitude (uniform distribution)
Pnoi = 0.0 # noise power (gaussian distribution)
Noise_Floor = 3e-3 # constant "noise floor" for Figure 6
Xn_dB_min = -60 # minimum displayed dB value
Xn_dB_max = +5 # minimum displayed dB value
#-------------------------------------------------

mk_lg = 8  # markersize large for stemplots
mk_sm = 6  # markersize small for stemplots
#---------------------------------------------
# Window functions - select one!
#------------------------------------------------
wn = sig.windows.boxcar(M_FFT * OSR); myWindow ='Rechteck'
#wn = sig.windows.hann(M_FFT * OSR, sym = False); myWindow ='Hann'
#wn = sig.windows.hamming(M_FFT * OSR , sym = False); myWindow ='Hamming'
#wn = sig.windows.flattop(M_FFT * OSR, sym = False); myWindow ='Flattop'
#wn = sig.windows.bartlett(M_FFT * OSR, sym = True); myWindow ='Bartlett'

#
# Normalized Equivalent Noise Bandwidth of window:
ENBW = len(wn)*np.sum(wn**2)/ np.sum(abs(wn))**2
# Coherent Gain
CGain = np.sum(wn)/len(wn)
print('ENBW = ', ENBW)
print('CGain = ', CGain)
#------------------------------------------------
tw = linspace(0,Tmeas,M_FFT * OSR) # "analog" time
t = linspace(0,Tmeas*L, M_FFT * OSR *L)
#t = arange(0,Tmeas*L,tstep)
nw = arange(0,M_FFT)  # discrete samples
n = arange(0,M_FFT*L)
#---------------------------------------------
# calculate random signal
noise_t = Anoi * (np.random.rand(len(t)) - 0.5)
noise_t += Pnoi * (np.random.randn(len(t)))
# Calculate "analog" signals - select one
xt = DC + A1 * cos(2.0*pi*fsig1*t + phi1) \
    + A2 * cos(2.0*pi*fsig2*t + phi2) + noise_t
#xt = np.sign(xt) # create rectangular function

#---------------------------------------------

xtw = xt[M_FFT*OSR:2*M_FFT*OSR] * wn # Window the time signal

xtw = np.tile(xtw, L) # repeat "oversampled" data L times
xn = xtw[::OSR] # sample data by picking every OSR sample

#==============================================================================
#-------------------------------------------------------------
#    Figure 1: Plot windowed "analog" and sampled time signal
#-------------------------------------------------------------
# create new figure(1) if it does not exist, else make it active
# clear it and return a reference to it.
fig1 = figure(1)#, figsize = (9.5,2), dpi = 300)
fig1.clf() #
ax11 = fig1.add_subplot(111) # use object oriented matplotlib API
                             # for more flexibility
# original signal x as black line
if SHOW_CFT:
    ax11.plot(t[0:M_FFT*OSR*L], xt[0:M_FFT*OSR*L], 'k-', lw = 1)
#    ax11.plot(t[M_FFT*OSR:(2*M_FFT-1)*OSR], xtw[M_FFT*OSR:(2*M_FFT-1)*OSR], 'k-')

# M_FFT windowed samples (T_meas) of xtw as red line + stem plot
if SHOW_DFT:
    ax12 = ax11.twiny()         # two plots with same y- but different x-axes
    ml_b, sl_b, bl_b = ax12.stem(n[M_FFT:2*M_FFT], xn[M_FFT:2*M_FFT])
    plt.setp(ml_b, 'markerfacecolor', 'b', 'markersize', mk_lg,
             'markeredgecolor', 'r', 'markeredgewidth', 2) # markers
    plt.setp(sl_b, 'color','r', 'linewidth', 2) # stemline
    plt.setp(bl_b, 'linewidth',0) # turn off baseline
    #    ax12.set_xlabel(r'$n \rightarrow$')
    ax12.grid('off') # dont plot x-gridlines for second x-axis
    ax12.set_xticks([]) # turn off ticks

    if SHOW_LABEL:
        ax12.text(0.5,0.07,
             r'$f_S = %.1f$ Hz, $N_{FFT} = %d$, $f_{sig1} = %.1f$ Hz'
               %(fs,M_FFT, fsig1), fontsize=16,
             ha="center", va="center",linespacing=1.5, transform = ax12.transAxes,
             bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='0.9'))

    ax12.set_xlim([M_FFT/2, (L-0.5) * M_FFT]) # match range for second x-axis with first one
    ax12.set_ylim(-0.2 * max(xt), 1.15 * max(xt))    # set ylim to min/max of xtw
    # Draw x-Axis:
    ax12.axhline(0, xmin = 0, xmax = 1, linewidth=2, color='k')


# sampled signal x as blue stems
if SHOW_SAMPLED:
    ax11.stem(t[0:M_FFT*OSR*L:OSR], xt[0:M_FFT*OSR*L:OSR], 'b-', lw = 2)

# sampled and zeroed signal x outside window
if SHOW_ZERO:
    ax11.stem(t[0:M_FFT*OSR:OSR], zeros(M_FFT), 'r-')
    ax11.stem(t[2*M_FFT*OSR:M_FFT*OSR*L:OSR], zeros(M_FFT*(L-2)), 'r-')


# show repeated measurement windows in grey before and after actual plot
if L > 1 and SHOW_REP:
    ax11.plot(t[0:M_FFT*OSR], xtw[0:M_FFT*OSR],
              t[(2*M_FFT-1)*OSR:], xtw[(2*M_FFT-1)*OSR:], color='grey', linestyle='--')
    ml_g1, sl_g1, bl_g1 = ax12.stem(n[0:M_FFT], xn[0:M_FFT])
    ml_g2, sl_g2, bl_g2 = ax12.stem(n[2*M_FFT:], xn[2*M_FFT:])
    plt.setp(ml_g1, 'markerfacecolor', 'grey', 'markersize', mk_sm)
    plt.setp(sl_g1, 'color','grey', 'linewidth', 1, linestyle='-')
    plt.setp(bl_g1, 'linewidth',0) # turn off baseline
    plt.setp(ml_g2, 'markerfacecolor', 'grey', 'markersize', mk_sm)
    plt.setp(sl_g2, 'color','grey', 'linewidth', 1, linestyle='-')
    plt.setp(bl_g2, 'linewidth',0) # turn off baseline


if SHOW_WINDOW:
    # Plot window function as a semi-transparent area
    ax11.fill_between(t[M_FFT*OSR:(2*M_FFT - 1)*OSR], wn[:-OSR]*np.max(xt)*1.1, color = (0,0,0,0.2),
                  lw = 1.5, edgecolor = 'black', linestyle = '-')

ax11.set_xlabel(r'Zeit / s $\rightarrow$')
ax11.set_ylabel(r'Amplitude / V $\rightarrow$')
ax11.set_xlim([Tmeas/2, (L-0.5) * Tmeas])
#
fig1.tight_layout(pad = 0.5, rect=[0, 0, 0.99, 0.99]) # make room for title: rect=[0, 0, 1, 0.9]
#plt.tight_layout()
#fig1.text(0.5, 0.99, "Gefensterte Zeitfunktion", ha='center', va = 'top',
#         fontsize=18, transform = fig1.transFigure) # create title

if EXPORT:
    fig1.savefig(BASE_DIR + 'WIN_Basic-Time_%s' %int(M_FFT) + FMT)
#-------------------------------------------------------------
#     Figure 2: Plot Window Function and its DFT
#-------------------------------------------------------------
if SHOW_FIG2_WINDOW:
    Wn = fftshift(fft(wn[::OSR], M_FFT*8))/(M_FFT)
    wf = fftshift(arange(len(Wn)))
    wf = fftshift(fftfreq(M_FFT*8))
    #
    fig2 = figure(2); fig2.clf()
    ax21 = fig2.add_subplot(211); grid(True)
    ax21.set_title('Fensterfunktion im Zeit- und Frequenzbereich')
    ax21.stem(n[:M_FFT], wn[::OSR])
    ax21.text(0.5, 0.2, myWindow, fontsize=16,
             ha="center", va="bottom", transform = ax21.transAxes,
             bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='0.9'))

    ax21.set_xlabel(r'$n \, \rightarrow $')
    ax21.set_ylabel(r'$w[n] \; \rightarrow $')
    ax21.set_xlim(-0.5, M_FFT - 0.5)
    ax21.set_ylim(-0.1, 1.1)
    ax21.axhline(0, xmin = 0, xmax = 1, linewidth=1, color='k')

    ax22 = fig2.add_subplot(212); plt.grid(True)
    # Window with log scale
#    ax22.plot(wf, np.maximum(20 * log10(abs(Wn)), Xn_dB_min), lw = 2)
#    ax22.set_ylabel(r'$\| W(\mathrm{e}^{\mathrm{j} 2 \pi F})\| \mathrm{/ dB} \rightarrow $ ')
    # Window with linear scale
    ax22.plot(wf, abs(Wn), lw = 2, color = 'b')
    ax22.fill_between(wf, abs(Wn), alpha = 0.2, color='b')
    ax22.set_ylabel(r'$\| W(\mathrm{e}^{\mathrm{j} 2 \pi F})\| \rightarrow $ ')
    ax22.set_ylim(0,1)


    ax22.set_xlim(-0.5, 0.5)
    ax22.set_xlabel(r'$F \; \rightarrow $')
    ax22.text(0.97, 0.95, '$CGain = %0.3f$\n$ENBW = %0.3f$'%(CGain, ENBW), fontsize=16,
             ha="right", va="top", transform = ax22.transAxes,
             bbox=dict(alpha=0.9,boxstyle="round,pad=0.1", fc='0.9', lw = 0) )
    fig2.tight_layout()

#-------------------------------------------------------------
#    CALCULATE DFT in lin and log scale
#-------------------------------------------------------------

# Calculate two-sided DFT and scale it with 1/M_FFT:
Xn = fft(xn, n=M_FFT) / (M_FFT * CGain) / sqrt(2)
# zero-padded DFT with OSR * M_FFT samples as an approximation for DTFT:
X_DTFT = fft(xn[0:M_FFT], n = M_FFT * OSR) / (M_FFT * CGain)

# f = [0 ... f_S[ = [0... f_S/2[, [-f_S/2 ... 0[
xf  = fftfreq(M_FFT, Ts)
xf_DTFT = fftfreq(M_FFT*OSR, Ts)
xfn = fftfreq(M_FFT, d=1.0/M_FFT)
# Corresponding freq. points at [0... f_S/2[, [-f_S/2 ... 0[
#
# Calculate Xn in dBs with a fixed minimum
Xn_dB = np.maximum(20*log10(abs(Xn)),Xn_dB_min)
Xn1 = 2.*abs(Xn[0:M_FFT//2]) + Noise_Floor;
Xn1[0] = Xn1[0] / 2
Xn1_dB = np.maximum(20*log10(abs(Xn1)),Xn_dB_min)

#-------------------------------------------------------------
#    Figure 3: PLOT SINGLE-SIDED DFT
#-------------------------------------------------------------
if SHOW_FIG3:
    fig3 = plt.figure(3)
    fig3.clf()
    #==========================================================================
    ax31 = fig3.add_subplot(211)
    ax31.grid(True)
    plt.title('Einseitige DFT $S[k]$') # Overall title
    ml, sl, bl = ax31.stem(xf[:M_FFT//2],Xn1) # plot vs freq.
    ml, sl, bl = ax31.stem(xfn[:M_FFT//2],Xn1) # plot vs. n
    plt.setp(ml, 'markerfacecolor', 'r', 'markersize', mk_lg) # markerline
    plt.setp(sl, 'color','r', 'linewidth', 2) # stemline
    plt.setp(bl, 'linewidth',0) # turn off baseline
    plt.ylabel(r'$|S(f)|$' )
    ax31.set_ylim(0,1.1)
#    ax31.set_xlim(-fs/50,fs/2.)
    ax31.axhline(0, xmin = 0, xmax = 1, linewidth=1, color='k')
    ax31.axvline(x=0, ymin = 0, ymax = 1, linewidth=1, color='k')
    ax31.set_xlabel(r'$f \; \mathrm{[Hz]} \;\rightarrow $')
    #
    ax32 = fig3.add_subplot(212)
    ml, sl, bl = ax32.stem(xfn[:M_FFT//2],Xn1_dB[:M_FFT//2],bottom=Xn_dB_min)
    plt.setp(ml, 'markerfacecolor', 'r', 'markersize', mk_lg) # markerline
    plt.setp(sl, 'color','r', 'linewidth', 2) # stemline
    plt.setp(bl, 'color','k') # black baseline

    ax32.axhline(0, xmin = 0, xmax = 1, linewidth=1, color='k')
    ax32.axvline(x=0, ymin = 0, ymax = 1, linewidth=1, color='k')

    ax32.set_xlabel(r'$k \; \rightarrow $')
    ax32.set_ylabel(r'$20 \log |S[k]| / \mathrm{V} \, \rightarrow$' )
    #==============================================================================
    ax32.grid(True)
    fig3.tight_layout()

#-------------------------------------------------------------
#     Figure 6: B & W DFT for skript and tests
#-------------------------------------------------------------
if SHOW_FIG6_BW:
    fig6 = plt.figure(6)
    fig6.clf()
    ax61 = fig6.add_subplot(111)
    #plt.title('Einseitige DFT $S[k]$') # Overall title
    #ml, sl, bl = ax61.stem(xf[:M_FFT//2],Xn1) # plot vs freq.
    if SHOW_LOG: 
        if not PLOT_EMPTY:
            ml, sl, bl = ax61.stem(xfn[:M_FFT//2],Xn1_dB, bottom = Xn_dB_min) # plot vs. n
        ax61.set_ylabel(r'$|S[k]|$ in dBW $\rightarrow$' )
        ax61.set_ylim(Xn_dB_min,Xn_dB_max)
    else:
        if not PLOT_EMPTY:
            ml, sl, bl = ax61.stem(xfn[:M_FFT//2],Xn1) # plot vs. n
        ax61.set_ylabel(r'$|S[k]|$ in V $\rightarrow$' )
        ax61.set_ylim(0,1.2)
    
    if not PLOT_EMPTY:
        plt.setp(ml, 'markerfacecolor', 'k', 'markersize', mk_lg) # markerline
        plt.setp(sl, 'color','k', 'linewidth', 2) # stemline
        plt.setp(bl, 'linewidth',0) # turn off baseline
    
    ax61.set_xlim(-M_FFT/100.,M_FFT/2.)
    minor_ticks = np.arange(0, M_FFT/2, 5)
    ax61.axhline(0, xmin = 0, xmax = 1, linewidth=1, color='k')
    ax61.axvline(x=0, ymin = 0, ymax = 1, linewidth=1, color='k')
    ax61.set_xlabel(r'$k \;\rightarrow $')
    if SHOW_LABEL:
        ax61.text(0.5,0.9,
         r'$f_S = %.1f$ kHz, $N_{FFT} = %d$, $f_{sig1} = %.1f$ kHz, $f_{sig2} = %.1f$ kHz'
           %(fs/1000,M_FFT, fsig1/1000, fsig2/1000), fontsize=16,
         ha="center", va="center",linespacing=1.5, transform = ax61.transAxes,
         bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='0.9'))
    ax61.grid(which='both')
    ax61.tick_params(which = 'both', direction = 'out')
    minor_ticksx = np.arange(0, M_FFT/2, 1)
    major_ticksx = np.arange(0, M_FFT/2, 5)
    loc = plticker.MultipleLocator(base=5.0) # this locator puts ticks at regular intervals
    ax61.yaxis.set_minor_locator(loc)
    ax61.set_xticks(major_ticksx)
    ax61.set_xticks(minor_ticksx, minor=True)
    ax61.grid(which='minor', alpha=0.5)
    ax61.grid(which='major', alpha=1)
    
    fig6.tight_layout()
    if EXPORT:
        fig6.savefig(BASE_DIR + FILENAME + '_bw_%s' %(M_FFT) + FMT)

#-------------------------------------------------------------
#     Figure 5: Plot Window DFT for slides
#-------------------------------------------------------------
if SHOW_FIG5_SLIDE:
    # single plot for slides
    fig5 = figure(5, figsize = (9,2), dpi = 300)
    ax5 = fig5.add_subplot(111); grid(True)
    # Window with linear scale
    ax5.plot(wf, abs(Wn), lw = 2, color = 'b')
    ax5.fill_between(wf, abs(Wn), alpha = 0.2, color='b')
    ax5.set_ylabel(r'$\| W(\mathrm{e}^{\mathrm{j} 2 \pi F})\| \rightarrow $ ')
    ax5.set_ylim(0,1)


    ax5.set_xlim(-0.5, 0.5)
    ax5.set_xlabel(r'$F \; \rightarrow $')
#    ax5.text(0.97, 0.95, '$CGain = %0.3f$\n$ENBW = %0.3f$'%(CGain, ENBW), fontsize=16,
#             ha="right", va="top", transform = ax22.transAxes,
#             bbox=dict(alpha=0.9,boxstyle="round,pad=0.1", fc='0.9', lw = 0) )
#    fig5.tight_layout()
    fig5.tight_layout(pad = 0.3, rect=[0.03, 0, 1, 1])


#-------------------------------------------------------------
#    Figure 4: CALCULATE AND PLOT TWO-SIDED DFT
#-------------------------------------------------------------
if SHOW_FIG4:
    if SHOW_FIG4_SLIDE:
        fig4 = figure(4, figsize = (5,4), dpi = 300)
    else:
        fig4 = figure(4)
    fig4.clf() #

    # fftshift centers freq. vector to f = [-f_S/2... f_S/2[
    Xn = fftshift(Xn)
    X_DTFT = fftshift(X_DTFT)
    # set Xn = 0 for very small magnitudes to eliminate
    # numeric errors in phase calculation:
    #Xn = np.real((abs(Xn/max(abs(Xn))) > 1.0e-10)) * Xn
    # - doesn't work - result is 0j instead of 0!

    for i in range(len(Xn)): #Xn = Xn * (abs(Xn[i]/max(abs(Xn))) < 1.0e-8)
        # set phase = 0 for very small magnitudes (numeric errors)
        if abs(Xn[i]/max(abs(Xn))) < 1.0e-8: Xn[i] = 0

    xf = fftshift(xf) # Center frequency vector
    xfn = fftshift(xfn) # Center frequency vector
    xf_DTFT = fftshift(xf_DTFT)

    ax41 = fig4.add_subplot(111)
    ax42 = ax41.twiny()  # two plots with same y- but different x-axes
    ax42.grid(False) # plot x-gridlines for second x-axis
    # Draw x- and y-Axis:
    ax42.axhline(0, linewidth=2, color='k')
    ax42.axvline(0, linewidth=1, color='k')

    cft = [[-fsig2, -fsig1, 0, fsig1, fsig2], [A2/2, A1/2, DC, A1/2, A2/2]]
#    cft = [[-fsig1, 0, fsig1], [A1/2, DC, A1/2]]
    m_cft,s_cft,b_cft = ax41.stem(cft[0], cft[1], markerfmt = 'k^', label = 'CFT')
    plt.setp(m_cft, 'markersize', 15) # marker
    plt.setp(s_cft, 'color','k', 'linewidth', 4) # stemline
    plt.setp(b_cft, 'linewidth', 0) # baseline

    ml, sl, bl = ax41.stem(xf,abs(Xn),'r', label = 'DFT')
    plt.setp(ml, 'markerfacecolor', 'r', 'markersize', mk_lg) # markerline
    plt.setp(sl, 'color','r', 'linewidth', 2) # stemline
    plt.setp(bl, 'linewidth',0) # turn off baseline
    ax41.plot(xf_DTFT, abs(X_DTFT), 'b', lw = 2, label = 'DTFT')

#  Pfeile (= Annotation ohne Text) mit den erwarteten Frequenzen und Amplituden:
#    sigs = ((fsig1, A1/2), (-fsig1, A1/2), (fsig2, A2/2), (-fsig2, A2/2), (0, DC))
#    for vec in sigs:
#        ax41.annotate('', xy=(vec[0], vec[1]), xytext=(vec[0], 0),
#    #    arrowprops=dict(facecolor='black',arrowstyle = '->', alpha = 0.4))
#        arrowprops=dict(facecolor='black', headwidth=15, frac = 0.2, alpha = 0.4))
    #ax41.grid('on')
   # min(lim_eps(xf,-0.05))
#    ax41.text(0, -0.1,
#        ('$f_S = %.1f\, \mathrm{Hz},\, N_{FFT} = %d, \, \Delta f = %.1f \, \mathrm{Hz},\, f_{sig1} = %.1f \, \mathrm{Hz}$' %(fs, M_FFT,fs/M_FFT, fsig1)),
#        fontsize=16, ha="center", va="center",linespacing=1.5,
#        bbox=dict(alpha=1,boxstyle="round,pad=0.1", fc='0.9'))
    ax41.text(0, -0.1,
        ('$f_S = %.0f\, \mathrm{Hz},\, N_{FFT} = %d, \, \Delta f = %.0f \, \mathrm{Hz}$' %(fs, M_FFT,fs/M_FFT)),
        fontsize=16, ha="center", va="center",linespacing=1.5,
        bbox=dict(alpha=1,boxstyle="round,pad=0.1", fc='0.9'))
    ax41.set_xlim(-fs/2, fs/2)
    ax42.set_xlim(-M_FFT/2, M_FFT/2) # match range for second x-axis with first one
    ax42.set_ylim(-0.2, 1.2)
    #plt.title('Zweiseitige CFT, DTFT und DFT $S[k]$') # Overall title
    ax41.set_xlabel(r'$f \;\mathrm{[Hz]} \; \rightarrow$')
    ax41.set_ylabel(r'$|S[k]| / V \rightarrow$' )
    ax42.set_xlabel(r'$k \rightarrow$')
    ax41.grid(True)
    ax41.legend(fontsize=16)

    fig4.tight_layout(pad = 0.5, rect=[0,0, 1, 1]) # make room for title
#    fig4.text(0.5, 0.99, 'Zweiseitige CFT, DTFT und DFT von $S[k]$',ha='center', va = 'top',
#             fontsize=18, transform = fig4.transFigure)
    if EXPORT:
        fig4.savefig(BASE_DIR + 'WIN_Basic-Noi_DFT%s' %int(M_FFT) + FMT)
    #ax41.set_title('Zweiseitige DFT $S[k]$') # Overall title
    #
    #ax42 = fig4.add_subplot(212)
    #ax42.stem(xf,angle(Xn)/pi)
    #ax42.set_ylabel(r'$\angle S[k] / \mathrm{rad} /\pi \rightarrow$' )
    ##ax42.set_ylim(lim_eps(Xn_dB, 0.1))

    ##ax42.set_xlim(lim_eps(xfn, 0.05))
    #ax42.set_xlabel(r'$n \; \rightarrow$')

    #plt.tight_layout()

    #plt.grid('on')
    #plt.savefig('D:/Daten/ueb-DFT_Basics_1-ML_DFT%s.png' %int(M_FFT))
    #==============================================================================
    



plt.show()