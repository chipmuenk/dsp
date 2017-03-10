# -*- coding: utf-8 -*-
"""
==== LTI_periodizitaet.py =================================================

Zeige und berechne Periodizitaet von abgetasteten Signalen

(c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, zeros, ones)

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)
    
EXPORT = False          
BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
# BASE_DIR = "D:/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
FILENAME = "LTI_periodizitaet" 
FMT = ".svg"

# This Greatest Common FRACTION function also works with fractions
def gcf(a, b):
    while b != 0:
        a, b = b, a % b # a = b, then b = a % b (remainder)
    return a 
    
# Return min / max of an array a, expanded by eps*(max(a) - min(a))
def lim_eps(a,eps):
    mylim = (min(a) -(max(a) - min(a))*eps, max(a) + (max(a)-min(a))*eps)
    return mylim  

#Initialize variables
#------------------------------------------------
fs = pi * 50 #240.0    # sampling frequency
Ts = 1.0/fs      # sampling period
N_man  = 0     # manual selection of N if N_man > 0 

fsig = 50.0    # base signal frequency
fsig2 = 60.0  # additional harmonic or component
DC   = 1.5    # DC-level (offset for cos-Signal)
A1   = 0.5   # Amplitude
A2   = 0.0
phi1 = 0.0 #-pi/2 + 0.01 # starting phase
phi2 = 0.0
#-------------------------------------------------
tstep = 1.0/(fsig*50) # time step for plotting "analog" signal
N = Nmax = int(fs / gcf(fs,fsig)) # Number of samples in one "digital" period
L = Lmax = int(fsig / gcf(fs,fsig)) # Number of "analog" periods
print('N =',N)
print('gcf(fs,fsig) = ', gcf(fs,fsig))
print('L =', L)
disp_L = True # highlight periodicity of anlog signal if possible
if N_man > 0:
    N = N_man
    L = int(fs/fsig)
if L > 20: # Too many periods for analog plot
    L = L_per = 10
    N = int(L / (fsig * Ts))
    disp_L = 'false'
else:
   L_per = L+1  # display number of periods of analog signal

Tmax = L_per/fsig # calculate timespan from L_per
N_Ts = Tmax / Ts # number of samples in Tmax

#------------------------------------------------

# numpy.arange(start,stop,step): create linear spaced arrays (vectors)
# alternative: numpy.linspace(start,stop,numsteps,endpoint=true)      
t = arange(0,Tmax,tstep) # "analog" time                            
n = arange(0,int(N_Ts),1) # discrete number of samples
# calculate "analog" signal:
xt = DC + A1 * cos(2.0*pi*fsig*t + phi1) + A2 * cos(2.0*pi*fsig2*t + phi2)
# calculate sampled signal
xn = DC + A1 * cos(2.0*pi*fsig*n*Ts + phi1) + A2 * cos(2.0*pi*fsig2*n*Ts + phi2)

#xt = DC + A1 * sgn(cos(2.0*pi*fsig*t + phi1)) 
#xn = DC + A1 * sgn(cos(2.0*pi*fsig*n*Ts + phi1))
#xt = np.sign(xt) # rect function
#xn = np.sign(xn) # rect function

# create new figure(1) if it does not exist, else make it active 
# and return a reference to it:
fig = figure(1); clf()
ax1 = fig.add_subplot(111) 
# plot x for L periods using blue line:
ax1.plot(t[0:L*50+1], xt[0:L*50+1], 'b-', label = '$x(t)$') 
ax1.plot(t[L*50:], xt[L*50:], color='grey', linestyle='-')   # plot rest of x 
ax1.set_xlabel(r'$t$ / s $\rightarrow$')
ylabel(r'$x(t), \, x[n]$ / V $\rightarrow$')
grid(axis='x') # plot y-gridlines for both x-axes
xlim([0,Tmax])
ax2 = ax1.twiny() # make two plots with same y- but different x-axes
markerline, stemlines, baseline = ax2.stem(n[0:N], xn[0:N], label = '$x[n]$') 
plt.setp(markerline, 'markerfacecolor', 'r', 'markersize', 10)
plt.setp(stemlines, 'color','r', 'linewidth', 2)
plt.setp(baseline, 'linewidth', 0) # turn off baseline
if disp_L==True:
    markerline, stemlines, baseline = ax2.stem(n[N:], xn[N:]) 
    plt.setp(markerline, 'markerfacecolor', 'grey', 'markersize', 8)
    plt.setp(stemlines, 'color','grey', 'linewidth', 1, linestyle='-')
    plt.setp(baseline, 'linewidth', 0) # turn off baseline
ax2.set_xlabel(r'$n \, \rightarrow$')
ax2.grid(True) # plot x-gridlines for second x-axis
xlim([0,N_Ts]) # match range for second x-axis with first one
ylbl = min(xt) + (max(xt) - min(xt)) * 0.3 # position for label
ax2.text(0.97, 0.04, '$f_S = %.1f$ Hz, $f_{sig} = %.1f$ Hz' %(fs,fsig),
         fontsize=18, ha="right", va="bottom",linespacing=1.5,
         transform = ax2.transAxes, # coordinates relativ to axis
         bbox=dict(alpha=0.9,boxstyle="round,pad=0.1", fc='0.9'))
if disp_L==True:
    # double-headed arrow for period length using annotate command with empty text:
    ax2.annotate('', (0, ylbl),(N, ylbl), xycoords='data', ha="center", va="center", size=18,
        arrowprops=dict(arrowstyle="<|-|>", facecolor = 'red', edgecolor='black' ))
        #see matplotlib.patches.ArrowStyle
    plt.axvline(x=N, linewidth=2, color='k', linestyle='--') 
    #
# textbox with values for N, L
ax2.text((N)/2.0,ylbl,'$N = %s, \, L = %s$' %(Nmax, Lmax), fontsize=18,ha="center",
         va="center", linespacing=1.5, bbox=dict(boxstyle="square", fc='white'))
              #    xycoords="figure fraction", textcoords="figure fraction")                         
ylim(lim_eps(xt,0.05))    # set ylim to min/max of xt
# Draw a horizontal line at y=0 from xmin to xmax (rel. coordinates):
plt.axhline() 
plt.text(0.5, 1.14, 'Kohärente Abtastung mit $N$ Samples über $L$ Perioden',
         horizontalalignment='center',
         fontsize=20,
         transform = ax2.transAxes)
plt.subplots_adjust(top=0.85,right=0.95)
if EXPORT:
    plt.savefig(BASE_DIR + FILENAME +"_%sHz" %int(fs) + FMT)


plt.show()