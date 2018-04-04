# -*- coding: utf-8 -*-
"""
Plot the spectrum / spectral leakage of window functions
Requires python 3.x with numpy, scipy and matplotlib modules
Christian Muenker under CC0 license
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal.windows as win
arrow = {'ec':'black','lw':2, 'arrowstyle':'<->'} # 'edge color', 'linewidth'
line_sl = {'ec':'green', 'lw':2, 'arrowstyle':'-', 'connectionstyle':'arc3, rad=-0.1'}
box = {'boxstyle':'round', 'ec':'white', 'fc':(1., 1., 1., 0.5)} # face color white with alpha

## User edits go here #################
N = 256             # window length
win_name = "Rectangular"   # window name for annotation
w = np.ones(N)      # rectangular window
#w = win.hann(N)    # comment out for hann window
xmin = -4.5
xmax = 9.4          # plotting limits for x-axis
dr = -80            # dynamic range: lower limit in dB for plotting
sl1 = 2.4           # bin of first sidelobe
#######################################
k = np.arange(N) # index 0 ... N-1
w = np.concatenate((w, np.zeros(7*N)), axis=0) # append zeros to window
sl1_idx = int(sl1 * 8) # index for first sidelobe
## Calculate the FFT ###################
H = np.abs(np.fft.fft(w))
H = np.fft.fftshift(H)  # center spectrum around 0
H = H / np.max(H)       # normalize to 1
H = 20 * np.log10(H)    # convert to dBs
np.clip(H, dr, 0, H)    # clip to the range dr ... 0
x = np.arange(xmin*8, xmax*8, dtype=int) # index from xmin ... xmax
H = H[x + 4*N]          # zoom in to the part from xmin ... xmax
H0_idx = int(-xmin*8)   # index for bin 0
## Create the plot #####################
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.fill_between(x/8, dr, H, facecolor='green', alpha=0.5) # W(k)
plt.grid(True)
## Annotations #########################
ax1.axhline(0, color='k', linestyle='--') # 0 dB line
ax1.text(0, dr/3, "Main Lobe", size=20, rotation=90.,
         ha="center", va="center", bbox = box)
ax1.annotate("", xy=(sl1, H[H0_idx + sl1_idx]), xytext=(sl1, 0),
            arrowprops=arrow, va = 'top', ha = 'right') # arrow to max. side lobe
ax1.annotate("Max. Side\nLobe Level", xy=(sl1 + 1./4,  H[H0_idx + sl1_idx]/2),
             ha='left', va='center')
ax1.annotate("", xy=(sl1 + 1/4, H[H0_idx + sl1_idx + 2]+2), xycoords='data',
            xytext=(xmax, H[-1]+2), textcoords='data', arrowprops=line_sl)
ax1.text((sl1 + xmax)/2, (H[H0_idx + sl1_idx] + H[-1])/2, "Side Lobes", size=18, 
         ha="left", va="bottom", color='green')
ax1.text(0, 0.9 *dr, "{0} window".format(win_name), bbox=box, ha='center', size=20)
ax1.set_xlabel('DFT bins')
ax1.set_ylabel(r'$W(k)$ in dB')
ax1.set_title('Spectral leakage from a sinusoidal signal')
#########################################
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([dr, 2])
plt.show() # now display the plot