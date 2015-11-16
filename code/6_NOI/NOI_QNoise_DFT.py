# -*- coding: utf-8 -*-
#################################################################
#
# NOI_QNoise_Filt.py
#
# Simulate quantization effects in the frequency domain! 
# A sinusoidal signal with fsig = f_S * N_per / N_FFT is being quantized.
# N_per is the number of signal periods within the DFT window (N_FFT data points).
# This way, the DFT is always calculated for an integer number of periods
# (coherent DFT).
#
# Try changing the following parameters:
# - N_per: The number of signal periods in the measurement window. Try to 
#    explain, why choosing 31 or 32 periods makes a huge difference
#
#
# C. Muenker, 2015
#
###########################################################################
from __future__ import division, print_function, absolute_import

import os
import numpy as np
from numpy import sin, cos, pi, array, arange, log10, zeros
#os.environ['MPLCONFIGDIR']='C:\\Users\\Muenker\\.matplotlib'

import matplotlib.pyplot as plt
import matplotlib

import dsp_fpga_fix_lib as fx

print(matplotlib.matplotlib_fname())
print(matplotlib.get_configdir())
#print(plt.style.available)
#plt.style.use(['default'])

basedir = os.path.dirname(os.path.abspath(__file__))
print(basedir)
#plt.style.use(os.path.join(basedir,'..','presentation_style.mplstyle'))
#plt.style.use('presentation_style.mplstyle')

def fold(num):
    """
    Bound the frequency index in a DFT to 0 ... N_FFT/2. Indices outside of 
    this range are folded back to this range in agreement with the sampling
    theorem. 
    """
    return abs((num + N_FFT/2) % N_FFT - N_FFT/2)
    
ANN_TITLE = False # print title
ANN_FREQ  = True # annotate signal frequency
ANN_SFDR  = True # annotate spurious free dynamic range
ANN_SNR   = False # annotate SNR
ANN_NOISE = True # annotage noise power
PLT_N_AVG = True # plot a line with average quantization noise power
PLT_HARMONICS = True # plot signal and harmonics as red lines

# Definiere Anzahl Samples, FFT - Länge, Samplingfrequenz etc.
N_FFT = 2048    # number of DFT data points
N_FFT_min = 1   # First frequency point to be evaluated: Biased quantization 
                # methods like floor (= truncation) produce a systematic DC-offset.bei Quantisierungsarten, die nicht mittelwertfrei sind ( z.B. 
                # floor = Truncation) werden aufgrund dieses Offets schlechtere
                # SNR-Werte berechnet. 
f_S = 500       # sampling frequency (for scaling the frequency axis)


Hmin = -140 # Min. y-Wert (dB) für Anzeige und Ersetzen von -infinity Werten
Hmax = 0    # Max. y-Wert (dB) für Anzeige
#########################################################################
# Parameters for a test signal allowing a coherent DFT without leakage
N_per = 61 # Number of signal periods in the DFT window
            # N_per should be less than N_FFT / 2 (otherwise there is aliasing)
            # and it should be a prime number to avoid periodicities of the 
            # quantization error resulting in a line spectrum
a_sig = 0.999  # Signal amplitude
k2 = 0   # 2nd order distortion, sin^2(x) = ½[1 – cos(2x)] -> k2/2 @ 2f and DC
k3 = 0   # 3rd order distortion, sin^3(x) = (3*sin(x) - sin(3*x))/4 -> k3/4 @ 3f

#----------------------------------------------------------------------------
# Initialize arrays to speed up processing
x  = zeros(N_FFT) # input signal
yq = zeros(N_FFT) # quantized signal
#########################################################################
Ts = 1 / f_S    # sampling period
t = arange(N_FFT) * Ts # define time vector 0 ... (N_FFT-1) * Ts
fsig = f_S * N_per / N_FFT # calculate signal frequency from N_per
x = a_sig * sin(2*pi*t*fsig+1) # test signal with some starting phase
#x = a_sig * sin(2*pi*t*fsig+1) + 0.0001*sin(2*pi*t*2*fsig) #Two-Tone test signal
##################################################################
# Distort signal with second and third order non-linearities
y = x + k2 * x * x - k3 * x**3
N_per = fold(N_per)
f_sig_fold = N_per / N_FFT * f_S
N_per2 = fold(N_per * 2)
f_sig2_fold = N_per2 / N_FFT * f_S
N_per3 = fold(N_per * 3)
f_sig3_fold = N_per3 / N_FFT * f_S
##################################################################
# Define and instantiate quantizer objects
q_adc = {'QI':0, 'QF':11, 'quant':'round', 'ovfl':'sat'}
adc      = fx.Fixed(q_adc) # adc quantizer instance

if adc.N_over > 0:
    print ("ADC overflows:", adc.N_over) 

##################################################################
### Input Quantization (ADC, q_adc)
### of signal + uniform dithering noise, -1/4 LSB < eps_N < 1/4 LSB

yq = adc.fix(y) # no dithering
#yq = adc.fix(y + adc.LSB/4. * np.random.rand(len(y)))

if adc.N_over: print('Overflows in ADC:  ', adc.N_over)

#################################################################
#
# Magnitude of DFT over f = 0 ... f_S/2, scaled mit 2 / N_FFT / sqrt(2)
#
# This yields correct scaling for:
# - SINGLE-SIDED spectra (0 ... f_S/2 ) : factor of 2
# - PERIODIC signalx (factor 1/N_FFT)
# - RMS values (factor 1 / sqrt(2) ) 
#-----------------------------------------------------------------
#Yq  = np.ones(N_FFT) * 1e-6 # create dummy test data
Yq  = 2 * np.abs(np.fft.fft(yq, N_FFT))[0:N_FFT/2] / N_FFT / np.sqrt(2)
Yq[0] /= np.sqrt(2.) # remove rms scaling + factor 2 for DC component
Yq_dB = np.maximum(20 * log10(Yq), Hmin) # replace log(0) = -inf by Hmin

Yqc = Yq.copy() # create a copy for calculating average noise power etc.
f   = np.fft.fftfreq(N_FFT, d=Ts) # generate vector with frequency points
#-----------------------------------------------------------------

Yq_sig = Yq[N_per]        # rms voltage of quantized test signal
Yq_sig_dB = 20 * log10(Yq_sig) #       "   in dBVrms


Yq_dc = Yq[0]        # DC-Value of quantized signal 
Yq_dc_dB = 20 * log10(Yq_dc)   # " in dB

PSigQ = Yq_sig ** 2      # power of quantized test signal, calculated from DFT
PSig = a_sig ** 2 / 2    # power of test signal, calculated from its amplitude
N_Q = adc.LSB ** 2 / 12  # total noise power, calculated from LSB size

# set DFT values = 0 for DC (up to N_FFT_min) and signal for finding the highest
# spurious line and the corresponding spurious free dynamic range (SFDR).
# Calculate the SIgnal to Noise And Distortion (SINAD) ratio and the 
# Total Harmonic Distortions (THD)
Yqc[N_per] = 0
N = N_FFT/2 - 1 # subtract one frequency point for eliminating the signal bin
for i in arange(N_FFT_min):
    Yqc[i] = 0
    N -= 1  

SINAD = 10 * log10(PSigQ / np.sum(Yqc[0:N_FFT/2] ** 2)) # SINAD
THD = 100 * (Yqc[N_per2] ** 2 + Yqc[N_per3] ** 2) / PSigQ # THD in %
    
Yq_max_spur_dB = 20*log10(np.max(Yqc))
SFDR = Yq_sig_dB - Yq_max_spur_dB
f_max_spur = np.argmax(Yqc)/ N_FFT * f_S

# Additionally, set second and third harmonic to zero for calculating average 
# noise power per bin for the display. The value is 
# used for filling the zeroed-out data points with useful data.

if k2 != 0: 
    Yqc[N_per2] = 0
    N -= 1
if k3 != 0:
    Yqc[N_per3] = 0
    N -= 1
 
e_N = np.sum(Yqc[0:N_FFT/2] ** 2) # total noise power from the remaining N bins
e_N_avg = e_N / N                  # average noise power per bin
e_N_avg_dB = 10*log10(e_N_avg)
Aq_avg = np.sqrt(e_N_avg)         # average rms noise voltage
e_N = e_N_avg * N_FFT/2   # total noise power from DFT without distortions and DC,
                        # calculated over all N_FFT/2 bins
# e_N = np.inner(Yqc[0:N_FFT/2], Yqc[0:N_FFT/2]) # same, using scalar product
e_N_dens = 2 * e_N / f_S # single-sided noise power density, calculated from total noise power
    
#------------------------------------------------------------------


SNR = 10 * log10(PSigQ / e_N) 
ENOB = (SNR - 1.7609)/6.0206
print("a_sig   = %2.4e V (signal amplitude)" %a_sig)
print("PSig    = %2.4e W (signal power)" %PSig)
print("Yq_sig  = %2.4e V (rms signal value from DFT)" %Yq_sig)
print("PSigQ   = %2.4e W (signal power from DFT)" %PSigQ)
print("Yq_dc   = %2.4e V (DC level from DFT)" %Yq_dc)

print("\nN_Q   = %2.4e W (total noise power from LSB)" %N_Q)
print("e_N     = %2.4e W (total noise power from DFT)" %e_N)
print("e_N_avg = %2.4e W/bin (avg. noise power / bin from DFT)" %e_N_avg)
print("e_N_dns = %2.4e W/Hz (avg. noise power density from DFT)" %e_N_dens)


print("\nTHD     = %2.2e %%" %THD)
print("SINAD   = %2.2f dB" %SINAD)
print("SNR     = %2.2f dB" %SNR)
print("ENOB    = %2.2f bits" %ENOB)

###############################################################################
#
# Plots
#
###############################################################################

fig4 = plt.figure(1)
fig4.clf()
ax4 = fig4.add_subplot(111)
ax4.set_xlim([0, f_S/2])
ax4.set_ylim([Hmin, Hmax])

# ------------- plot quantization noise in dB ----------------------------------
ax4.plot(f[0:N_FFT/2], Yq_dB[0:N_FFT/2],'k',linewidth = 2, label = r'$N_Q(f)$')
if PLT_HARMONICS:
    plt_params = {'lw':4, 'alpha':0.5, 'color':'r'}
    ax4.plot([f_sig_fold, f_sig_fold], [Hmin, Yq_dB[N_per]], **plt_params)
    if k2 != 0:
        ax4.plot([f_sig2_fold, f_sig2_fold], [Hmin, Yq_dB[N_per2]], **plt_params)
    if k3 != 0:
        ax4.plot([f_sig3_fold, f_sig3_fold], [Hmin, Yq_dB[N_per3]], **plt_params)

# ------------- plot a line with average quantization noise power ----------
if PLT_N_AVG: 
    ax4.plot([0, f_S/2] , [e_N_avg_dB, e_N_avg_dB], color='lightblue', linewidth = 2,
        linestyle = '-')

# ------------- find and annotate the highest spurious spectral line ----------
if ANN_SFDR:
    ax4.annotate('', (f_max_spur, Yq_max_spur_dB),(f_max_spur, Yq_sig_dB),
        xycoords='data', ha="center", va="center", size=24,
        arrowprops=dict(arrowstyle="<|-|>", facecolor = 'red', edgecolor='black', 
                        lw = 1, shrinkA = 0, shrinkB = 3)) #see matplotlib.patches.ArrowStyle
            
    ax4.text(f_max_spur, (Yq_max_spur_dB + Yq_sig_dB)/2, '$SFDR = %.1f$ dB' %(SFDR),
        fontsize=18, ha="center", va="center",linespacing=1.5, rotation = 90,
        bbox=dict(alpha=0.9,boxstyle="round,pad=0.2", fc='1', lw = 0))    

# ------------- annotate the signal frequency ----------
if ANN_FREQ:
    info_str3 = r'$f_{{sig}} = ${0:.0f} / {1:.0f} $f_S$'.format(N_per, N_FFT)
    ax4.text(0.97, 0.97, info_str3, ha='right', va='top', transform = ax4.transAxes, 
        fontsize = 18, bbox=dict(facecolor='0.9', alpha=0.9, boxstyle="round, pad=0.2"))

info_str = ""
# ------------- annotate the SNR and ENOB ----------
if ANN_SNR:
    info_str = r'SNR = {0:.2f} dB, ENOB = {1:.2f} bits'.format(SNR, ENOB)

# ------------- annotate the noise power ----------
if ANN_NOISE:
    if not info_str == "":
        info_str += "\n"
    info_str += r'$e_N =$ {0:.2f} dBW $\equiv$ {1:.2f} dBW/bin'.format(10*log10(e_N), e_N_avg_dB, )

ax4.text(0.97, 0.02, info_str, ha='right', va='bottom', transform = ax4.transAxes,
    bbox=dict(facecolor='0.9', alpha=0.9, boxstyle="round, pad=0.2"))

ax4.set_xlabel(r'$f$ / Hz $\rightarrow$')
ax4.set_ylabel(r'$\mathbf{Linienspektren:}\; S_{q}(f) \, \mathrm{in\;dBW} \;\rightarrow$')
if ANN_TITLE:
    ax4.set_title('DFT des quantisierten Signals $(N_{DFT} = %d)$' %(N_FFT))
#ax4.legend(loc = 'best')

ax4.grid(True)

# Create second y-axis to correctly read the power of wideband signals
ax4b = ax4.twinx()
ax4b.set_ylim(ax4.get_ylim()+ 10*log10(N_FFT/2))
ax4b.set_ylabel(r'$\mathbf{Breitbandspektren:}\; N_q(f)\;  \mathrm{in \; dBW} \; \rightarrow$')

plt.tight_layout()
plt.show()
