# -*- coding: utf-8 -*-
#################################################################
#
# NOI_DFT.py
#
# Darstellung von Signal und Quantisierungsrauschen mit DFT 
# Korrekte Skalierung von Signal und Breitbandrauschen
#
# C. Muenker, 27-5-2015
#
#
###########################################################################
from __future__ import division, print_function
import numpy as np
from numpy import sin, cos, pi, array, arange, log10, zeros, tan, sqrt
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.pyplot import plot, grid

import dsp_fpga_fix_lib as fx
#########################################################################
# Definiere Anzahl Samples, FFT - Länge, Samplingfrequenz etc.
N_FFT = 2048    # Anzahl Datenpunkte für FFT 
TWIN_AX = False # Drucke zweite Achse für Rauschleistung
PRINT_AVG = True
PRINT_SNR = False # Drucke SNR und ENOB
PRINT_NOISE = False # Drucke Rauschleistung

DITHER = True

f_S = 2048       # Abtastfrequenz (eigentlich beliebig, ergibt Skalierungsfaktor)
f_S_Print = 50   # Abtastfrequenz für Plot
unit = ""
Ts = 1 / f_S    # Abtastperiode
t = arange(N_FFT) * Ts # Zeitvektor 0 ... (Nx-1) Ts
#f = [0:1:L-1] / N_FFT * f_S # Frequenzvektor 0 ... f_S in N_FFT Schritten
#f = f_S/2*np.linspace(0,1,N_FFT/2) # Frequenzvektor 0 ... f_S/2 in N_FFT/2 Schritten
Hmin = -150 # Min. y-Wert (dB) für Anzeige und Ersetzen von -infinity Werten
Hmax = 2    # Max. y-Wert (dB) für Anzeige
#########################################################################
# Vorbelegen der Vektoren mit Nullen zur Initialisierung und Beschleunigung
x  = zeros(N_FFT) # Eingangssignal
xq = zeros(N_FFT) #   "  (quantisiert durch ADC)

#########################################################################
# Definiere Testsignal für eine kohärente DFT ohne Leckeffekt 
N_per = 53 # Anzahl der Perioden des Signals im DFT-Rechteckfenster
            # N_per muss kleiner als N_FFT / 2 sein (sonst gibt's Aliasing!)
            # und am besten eine Primzahl, z.B. 63, 131, 251, 509, 1021, ...
            # N_per entspricht außerdem der Signalfrequenz!
a_sig = 1  # Signalamplitude
fsig = f_S * N_per / N_FFT
x = a_sig * sin(2*pi*t*fsig+1)# + 2^-18 #Testsignal mit Startphase
#x = a_sig * sin(2*pi*t*fsig+1) + 0.001*sin(2*pi*t*251) #Two-Tone Testsignal

#########################################################################
# Definiere Quantizer-Objekte:
# 

q_adc = {'QI':1, 'QF':15, 'quant':'round', 'ovfl': 'sat'}

fx_adc = fx.Fixed(q_adc) # instanstiiere Quantizer

##################################################################
### Input Quantization (ADC, q_adc)

if DITHER:
    ### add uniform dithering noise, -1/4 LSB < eps_N < 1/4 LSB
    x += 2**(-q_adc['QF']-2)*np.random.rand(len(x))

xq = fx_adc.fix(x) # quantisiere

if fx_adc.N_over: print('Anzahl der Überläufe = ', fx_adc.N_over) 


#################################################################
#
# FFT über N_FFT Datenpunkte skaliert mit 1/N_FFT über Freq. von 0 ... f_S
#
# Um korrekte Amplituden bei einseitiger Darstellung (0 ... f_S/2 anstatt 
# -f_S/2 ... f_S/2) zu erhalten, werden die Spektralwerte verdoppelt:
Yq = 2 * abs(np.fft.fft(xq[0:N_FFT], N_FFT))/ N_FFT # Amplituden
Yq_eff = Yq / sqrt(2) # Effektivwerte

f = np.fft.fftfreq(N_FFT, d = 1./f_S)[0:N_FFT/2] # DFT Frequenzen 0 ... f_S/2
k = np.arange(N_FFT/2) # DFT Index
#------------------------------------------------------------------
# 
Yq_sig = Yq_eff[N_per]     # Eff.Wert des quantisierten Testsignals
PSigQ = Yq_sig ** 2        # Leistung des quantisierten Testsignals
Yq_sig_dB = 10 * log10(PSigQ) # " in dB

PSig = a_sig ** 2 / 2 # Leistung des Testsignals
PSig_dB = 10*log10(PSig)
#
Yq_dc = Yq[0]/2.               # DC-Wert des Ausgangssignals (für Debugging)
Yq_dc_dB = 20 * log10(Yq_dc)   # " in dB

# Überspringe Signalamplitude bei Berechnung der mittleren Rauschleistung / bin
Pq_avg = (np.sum(Yq_eff[0:N_per] ** 2) + np.sum(Yq_eff[N_per+1:N_FFT/2] ** 2)) / (N_FFT/2-1)
Pq_avg_dB = 10 * log10(Pq_avg)

Yq_eff[N_per] = np.sqrt(Pq_avg)
#Yq[N_per] = 0	# oder ersetze einfach durch Null




# Gesamtrauschleistung im Frequenzband von 0 ... f_S/2: e_N = Integral (Yq^2):
# Diskrete Frequenzbänder: Integral -> Summe P = SUM A_eff^2(i)

e_N = np.inner(Yq_eff[0:N_FFT/2], Yq_eff[0:N_FFT/2])
#e_N = Pq_avg * (N_FFT/2 -1)
SNR = 10 * log10(PSig / e_N) 
ENOB = (SNR - 1.7609)/6.0206

###############################################################################
#
# Grafische Darstellung
#
###############################################################################


fig4 = plt.figure(1)
fig4.clf()
ax4 = fig4.add_subplot(111)
ax4.set_xlim([0, f_S/2])
ax4.set_ylim([Hmin, Hmax])

# Rekonstruiere Signalamplitude:
Yq_eff[N_per] = Yq_sig
# Logarithmische Darstellung des Quantisierungsrauschens
# log(0) = -Infinity muss vor Plot durch Hmin ersetzt werden:
Yq_eff_dB = np.maximum(20 * log10(Yq_eff), np.ones(len(Yq_eff))*Hmin) 

#
# Quantisierungsrauschen
ax4.plot(f, Yq_eff_dB[0:N_FFT/2],'k',linewidth = 2, label = r'$N_Q(f)$')
# mittlere Rauschleistung
ax4.plot([0, f_S/2] , [Pq_avg_dB, Pq_avg_dB], color=(0.8,0.8,0.8), linewidth = 2,
    linestyle = '-')
    
ax4.set_title('DFT des quantisierten Signals $s_q[n]$', y=1.01)
ax4.set_ylabel(r'$S_q[k]\; \mathbf{[dBW]} \;\rightarrow$')

#
if unit == "":
    ax4.set_xlabel(r'$k \rightarrow$')
    ax4.get_xaxis().get_major_formatter().set_scientific(False)
else:
    ax4.set_xlabel(r'$f$ [%s] $\rightarrow$' %(unit))
    


#ax4.legend()#(loc = 6)
ax4.grid(True)

if TWIN_AX == True:
    ax4b = ax4.twinx()
    ax4b.set_ylim(ax4.get_ylim()+ 10*log10(N_FFT/2))
    ax4b.set_ylabel(r'$N_q(f)\;  \mathbf{[dBW / bin]} \; \rightarrow$')
else:
    ax4b = ax4


if unit == "":
    ax4.text(fsig * 1.5, 10*log10(PSig),r'Signal: $k =$ %.0d, $P = %.2f$ dBW'
         %(fsig, PSig_dB), ha='left', va='top', transform = ax4.transData, 
        bbox=dict(facecolor='0.9', alpha=0.9, boxstyle="round, pad=0.2"))
else:
    pass

ax4.text(f_S/2 * 0.95, 10*log10(PSig),r'$N_{FFT}\,=$ %d' '\n' r'$f_S = %d$ kHz'
         %(N_FFT, f_S_Print), ha='right', va='top', transform = ax4.transData, 
        bbox=dict(facecolor='0.9', alpha=0.9, boxstyle="round, pad=0.2"))
        
# e_N: Gesamtrauschleistung im Frequenzband von 0 ... f_S/2: e_N = Integral (Yq^2):j
if PRINT_AVG:
    ax4b.text(f_S / 4, Pq_avg_dB,r'$\overline{P_q}=$ %.2f dBW'
       %(Pq_avg_dB), ha='center', va='center', transform = ax4b.transData, 
       bbox=dict(facecolor='1.0', alpha=1.0, boxstyle="round, pad=0.2"))

if PRINT_NOISE:
    ax4b.text(fsig * 1.5, 10*log10(e_N),r'Noise: $e_N =$ %.2e W = %.2f dBW'
       %(e_N, 10*log10(e_N)), ha='left', va='center', transform = ax4b.transData, 
       bbox=dict(facecolor='0.9', alpha=0.9, boxstyle="round, pad=0.2"))
# Textbox, positioniert an relativen- statt Datenkoordinaten (transform = ...)

if PRINT_SNR:
    ax4.text(0.97,0.02,r'SNR = %.2f dB, ENOB = %.2f bits' %(SNR, ENOB),
             ha='right', va='bottom', transform = ax4.transAxes, 
             bbox=dict(facecolor='0.8', alpha=1, boxstyle="round, pad=0.2"))
    
plt.tight_layout()

plt.show()
