# -*- coding: utf-8 -*-
"""
==== NOI_QNoise_Filt.py ===================================================

 Testbench zur Simulation von Quantisierungseffekten in Filtern
 Ein Sinus-Signal mit fsig = f_S * N_per / N_FFT wird als Eingangssignal
 für ein Filter mit quantisierter Signalarithmetik verwendet. N_per ist 
 dabei die Anzahl der Perioden innerhalb der N_FFT Punkte. Dadurch wird 
 eine FFT über eine ganze Zahl von Perioden durchgeführt, es wird keine
 bzw. eine Rechteckfensterung benötigt.

 An das zu testende Filter werden die Koeffizientenvektoren a, b übergeben,
 der Vektor x mit den Datenpunkten und verschiedene Faktoren und Quantizer-
 objekte.


 C. Muenker, 27-5-2011

 ToDo: Ausblenden des DC-Werts verringert Rauschleistung um (N_FFT-1)/N_FFT

===========================================================================
"""
from __future__ import division, print_function, unicode_literals
import numpy as np
from numpy import sin, cos, pi, array, arange, log10, zeros, tan
import scipy.signal as sig
import time # for benchmarking the simulation

import sys
sys.path.append('..')
import dsp_fpga_fix_lib as fx

import matplotlib.pyplot as plt

##plt.rcParams['text.usetex'] = True # veeery slow
#if plt.rcParams['text.usetex'] == True:
##    rc('font', **{'family' : 'sans-serif', 'sans-serif':['Computer Modern Sans serif']})
#    rc('font', **{'family' : 'serif', 'serif':['Times']})
#    plt.rcParams['font.weight'] = 'normal'
#    plt.rcParams['font.size'] = 14.0
#else:
#    rc('font', **{'family' : 'serif', 'serif':['Times New Roman']})
#    plt.rcParams['mathtext.fontset'] = 'cm' # 'stixsans', 'cm', 'stix'
##    plt.rcParams['mathtext.default'] = 'regular'
#    plt.rcParams['font.size'] = 14.0

from matplotlib.pyplot import plot, grid


DEBUG_PLOT_xy = False # Debugging: Plot FX.quantized stimulus and filter output
DEBUG_PLOT_FFT_raw = False  # Debugging: Plot raw FFT
DEBUG_STATE_VARS = False # Plot state variables s1 vs s2


#########################################################################
# Definiere Anzahl Samples, FFT - Länge, Samplingfrequenz etc.
N_FFT = 2000    # Anzahl Datenpunkte für FFT 
N_FFT_min = 3   # Erster Frequenzpunkt, der ausgewertet wird: Yq[0] ist DC-Wert
                # bei Quantisierungsarten, die nicht mittelwertfrei sind ( z.B. 
                # floor = Truncation) werden aufgrund dieses Offets schlechtere
                # SNR-Werte berechnet. Wenn der DC-Offset keine Rolle spielt,
                # sollte N_FFT_min = 2 oder 3 gesetzt werden.
N_Settle = 500  # Die ersten N_Settle Punkte der Simulation werden wegen Ein-
                # schwingeffekten verworfen.
Nx = N_FFT + N_Settle # Anzahl der Datenpunkte insgesamt

f_S = 500       # Abtastfrequenz (eigentlich beliebig, ergibt Skalierungsfaktor)
Ts = 1 / f_S    # Abtastperiode
t = arange(Nx) * Ts # Zeitvektor 0 ... (Nx-1) Ts
#f = [0:1:L-1] / N_FFT * f_S # Frequenzvektor 0 ... f_S in N_FFT Schritten
#f = f_S/2*np.linspace(0,1,N_FFT/2) # Frequenzvektor 0 ... f_S/2 in N_FFT/2 Schritten
Hmin = -150 # Min. y-Wert (dB) für Anzeige und Ersetzen von -infinity Werten
Hmax = 2    # Max. y-Wert (dB) für Anzeige
#########################################################################
# Vorbelegen der Vektoren mit Nullen zur Initialisierung und Beschleunigung
x  = zeros(Nx) # Eingangssignal
xq = zeros(Nx) #        " (quantisiert durch ADC)
y  = zeros(Nx) # Ausgangssignal des Filters
yq = zeros(Nx) #        " (requantisiert durch DAC)
#########################################################################
# Definiere Testsignal für eine kohärente DFT ohne Leckeffekt 
N_per = 53 # Anzahl der Perioden des Signals im DFT-Rechteckfenster
            # N_per muss kleiner als N_FFT / 2 sein (sonst gibt's Aliasing!)
            # und am besten eine Primzahl, z.B. 63, 131, 251, 509, 1021, ...
a_sig = 0.99  # Signalamplitude
fsig = f_S * N_per / N_FFT
x = a_sig * sin(2*pi*t*fsig+1)# + 2^-18 #Testsignal mit Startphase
#x = a_sig * sin(2*pi*t*fsig+1) + 0.001*sin(2*pi*t*251) #Two-Tone Testsignal

#########################################################################
# Definiere Quantizer-Objekte:
# 
# "Quantizer-Objekt" für Python
# z.B. 16 Binärstellen insgesamt, davon 15 Frac.-Bits -> [-1.0 ... 0.9999]:
# q = [0,15], Syntax des Fixpoint-Pakets: xq = fixed(1,15,x)

#---------------------- v4 ------------------------------------
q_adc = {'QI':0, 'QF':15, 'quant':'floor', 'ovfl':'sat'}

q_coeff = {'QI':1, 'QF':8, 'quant':'round'}
q_mul = {'QI':0, 'QF': 15, 'quant':'fix', 'ovfl': 'wrap'}
q_acc = {'QI':0, 'QF': 22, 'quant':'fix', 'ovfl': 'wrap'}

q_dac = {'QI':0, 'QF':20, 'quant':'floor'}

##################################################################
# Variante 1
#c = [0.08, 0.5, 2]
#a = [0.7, 0.48]
# Variante 2
#c = [0.16, 0.25, 2]
#a = [0.7, 0.96]
# Variante 3
# g = [0.16, 1/2.8, 1.4]
# a = [1, 0.96]
#
#b = ones(1,8)
#b = sig.remez(10, [0, 0.15, 0.35, 0.5],[1, 0],[1, 1]) # halfband-filter
b = [-0.123, 0, 0.379, 0.5, 0.379, 0, -0.123] # halfband-filter
b = [0.01623, 0, -0.06871, 0, 0.30399, 0.5, 0.30399, 0, -0.06871, 0, 0.01623]
#b = [0.99, 0.]
#b=1
#a = array([1,])#,zeros(len(b)-1)]
a = 1.0
g = 1.0
##################################################################
# Instanziiere Quantizerobjekte und Fxp-Filter
adc      = fx.Fixed(q_adc) # adc quantizer instance
fil_ma_q = fx.FIX_filt_MA(q_mul, q_acc) # fxpoint filter instance
dac      = fx.Fixed(q_dac) # dac quantizer instance
coeffq   = fx.Fixed(q_coeff)

### quantize coefficients
aq = coeffq.fix(a)
bq = coeffq.fix(b)
gq = coeffq.fix(g)

if coeffq.N_over > 0:
    print ("Coefficient overflows:", coeffq.N_over) 

### Berechnung der Übertragungsfunktion mit idealen und quantisierten Koeff.
# idealer Betragsgang
[w, Hf] = sig.freqz(b, a, worN = int(N_FFT/2)) 
Hf = abs(Hf) * g * a_sig # Idealer Betragsgang zwischen f = 0 ... f_S/2 an N_FFT/2 Punkten
f = w / (2 * pi) * f_S                  # translate w to absolute frequencies

# Betragsgang mit quantisierten Koeffizienten
[w, Hfq] = sig.freqz(bq, aq, int(N_FFT/2))
Hfq = abs(Hfq) * gq * a_sig

wtest = array([fsig, f_S/2]) * 2 * pi / f_S # Testfrequenzen f_sig und f_S/2 (dummy-Punkt, ftest muss Vektor sein)
w, H_sig_tst = sig.freqz(b,a,wtest) # Betrag der Übertragungsfunkt. bei ftest                         
H_sig = abs(H_sig_tst[0])* g * a_sig # Ideale Übertragungsfunktion bei fsig


##################################################################
### Input Quantization (ADC, q_adc)
### of signal + uniform dithering noise, -1/4 LSB < eps_N < 1/4 LSB

#xq = adc.fix(x) # no dithering
xq = adc.fix(x + adc.LSB/4. * np.random.rand(len(x)))

if adc.N_over: print('Overflows in ADC:  ', adc.N_over)

##################################################################
### Filterung
#

# start timer
t_cpu = time.clock()

#y = xq # pass-through

y = fil_ma_q.fxp_filt_df(xq, bq)

print ('Total CPU time: %.5g s' %(time.clock()-t_cpu) )

if fil_ma_q.N_over: print('Overflows in filter:', fil_ma_q.N_over)

#
##################################################################
# Ausgangsquantisierung (z.B. Accu -> DSP, DAC etc.)

yq = dac.fix(y)
if dac.N_over: print('Overflows in DAC:', dac.N_over)

#################################################################
#
# FFT der letzten N_FFT Datenpunkte nach Abklingen des Einschwingprozesses, 
# skaliert mit 1/N_FFT über Freq. von 0 ... f_S
# Auf die letzten Datenpunkte (entsprechend der Länge des FIR-Filters) kann 
# nicht mehr die volle Filterlänge angewendet werden, sie müssen daher 
# ausgeblendet werden.
#
# Um korrekte Amplituden bei einseitiger Darstellung (0 ... f_S/2 anstatt 
# -f_S/2 ... f_S/2) zu erhalten, werden die Spektralwerte verdoppelt:
Yq = 2 * abs(np.fft.fft(yq[N_Settle-len(b)-2:N_Settle-len(b)-2 + N_FFT],
                           N_FFT))/ N_FFT  
#Yq = 2 * abs(np.fft.fft(yq[N_Settle-len(b):N_Settle + N_FFT-len(b)], N_FFT))/N_FFT
#
#------------------------------------------------------------------
# 
Yq_sig = Yq[N_per]           # Amplitude des quantisierten Testsignals
Yq_sig_dB = 20 * log10(Yq_sig) # " in dB
PSigQ = Yq_sig ** 2 / 2         # Leistung des quantisierten Testsignals
PSig = a_sig ** 2 / 2 # Leistung des Testsignals
#
Yq_dc = Yq[0]/2.               # DC-Wert des Ausgangssignals (für Debugging)
Yq_dc_dB = 20 * log10(Yq_dc)   # " in dB


# Überspringe Signalkomponente bei Berechnung des Rauschmittelwerts 
Pq_avg = np.mean(Yq[0:N_per] ** 2) + np.mean(Yq[N_per+1:N_FFT//2] ** 2)

Yq[N_per] = abs(Yq_sig - H_sig) 
Yq[N_per] = 0	# oder ersetze einfach durch Null ...					
# Logarithmische Darstellung des Quantisierungsrauschens
# log(0) = -Infinity muss vor Plot durch Hmin ersetzt werden:
#Yq_dB = np.maximum(20 * log10(Yq), np.ones(len(Yq))*Hmin) 
Yq_dB = 20 * np.log10(Yq)
# Mittlere Rauschleistungsdichte:
Yq_avg = 10*log10(np.mean (Yq[N_FFT_min:N_FFT//2] * Yq[N_FFT_min:N_FFT//2])/2)

# Gesamtrauschleistung im Frequenzband von 0 ... f_S/2: e_N = Integral (Yq^2):
# Diskrete Frequenzbänder: Integral -> Summe
# Faktor 1/2: DFT liefert AMPLITUDEN des Schmalbandrauschens -> P = SUM A^2(i)/2

e_N = 0.5 * np.inner(Yq[N_FFT_min:N_FFT//2], Yq[N_FFT_min:N_FFT//2])
SNR = 10 * log10(PSig / e_N) 
ENOB = (SNR - 1.7609)/6.0206


###############################################################################
#
# Grafische Darstellung
#
###############################################################################


#------------------------------------------------------------------
# Quantisierungsrauschen 
#------------------------------------------------------------------

fig4 = plt.figure(1)
fig4.clf()
ax4 = fig4.add_subplot(111)
ax4.set_xlim([0, f_S/2])
ax4.set_ylim([Hmin, Hmax])

# Idealer Betrags-Frequenzgang (einseitig) aus Koeffizienten
ax4.plot(f, 20*log10(Hf), 'b', lw = 2, label = r'$H(f)$ ideal')
ax4.plot(f, 20*log10(Hfq), 'k--', lw = 2, label = r'$H_Q(f)$ quant. Koeff.')
#
# Quantisierungsrauschen
ax4.plot(f, Yq_dB[0:N_FFT//2],'k',linewidth = 2, label = r'$N_Q(f)$')
# mittlere Rauschleistung
ax4.plot([0, f_S/2] , [Yq_avg, Yq_avg], color=(0.8,0.8,0.8), linewidth = 2,
    linestyle = '-')
# Markiere Signalamplitude
ax4.plot([fsig, fsig] , [Hmin, Yq_sig_dB], 'r', linewidth = 2)
#
ax4.set_xlabel(r'$f$ [Hz] $\rightarrow$')
ax4.set_ylabel(r'$S, \;Y_q(f)\; \mathbf{[dB]} \;\rightarrow$')
ax4.set_title('DFT des quantisierten Signals')

ax4.legend(loc = 'center left')
ax4.grid(True)

ax4b = ax4.twinx()
ax4b.set_ylim(ax4.get_ylim()+ 10*log10(N_FFT/2))
ax4b.set_ylabel(r'$N_q(f)\;  \mathbf{[dBW / bin]} \; \rightarrow$')


# e_N: Gesamtrauschleistung im Frequenzband von 0 ... f_S/2: e_N = Integral (Yq^2):j
#ax4b.text(f_S/4,10*log10(e_N),r'$
ax4b.text(0.97, 0.1,r'$e_N =$ %.2e W = %.2f dBW'
     %(e_N, 10*log10(e_N)), ha='right', va='center', transform = ax4b.transAxes, 
    bbox=dict(facecolor='0.9', alpha=0.9, boxstyle="round, pad=0.1"))

# Textbox, positioniert an relativen- statt Datenkoordinaten (transAxes)
ax4.text(0.97,0.02,r'SNR = %.2f dB, ENOB = %.2f bits'
     %(SNR, ENOB), ha='right', va='bottom', transform = ax4.transAxes, 
     bbox=dict(facecolor='0.9', alpha=1))
    
plt.tight_layout()


#text(0.01,1.03,['f_S = ', num2str(f_S,4), ' Hz'],'FontSize',12)# 'HorizontalAlignment', 'center')
##text(0.7,1.03,['f_{sig} = ', num2str(fsig,4), ' Hz'], 'HorizontalAlignment', 'center')
#text(1,1.03,['H(f_{sig} = ', num2str(fsig,4), ' Hz',') = ', ...
# num2str(20*log10(H_sig),4), ' dB'],'FontSize',12,'HorizontalAlignment', 'right')
##text(1,1.03,['H(f_{sig} = ) = ', num2str(20*log10(H_sig),4), ' dB'],'HorizontalAlignment', 'right')
#------------------------------------------------------------------
# DEBUGGING: Transiente Darstellung des quantisierten Signals 
if DEBUG_PLOT_xy:
    plt.figure(2)
    plt.clf()
    plot(t, xq, t, yq, t, x)
    plt.xlabel('t')
    plt.ylabel('xq, yq')
    plt.legend(['xq', 'yq', 'x'])
    grid(True)
#------------------------------------------------------------------
# DEBUGGING: Zustandsvariablen / Ausgangsamplitude im x/y-Diagramm 
#           zur Darstellung von Korrelationen und Amplitudenbereich
if DEBUG_STATE_VARS:
    plt.figure(3)
    plt.clf()
    plot (s[:-1],s[1:])
    plot(s, yq)
    plt.xlabel('s_1[n]')
    plt.ylabel('s_2[n]')
    # plt.ylabel('y[n]')
    grid (True)
#------------------------------------------------------------------
# DEBUGGING: Zweiseitige FFT des quantisierten Signals, YQ über k
if DEBUG_PLOT_FFT_raw:

    # DC und Signalfrequenz wurden noch nicht ersetzt, 0 wird durch 1e-12
    # ersetzt, da ansonsten log Plot bei Werten von Null abbricht )
    plt.figure(4)
    plt.clf()
    f2 = f_S*np.linspace(0,1.,N_FFT)
    f2 = np.fft.fftfreq(N_FFT) * f_S
    f2 = np.fft.fftshift(f2)
    
    Yq = np.fft.fftshift(Yq)
    plot (f2,20*np.log10(Yq[0:N_FFT]+1e-15)) # Zweiseitige FFT (0 ... f_S) 
						  # von yq, geplottet bis f_S
    #plot (f2,(Yq[0:N_FFT])) # linearer Plot
    plt.xlabel(r'$f [Hz] \; \rightarrow$')
    plt.ylabel(r'$Y_q(f)$ [dB] ->')
    plt.title('Raw FFT')
    grid (True)
#------------------------------------------------------------------

plt.show()
