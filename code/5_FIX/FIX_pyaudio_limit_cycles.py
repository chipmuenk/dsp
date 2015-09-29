#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===========================================================================
# FIX_pyaudio_limit_cycles.py
#
# Code-Beispiel zur Filterung von Audio WAV-Dateien mit rekursiven Filtern:
#
# Eine Audio-Datei wird blockweise eingelesen, in numpy-Arrays umgewandelt 
# beide Kanäle werden gefiltert und die Datei wird auf
# ein Audio-Device ausgegeben.
#
# 
#===========================================================================
#from __future__ import division, print_function, unicode_literals # v3line15

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

import dsp_fpga_lib as dsp
import dsp_fpga_fix_lib as fx
#------------------------------------------------------------------ v3line30
# Ende der gemeinsamen Import-Anweisungen
import pyaudio
import wave

def IIR1(q_inst, x, a):
    """
    Rekursives Fixpoint-Filter mit y[i] = Q< x[i-1] + a y[i-1] >
    Da y immer von der Vorgeschichte abhängt, kann die Rechnung nicht vekto-
    risiert werden : Langsam!
    """
    y = zeros(len(x))
    for i in range(0,len(x)-1):
#        y[i+1] = x[i] + a * y[i]  # no quantization       
        y[i+1] = q_inst.fix(x[i] + a * y[i])
    print(np.max(y))
    return y
    
def FIR1(q_inst, x, a):
    """
    Transversales Fixpoint-Filter mit y[i] = Q< x[i] + a x[i-1] >
    Da y immer von der Vorgeschichte abhängt, kann die Rechnung nicht vekto-
    risiert werden : Langsam!
    """
    y = zeros(len(x))
    for i in range(0,len(x)-1):
#        y[i+1] = x[i] + a * y[i]  # no quantization       
        y = q_inst.fix(x + np.append(0, a * x[1:]))
    print(np.max(y))
    return y

np_type = np.int16
wf = wave.open(r'C:\Windows\Media\chord.wav', 'rb') # open WAV-File in read mode
wf = wave.open(r'D:\Daten\share\Musi\wav\01 - Santogold - L.E.S Artistes.wav')
p = pyaudio.PyAudio() # instantiate PyAudio + setup PortAudio system

# open a stream on the desired device with the desired audio parameters 
# for reading or writing
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True) 
CHUNK = 1024 # number of samples per frame

q_obj = {'Q':20.0,'quant':'fix','ovfl':'none'} # integer quantization
fx_Q_l = fx.Fixed(q_obj)
fx_Q_r = fx.Fixed(q_obj) 


# initialize arrays for samples
samples_in = samples_out = zeros(CHUNK*2, dtype=np_type) # stereo
samples_l  = samples_r = zeros(CHUNK, dtype=np_type) # mono

data_out = 'dummy'


while data_out:

# read CHUNK frames to string, convert to numpy array and split in R / L chan.:
# R / L samples are interleaved, each sample is 16 bit = 2 Bytes
    samples = np.fromstring(wf.readframes(CHUNK), dtype=np_type)

#---------------------------------------------------------------------------
## dtype = np.int16 (16 bits): 1 ndarray element = 1 sample :
    samples_l = samples[0::2]
    samples_r = samples[1::2]
    if len(samples_r) < CHUNK: # check whether frame has full length
        samples_out = samples_np = zeros(len(samples), dtype=np_type)
        samples_l = samples_l = zeros(len(samples)/2, dtype=np_type)

# Do the filtering:
    
#    samples_out[0::2] = fx_Q_l.fix(samples_l)
#    samples_out[1::2] = fx_Q_r.fix(samples_r)
    samples_out[0::2] = FIR1(fx_Q_l, samples_l, -0.1)
    samples_out[1::2] = FIR1(fx_Q_r, samples_r, -0.1)    
       
## Stereo signal processing: This only works for sample-by-sample operations,
## not e.g. for filtering where consecutive samples are combined
    
#    samples_new, N_ov = dsp.fixed(q_obj, samples)
#    samples_new = abs(samples)
#    samples_new = (samples.astype(np.float32))**2 /2**15
#    samples_new = sqrt(samples.astype(np.float32)**2 )
    
    data_out = np.chararray.tostring(samples_out) # convert back to string
    stream.write(data_out) # play audio by writing audio data to the stream (blocking)
#    data_out = wf.readframes(CHUNK) # direct streaming without numpy

stream.stop_stream() # pause audio stream
stream.close() # close audio stream

print(fx_Q_r.N_over)

p.terminate() # close PyAudio & terminate PortAudio system