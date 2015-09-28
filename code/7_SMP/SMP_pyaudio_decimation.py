#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===========================================================================
# SMP_pyaudio_decimation.py
#
# Code-Beispiel zur Dezimation (mit / ohne Filterung) von Audio-Signalen
#
# Eine Audio-Datei wird blockweise eingelesen, in numpy-Arrays umgewandelt 
# dann werden Samples entnommen und die Datei wird auf
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
#------------------------------------------------------------------ v3line30
# Ende der gemeinsamen Import-Anweisungen
import pyaudio
import wave

np_type = np.int16

wf = wave.open(r'C:\Windows\Media\chord.wav', 'rb') # open WAV-File in read mode
#wf = wave.open(r'D:\Musik\wav\Jazz\07 - Duet.wav')
wf = wave.open(r'D:\Daten\share\Musi\wav\Feist - My Moon My Man_sel.wav')

#wf = wave.open(r'D:\Daten\share\Musi\wav\Feist - My Moon My Man.wav')
#wf = wave.open(r"D:\Daten\share\Musi\wav\12-Santana _ She's Not There.wav")
#wf = wave.open(r'D:\Daten\share\Musi\wav\The White Stripes-Seven Nation Army.wav')
wf = wave.open(r'D:\Daten\share\Musi\wav\01 - Santogold - L.E.S Artistes.wav')

R = 16 # downsampling ratio - only use R = 2^k
DECIMATE = True  # if TRUE: filter samples before downsampling

rate_out = wf.getframerate() / R

p = pyaudio.PyAudio() # instantiate PyAudio + setup PortAudio system

# open a stream on the desired device with the desired audio parameters 
# for reading or writing
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=rate_out,
                output=True) 
CHUNK = 4096 # number of samples in one output frame

# initialize arrays for samples
samples_in = zeros(CHUNK*R*2, dtype='float32') # stereo samples in
samples_out = zeros(CHUNK*2, dtype=np_type) # stereo samples out (decimated)
samples_l  = samples_r = zeros(CHUNK, dtype='float32') # mono samples

data_out = 'dummy' 

while data_out: # keep reading + writing data until write buffer is empty

# read CHUNK frames to string, convert to numpy array and split in R / L chan.:
# R / L samples are interleaved, each sample is 16 bit = 2 Bytes
# scale down signal to avoid overflows caused by filter ripple in the passband
    samples_in = np.fromstring(wf.readframes(CHUNK*R), dtype=np_type)*0.9

#---------------------------------------------------------------------------
## dtype = np.int16 (16 bits): 1 ndarray element = 1 sample :
    if DECIMATE:
        samples_l = sig.decimate(samples_in[0::2], R, ftype='FIR')
        samples_r = sig.decimate(samples_in[1::2], R, ftype='FIR')
    else:
        samples_l = samples_in[0::2*R] # take every R'th sample, starting with 0
        samples_r = samples_in[1::2*R] # take every R'th sample, starting with 1
    if len(samples_r) < CHUNK: # check whether frame has full length
        samples_out = zeros(np.ceil(len(samples_in)/(2*R))*2, dtype=np_type)
        samples_l   = samples_r  = zeros(np.ceil(len(samples_in)/(2*R)), 
                                             dtype=np_type)

# Do some numpy magic here
    samples_out[0::2] = samples_l
    samples_out[1::2] = samples_r
       
## Stereo signal processing: This only works for sample-by-sample operations,
## not e.g. for filtering where consecutive samples are combined
    
#    samples_new = abs(samples)

# convert numpy data in arbitrary format  back to string    
#    data_out = np.chararray.tostring(samples_np.astype(np_type)) 
    data_out = np.chararray.tostring(samples_out) # convert back to string
    stream.write(data_out) # play audio by writing audio data to the stream (blocking)
#    data_out = wf.readframes(CHUNK) # direct streaming without numpy

stream.stop_stream() # pause audio stream
stream.close() # close audio stream

p.terminate() # close PyAudio & terminate PortAudio system
print("Finished!")