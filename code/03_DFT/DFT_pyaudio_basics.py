#!/usr/bin/env python
# -*- coding: utf-8 -*-
#===========================================================================
# DFT_pyaudio_basics.py
#
# Demo für framebasierte DFT
#
# Eine Audio-Datei wird blockweise eingelesen, in numpy-Arrays umgewandelt 
# dann wird die DFT von linkem und rechten Kanal berechnet, ein Teil der 
# DFT-Bänder wird zu Null gesetzt und davon die inverse DFT berechnet. Das
# Ergebnis wird wieder als Audiostream ausgegeben.
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
np_type = np.int16
wf = wave.open(r'C:\Windows\Media\chord.wav', 'rb') # open WAV-File in read mode
wf = wave.open(r'D:\Daten\share\Musi\wav\07 - Danny Gottlieb with John McLaughlin - Duet.wav')
#wf = wave.open(r'D:\Daten\share\Musi\wav\Feist - My Moon My Man.wav')
#wf = wave.open(r'D:\Daten\share\Musi\wav\01 - Santogold - L.E.S Artistes.wav')
p = pyaudio.PyAudio() # instantiate PyAudio + setup PortAudio system

# open a stream on the desired device with the desired audio parameters 
# for reading or writing
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True) 
CHUNK = 256 # number of samples in one frame


# initialize arrays for samples
samples_in = samples_out = zeros(CHUNK*2, dtype=np_type) # stereo
samples_l  = samples_r = zeros(CHUNK, dtype=np_type) # mono

data_out = 'dummy'


while data_out:

# read CHUNK frames to string, convert to numpy array and split in R / L chan.:
# R / L samples are interleaved, each sample is 16 bit = 2 Bytes
    samples_in = np.fromstring(wf.readframes(CHUNK), dtype=np_type)

## Example for dtype = np.int8 (8 bits) = 1 ndarray element, 
##    two consecutive bytes / ndarray elements = 1 sample
## Split signal into L and R channel:   
#    samples_l[0::2] = samples_in[0::4]
#    samples_l[1::2] = samples_in[1::4]
#    samples_r[0::2] = samples_in[2::4]
#    samples_r[1::2] = samples_in[3::4]
#
## Do some numpy magic with samples_l and samples_r
#               ...
#
#  And re-combine L and R channel:
#    samples_out[0::4] = samples_l[0::2]
#    samples_out[1::4] = samples_l[1::2]
#    samples_out[2::4] = samples_r[0::2]
#    samples_out[3::4] = samples_r[1::2]
#---------------------------------------------------------------------------
## dtype = np.int16 (16 bits): 1 ndarray element = 1 sample :
    samples_l = samples_in[0::2]
    samples_r = samples_in[1::2]
    if len(samples_r) < CHUNK: # check whether frame has full length
        samples_out = samples_np = zeros(len(samples_in), dtype=np_type)
        samples_l = samples_l = zeros(len(samples_in)/2, dtype=np_type)

# ---- Numpy Magic happens here (swap L and R channel) ------------------------
    
    fft_l = fft(samples_in[0::2]) # convert to frequency domain
    fft_r = fft(samples_in[1::2])
    fft_l[63:-63] = 0 # set NFFT-bins between 63 ... NFFT-63 = 0
    fft_r[63:-63] = 0 # maintaining  symmetric spectrum -> real-valued time signal
    
    ifft_l = ifft(fft_l).astype(np_type).real # delete imaginary part, abs() doesn't
    ifft_r = ifft(fft_r).astype(np_type).real #   work with bipolar signals
    
    samples_out[0::2] = ifft_l # convert back to time domain
    samples_out[1::2] = ifft_r
         
#    data_out = np.chararray.tostring(samples_np.astype(np_type)) # convert back to string
    data_out = np.chararray.tostring(samples_out) # convert back to string
    stream.write(data_out) # play audio by writing audio data to the stream (blocking)

stream.stop_stream() # pause audio stream
stream.close() # close audio stream

p.terminate() # close PyAudio & terminate PortAudio system

# see: http://stackoverflow.com/questions/23370556/recording-24-bit-audio-with-pyaudio
# http://stackoverflow.com/questions/16767248/how-do-i-write-a-24-bit-wav-file-in-python?