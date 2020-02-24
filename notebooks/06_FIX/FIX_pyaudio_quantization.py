# -*- coding: utf-8 -*-
"""
=== FIX_pyaudio_quantization.py ============================================

Demonstrate quantization effects with audio signals:

Read an audio file frame by frame, quantize the samples and stream the data
to an audio device via pyaudio.
 
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import sys
sys.path.append('..')
import dsp_fpga_fix_lib as fx

import pyaudio
import wave
import os

np_type = np.int16 # format of audio samples
CHUNK = 1024 # number of stereo samples per frame

path = '/home/muenker/Daten/share/Musi/wav/'
#path = '../_media/'

#filename = 'chord.wav'
filename = '07 - Danny Gottlieb with John McLaughlin - Duet.wav'
#filename = 'Ole_16bit.wav'
#filename = '01 - Santogold - L.E.S Artistes.wav'
#filename = 'SpaceRipple.wav'

wf = wave.open(os.path.join(path, filename))
n_chan = wf.getnchannels() # number of channels in wav-file
w_samp = wf.getsampwidth() # wordlength of samples
rate_in = wf.getframerate() # samplerate in wav-file

print("Channels:", n_chan, "\nSample width:",w_samp,"bytes\nSample rate:",rate_in)

p = pyaudio.PyAudio() # instantiate PyAudio + setup PortAudio system

# open a stream on the desired device with the desired audio parameters 
# for reading or writing
stream = p.open(format=p.get_format_from_width(w_samp),
                channels=n_chan,
                rate=rate_in,
                output=True) 

# Define quantization mode and create a quantization instance for each channel
# quantize with just a few bits:
q_obj = {'Q':-2.15,'quant':'round','ovfl':'sat'} # try 'quant':'round', 'ovfl':'sat'

# Overflows QI = -1 means the MSB is 2^{-1} = 0.5
#q_obj = {'Q':-1.15,'quant':'fix','ovfl':'wrap'} # try  'ovfl':'sat'

fx_Q_l = fx.Fixed(q_obj)
fx_Q_r = fx.Fixed(q_obj) 

# initialize arrays for audio samples
samples_in = zeros(CHUNK*2, dtype=np_type) # stereo int16
samples_out = zeros(CHUNK*2, dtype=float) # stereo float
samples_l  = samples_r = zeros(CHUNK, dtype=np_type) # separate channels int16

data_out = 'start'

while data_out:

# read CHUNK stereo samples to string and convert to numpy array.
# R / L samples are interleaved, each sample is 16 bit wide (dtype = np.int16)
    samples_in = np.fromstring(wf.readframes(CHUNK), dtype=np_type)

    # split interleaved data stream into R and L channel:
    samples_l = samples_in[0::2]
    samples_r = samples_in[1::2]
    if len(samples_r) < 2:
        break # break out of the while loop when out of data
    # Check whether there was enough data for a full frame
    if len(samples_r) < CHUNK: # check whether frame has full length
        samples_out = samples_np = zeros(len(samples_in), dtype=float)
#        samples_l = samples_r = zeros(len(samples_in)/2, dtype=np_type)

# - Convert from 16 bit integer to floating point in the range -1 ... 1
# - Quantize 
# - Construct interleaved data stream from R/L channel (still as floating point)
    
# Process L and R channel separately
#    samples_out[0::2] = fx_Q_l.fix(samples_l/2**15)
#    samples_out[1::2] = fx_Q_r.fix(samples_r/2**15)

# Stereo signal processing: This only works for sample-by-sample operations,
# not e.g. for filtering where consecutive samples have to be combined
    samples_out = fx_Q_r.fix(samples_in / 2. **15)

# Do explicit type casting to 16 bin and convert data back to string 
    data_out = np.chararray.tostring((samples_out * 2.**15).astype(np_type)) # convert back to string
#    data_out = wf.readframes(CHUNK) # direct streaming without numpy
    stream.write(data_out) # play audio by writing audio data to the stream (blocking)

stream.stop_stream() # pause audio stream
stream.close() # close audio stream

p.terminate() # close PyAudio & terminate PortAudio system
print("Overflows: ", fx_Q_r.N_over)
print("Closed audio stream!")