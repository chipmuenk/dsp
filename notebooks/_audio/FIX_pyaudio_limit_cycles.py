# -*- coding: utf-8 -*-
"""
==== FIX_pyaudio_limit_cycles.py ===========================================

 Code-Beispiel zur Filterung von Audio WAV-Dateien mit FIR / IIR 
 Fixpoint-Filtern:

 Eine Audio-Datei wird blockweise eingelesen, in numpy-Arrays umgewandelt 
 beide Kanäle werden gefiltert und die Datei wird auf
 ein Audio-Device ausgegeben.

===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import sys
sys.path.append('..')
import dsp_fpga_fix_lib as fx

import os
import pyaudio
import wave

np_type = np.int16 # data format for audio sample
CHUNK = 1024 # number of samples per frame
# path = '/home/muenker/Daten/share/Musi/wav/'
path = '../_media/'
# filename = 'Ole_16bit.wav'
filename = 'SpaceRipple.wav'


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


wf = wave.open(os.path.join(path, filename))
n_chan = wf.getnchannels() # number of channels in wav-file
w_samp = wf.getsampwidth() # wordlength of samples
rate_in = wf.getframerate() # samplerate in wav-file

print("Channels:", n_chan, "\nSample width:",w_samp,"bytes\nSample rate:",rate_in)
MONO = n_chan == 1 # test if audio file is mono

p = pyaudio.PyAudio() # instantiate PyAudio + setup PortAudio system

# open a stream on the desired device with the desired audio parameters 
# for reading or writing
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                output=True) 

q_obj = {'Q':14.0,'quant':'round','ovfl':'wrap'} # integer quantization
fx_Q_fil = fx.Fixed(q_obj)
fx_Q_l = fx.Fixed(q_obj)
fx_Q_r = fx.Fixed(q_obj)



# initialize arrays for samples
samples_in = samples_out = zeros(CHUNK*2, dtype=np_type) # stereo
samples_l  = samples_r = zeros(CHUNK, dtype=np_type) # mono

data_out = 'dummy'


while data_out:

# read CHUNK frames to string, convert to numpy array and split in R / L chan.:
# R / L samples are interleaved, each sample is 16 bit = 2 Bytes
    samples_in = np.fromstring(wf.readframes(CHUNK), dtype=np_type)

#---------------------------------------------------------------------------
## dtype = np.int16 (16 bits): 1 ndarray element = 1 sample :
    samples_l = samples_in[0::2]
    samples_r = samples_in[1::2]
    if len(samples_r) < 2:
        break # break out of the while loop when out of data
    # Check whether there was enough data for a full frame
    if len(samples_r) < CHUNK: # check whether frame has full length
        samples_out = samples_np = zeros(len(samples_in), dtype=np_type)
        samples_l = samples_l = zeros(len(samples_in)/2, dtype=np_type)

# Do the filtering:
    if MONO:
        print(max(samples_in))
#        samples_out = fx_Q_l.fix(samples_in).astype(np_type)
        samples_out = IIR1(fx_Q_l, samples_in, -0.1).astype(np_type)
#        samples_out = samples_in
        print(max(samples_out))
    else:
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

print("Overflows: ", fx_Q_r.N_over)

p.terminate() # close PyAudio & terminate PortAudio system