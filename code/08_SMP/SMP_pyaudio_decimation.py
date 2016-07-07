# -*- coding: utf-8 -*-
"""
=== SMP_pyaudio_decimation.py =============================================


 Code-Beispiel zur Dezimation (mit / ohne Filterung) von Audio-Signalen

 Eine Audio-Datei wird blockweise eingelesen, in numpy-Arrays umgewandelt 
 dann werden Samples entnommen und die Datei wird auf
 ein Audio-Device ausgegeben.
 
===========================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
from numpy import (pi, log10, exp, sqrt, sin, cos, tan, angle, arange,
                    linspace, array, zeros, ones)
from numpy.fft import fft, ifft, fftshift, ifftshift, fftfreq
import scipy.signal as sig

import matplotlib.pyplot as plt
from matplotlib.pyplot import (figure, plot, stem, grid, xlabel, ylabel,
    subplot, title, clf, xlim, ylim)

import pyaudio
import wave
import os

np_type = np.int16

path = '/home/muenker/Daten/share/Musi/wav/'
path = '../_media/'
#filename = 'chord.wav'
filename = '07 - Danny Gottlieb with John McLaughlin - Duet.wav'
filename = 'Ole_16bit.wav' # 
#filename = '01 - Santogold - L.E.S Artistes.wav'
#filename = 'ComputerBeeps2.wav'
filename = 'SpaceRipple.wav'

wf = wave.open(os.path.join(path, filename))
n_chan = wf.getnchannels() # number of channels in wav-file
w_samp = wf.getsampwidth() # wordlength of samples
rate_in = wf.getframerate() # samplerate in wav-file

print("Channels:", n_chan, "\nSample width:",w_samp,"bytes\nSample rate:",rate_in)

NFFT = 256
R = 4 # downsampling ratio - only use R = 2^k ?
FILTER = True  # if TRUE: filter samples before downsampling
CHUNK = 8192 # number of samples in one OUTPUT frame (= R input frames)
N_CHUNK_MAX = CHUNK / rate_in *1000 # read max. 10 chunks

rate_out = rate_in // R
print(rate_out)

p = pyaudio.PyAudio() # instantiate PyAudio + setup PortAudio system

# open a stream on the desired device with the desired audio parameters 
# for reading or writing
stream = p.open(format=p.get_format_from_width(w_samp),
                channels=n_chan,
                rate=rate_out,
                output=True)

STEREO = n_chan == 2

#print (channels)
# initialize arrays for samples
samples_in = zeros(CHUNK*R*n_chan, dtype='float32') # mono or stereo input samples
samples_l  = samples_r = zeros(CHUNK, dtype=np_type) # R/L output samples
samples_out = zeros(CHUNK*n_chan, dtype=np_type) # mono / stereo output samples (decimated)
samples = np.ndarray(1)

data_out = 'start' 
n_chunk = 0

while data_out and n_chunk < N_CHUNK_MAX: # keep reading + writing data until write buffer is empty

# read CHUNK frames to string, convert to numpy array and split in R / L chan.:
# R / L samples are interleaved, each sample is 16 bit = 2 Bytes
# scale down signal to avoid overflows caused by filter ripple in the passband
    n_chunk += R
    samples_in = np.fromstring(wf.readframes(CHUNK*R), dtype=np_type)*0.9
#    if len(samples_in):print(len(samples_in), max(samples_in))

#---------------------------------------------------------------------------
# de-interleave WAV data stream into R and L channel:
    if FILTER:
        # apply decimation filter before downsampling by  a factor of R
        samples_l = sig.decimate(samples_in[0::n_chan], R, ftype='FIR')
        if STEREO:
            samples_r = sig.decimate(samples_in[1::2], R, ftype='FIR')
    else:
        # downsample by R, picking every Rth sample / every 2*Rth sample (stereo)
        samples_l = samples_in[0::n_chan*R] # take every R'th sample, starting with 0
        if STEREO:
            samples_r = samples_in[1::2*R] # take every 2*R'th sample, starting with 1
        
    if len(samples_l) < (CHUNK//n_chan):
        print(len(samples_l), "last frame!")# check whether frame has full length
        samples_out = zeros(np.ceil(len(samples_in)/(n_chan*R))*n_chan, dtype=np_type)

# interleave R and L channel
    samples_out[0::n_chan] = samples_l
    if STEREO:
        samples_out[1::2] = samples_r
        
    fig = figure(1)
    print(np.shape(samples_out))
    plot(arange(len(samples_l)), samples_l)
       
## Stereo signal processing: This only works for sample-by-sample operations,
## not e.g. for filtering where consecutive samples are combined
    
#    samples_new = abs(samples)

# convert numpy data in arbitrary format  back to string    
#    data_out = np.chararray.tostring(samples_np.astype(np_type)) 
    data_out = np.chararray.tostring(samples_out.astype(np.int16)) # convert back to string
    samples = np.concatenate((samples,samples_l))
    stream.write(data_out) # play audio by writing audio data to the stream (blocking)
#    data_out = wf.readframes(CHUNK) # direct streaming without numpy

stream.stop_stream() # pause audio stream
stream.close() # close audio stream

p.terminate() # close PyAudio & terminate PortAudio system
print("Closed audio stream!")

# ============== Create spectrogram =========================================
dbmin = -100; dbmax = 0 # Limits for log. display
win = sig.windows.kaiser(NFFT,12, sym = False) # needs NFFT and shape parameter beta
figure(2)
Pxx, freqs, bins, im = plt.specgram(samples / (NFFT * 2**15), NFFT=NFFT, Fs=rate_out, 
                            noverlap=NFFT/2, mode = 'magnitude', window = win, 
                            scale = 'dB', vmin = dbmin, vmax = dbmax)
xlabel(r'$t \; \mathrm{in \; s}\;\rightarrow$', fontsize = 16)
ylabel(r'$f \; \mathrm{in \; Hz}\;\rightarrow$', fontsize = 16)
xlim([0,len(samples)/rate_out])
ylim([0,rate_out/2])
plt.colorbar(label = "Specgram")
plt.tight_layout()
plt.show()