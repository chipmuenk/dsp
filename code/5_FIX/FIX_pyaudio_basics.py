# -*- coding: utf-8 -*-
#===========================================================================
# FIX_pyaudio_basics.py
#
# Demo für die Einbindung von pyAudio
#
# Eine Audio-Datei wird blockweise eingelesen, in numpy-Arrays umgewandelt 
# dann werden linker und rechter Kanal getauscht und die Datei wird auf
# ein Audio-Device ausgegeben.
#
# 
#===========================================================================
from __future__ import division, print_function, unicode_literals # v3line15
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
import os
#import _portaudio
import pyaudio
import wave

def setupAudio(p):
    """
    Create and manage selection box for audio interfaces
    """
    deviceList = []
    device_out_list = []
    device_in_list = []
    print("Device count", p.get_device_count())

#    p = pyaudio.PyAudio() # instantiate PyAudio, start PortAudio system + list devices
    defaultInIdx = 1# p.get_default_input_device_info()['index']
    defaultOutIdx = 1# p.get_default_output_device_info()['index']

    print("Defaultin", defaultInIdx)
    for i in range(p.get_device_count()):
         deviceList.append(p.get_device_info_by_index(i))

         print (deviceList[i])
         if deviceList[i]['maxInputChannels'] > 0:
             if i == defaultInIdx:
                 device_in_list.append(('* '+ deviceList[i]['name'], str(i)))                 
             else:
                 device_in_list.append((deviceList[i]['name'], str(i)))
         else:
             if i == defaultOutIdx:
                 device_out_list.append(('* '+ deviceList[i]['name'], str(i)) )                
             else:
                 device_out_list.append((deviceList[i]['name'], str(i)))
#        print("Default Output Device : %s" % self.p.get_default_output_device_info()['name'])
#        self.comboBoxAudioOut.addItems(deviceList)



np_type = np.int16


path = '/home/muenker/Daten/share/Musi/wav/'
filename = 'Ole_16bit.wav'
filename = 'SpaceRipple.wav'

p = pyaudio.PyAudio() # instantiate PyAudio + setup PortAudio system
setupAudio(p)

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

MONO = n_chan == 1
CHUNK = 1024 # number of samples in one frame


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
    if len(samples_in) < 2 * CHUNK: # check whether frame has full length
        samples_out = samples_np = zeros(len(samples_in), dtype=np_type)
        samples_l = samples_r = zeros(len(samples_in)/2, dtype=np_type)

# ---- Numpy Magic happens here (swap L and R channel) ------------------------
    if MONO:
        samples_out = samples_in
    else:
        samples_out[0::2] = samples_in[1::2]
        samples_out[1::2] = samples_in[0::2]
#    samples_out = samples_in
      
## Stereo signal processing: This only works for sample-by-sample operations,
## not e.g. for filtering where consecutive samples are combined
    
#    samples_out = abs(samples_in)
#    samples_out = ((samples_in.astype(np.float32))**2 /2**15).astype(np_type)
#    samples_out = sqrt(samples_in.astype(np.float32)**2).astype(np_type)
    
#    data_out = np.chararray.tostring(samples_np.astype(np_type)) # convert back to string
    data_out = np.chararray.tostring(samples_out) # convert back to string
    stream.write(data_out) # play audio by writing audio data to the stream (blocking)
#    data_out = wf.readframes(CHUNK) # direct streaming without numpy

stream.stop_stream() # pause audio stream
stream.close() # close audio stream

p.terminate() # close PyAudio & terminate PortAudio system
print("closed audio stream!")

# see: http://stackoverflow.com/questions/23370556/recording-24-bit-audio-with-pyaudio
# http://stackoverflow.com/questions/16767248/how-do-i-write-a-24-bit-wav-file-in-python?