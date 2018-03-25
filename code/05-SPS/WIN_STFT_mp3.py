# -*- coding: utf-8 -*-
"""
=== WIN_STFT_mp3.py =======================================================

 Demonstriere Short Time Fourier Transform und Aufruf von externen Programmen 
 (MP3-Decoder lame.exe)
 
 TODO:   raw_array entfernen
         Auflösung erhöhen
         lame unter Uníx einbinden

 (c) 2016 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
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

import subprocess, os
from array import array as raw_array
  
mp3file = "../_media/Bluesskala.mp3"
    
sf = 44100
  
samples = 2048
step = 1024
per = 1.0/sf
  
steps = 0
  
img = []
x = fftfreq(samples,per)[:samples/2]
  
  
def hann(size):
  
    n = arange(0,size,1)
    win = 0.5*(1.0 - cos((2*pi*n)/(size-1)))
    return win
  
  
class Mp3Reader:
    '''Reads samples from stereo  mp3 file'''
    def __init__(self,fname, channels=2):
        self.dec = subprocess.Popen(("lame", "--decode", "-t", "--quiet", 
                                   mp3file, "-"), stdout=subprocess.PIPE)
        self.stream = self.dec.stdout
        self.channels = channels
  
    def read(self,n):
        bytes = n*2*self.channels
        input = self.stream.read(bytes)
        if not input:
            return
        a = raw_array('h')
        a.fromstring(input)
        a = array(a)
        if self.channels > 1:
            a.shape = -1,self.channels
            a = np.mean(a,1)
        return a
  
  
class STFT:
    def __init__(self, reader, win_size, step=None, win_func=None):
        self.reader = reader
        self.win_size = win_size
        self.prev_win = None
        if not step:
            step = win_size
        self.step = step
        if win_func:
            self.window = win_func(self.win_size)
        else:
            self.window = None
  
  
    def __iter__(self):
        return self
  
    def next(self):
        if self.prev_win == None:
            readin = self.reader.read(max(self.win_size,self.step))
            if not readin == None or len(readin) < self.win_size:
                raise StopIteration
            data=readin[:self.win_size]
        else:
  
            readin = self.reader.read(self.step)
  
            if readin == None or len(readin) < self.step:
                raise StopIteration
  
            if self.step >= self.win_size:
                data=readin[:self.win_size]
            else:
                data = zeros(self.win_size)
                data[0:self.win_size-self.step] = self.prev_win[self.step:]
                data[self.win_size-self.step:] = readin[0:]
        self.prev_win = data
        if self.window != None:
            input = self.window*data
  
        else:
            input = data
        return fft(input)[:self.win_size/2]
    def _get_data(self):
        pass
  
  
def process_win(win):
    global steps
    steps += 1
    #print "Step", steps
    #plot(win)
    y = np.log(abs(win)+0.0000001)
    if len(y) > 1024:
        a = y.reshape(1024,-1)
        b = np.mean(a,1)
    else:
        b = y
    #b=y
    img.append(b[::-1])
  
  
  
x = fftfreq(samples,per)[:samples/2]
  
if not os.access(mp3file, os.R_OK):
    raise Exception("File %s does not exists or is not readable" % (mp3file,))
  
r = Mp3Reader(mp3file,2)
stft = STFT(r, samples, step, hann)
for win in stft:
    process_win(win)
  
img = array(img)
extent = (per*samples/2.0, per* (step*steps +samples/2), 0, sf/2000.0)
plt.imshow(img.T, extent=extent, aspect='auto')
title("Spectrogram of '" + mp3file + "'")
xlabel("Time(s)")
ylabel("Frequency (kHz)")

figure()
#plt.specgram(r, NFFT = 2048)
plt.show()