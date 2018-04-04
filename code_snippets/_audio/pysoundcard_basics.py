# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:51:47 2016

@author: muenker
"""
import sys, os
import numpy as np
from scipy.io.wavfile import read as wavread
from pysoundcard import Stream

()


path = '/home/muenker/Daten/share/Musi/wav/'
#path = '../_media/'
#filename = 'chord.wav'
filename = '07 - Danny Gottlieb with John McLaughlin - Duet.wav'
filename = 'Ole_16bit.wav' # 
filename = '01 - Santogold - L.E.S Artistes.wav'
filename = '01 - Santogold - L.E.S Artistes_20s.wav'
#filename = 'ComputerBeeps2.wav'
filename = 'SpaceRipple.wav'


fs, wave = wavread(os.path.join(path, filename))
"""Play an audio file."""

#fs, wave = wavread(sys.argv[1])
wave = np.array(wave, dtype=np.float32)
wave /= 2**15 # normalize -max_int16..max_int16 to -1..1

blocksize = 512
s = Stream(samplerate=fs, blocksize=blocksize)
s.start()
s.write(wave)
s.stop

