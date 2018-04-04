# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 09:51:47 2016

@author: muenker
"""

# needs libsndfile and PySoundfile
import os
import soundfile as sf


path = '/home/muenker/Daten/share/Musi/wav/'
#path = '../_media/'
#filename = 'chord.wav'
filename = '07 - Danny Gottlieb with John McLaughlin - Duet.wav'
filename = 'Ole_16bit.wav' # 
filename = '01 - Santogold - L.E.S Artistes.wav'
filename = '01 - Santogold - L.E.S Artistes_20s.wav'
#filename = 'ComputerBeeps2.wav'
filename = 'SpaceRipple.wav'


data, samplerate = sf.read(os.path.join(path, filename))
sf.write(os.path.join(path, 'new_file.ogg'), data, samplerate)
