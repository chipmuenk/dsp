#! /usr/bin/python
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 22:25:32 2015

@author: Christian Muenker
"""

import math
import numpy
import pyaudio


def sine(frequency, length, rate):
    length = int(length * rate)
    factor = float(frequency) * (math.pi * 2) / rate
    return numpy.sin(numpy.arange(length) * factor)


def play_tone(stream, frequency=440, length=1, rate=44100):
    chunks = []
    chunks.append(sine(frequency, length, rate))
    chunks.append(sine(2 * frequency, length, rate))
    chunks.append(sine(frequency / 2., length, rate))

    chunk = numpy.concatenate(chunks) * 0.25

    stream.write(chunk.astype(numpy.float32).tostring())
    
#if __name__ == '__main__':
#    p = pyaudio.PyAudio()
#    stream = p.open(format=pyaudio.paFloat32,
#                    channels=1, rate=44100, output=1)
#
#    play_tone(stream)
#
#    stream.close()
#    p.terminate()
    
#------------------------------

 
