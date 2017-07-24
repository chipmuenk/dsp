# -*- coding: utf-8 -*-
#!/usr/bin/python
# BPSK digital modulation: modified example
# by Ivo Maljevic

from scipy import *
from math import sqrt, ceil  # scalar calls are faster
from scipy.special import erfc
import matplotlib.pyplot as plt

rand   = random.rand
normal = random.normal

SNR_MIN   = 0
SNR_MAX   = 10
SNR = 9
FrameSize = 10000
Eb_No_dB  = arange(SNR_MIN,SNR_MAX+1) # list of SNR values in steps of 1 dB
Eb_No_lin = 10**(Eb_No_dB/10.0)  # same in linear scale

# Allocate memory
Pe        = empty(shape(Eb_No_lin))
BER       = empty(shape(Eb_No_lin))

# signal vector (for faster exec we repeat the same frame)
s = 2*random.randint(0,high=2,size=FrameSize)-1
s = s.repeat(4)

loop = 0
snr = SNR
No        = 1.0/snr
Pe  = 0.5*erfc(sqrt(snr))
nFrames   = ceil(100.0/FrameSize/Pe)
error_sum = 0
scale = sqrt(No/2)

for frame in arange(nFrames):
   # noise:
   n = normal(scale=scale, size=FrameSize*4)
   # received signal + noise:
   x = s + n
   # detection (information is encoded in signal phase)
   y = sign(x[2::4]) # take every 4th value
   # error counting
   err = where (y != s[::4])
   error_sum += len(err[0])

   # end of frame loop
   ##################################################

BER[loop] = error_sum/(FrameSize*nFrames)  # SNR loop level
print 'Eb_No_dB=%2d, BER=%10.4e, Pe[loop]=%10.4e' % \
        (Eb_No_dB, BER, Pe)

plt.semilogy(Eb_No_dB, Pe,'r',linewidth=2)
plt.semilogy(Eb_No_dB, BER,'-s')
plt.grid(True)
plt.legend(('analytical','simulation'))
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.show()