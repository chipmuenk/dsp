# -*- coding: utf-8 -*-
"""
===== FIL_BPSK_SNR.py =================================================
  
 Simulate Bit Error Rate (BER) of BPSK digital modulation vs. SNR
 by Ivo Maljevic
 
=======================================================================
"""
from __future__ import division, print_function, unicode_literals

import numpy as np
import numpy.random as rnd
from numpy import arange, sqrt
import matplotlib.pyplot as plt
from scipy.special import erfc

SNR_MIN     = 0 # dB
SNR_MAX     = 9 # dB
Eb_No_dB    = arange(SNR_MIN,SNR_MAX+1)
SNR         = 10**(Eb_No_dB/10.0)  # linear SNR

Pe          = np.empty(np.shape(SNR))
BER         = np.empty(np.shape(SNR))

loop = 0
for snr in SNR:      # SNR loop
     Pe[loop] = 0.5*erfc(sqrt(snr)) # calculate error probability
     VEC_SIZE = int(np.ceil(100/Pe[loop]))  # vector length is a function of Pe
    
     # signal vector, new vector for each SNR value
     s = 2*rnd.randint(0,high=2,size=VEC_SIZE)-1
    
     # linear power of the noise; average signal power = 1
     No = 1.0/snr
    
     # noise
     n = sqrt(No/2)*rnd.randn(VEC_SIZE)
    
     # signal + noise
     x = s + n
    
     # decode received signal + noise
     y = np.sign(x)
    
     # find erroneous symbols
     err = np.where(y != s)
     error_sum = float(len(err[0]))
     BER[loop] = error_sum/VEC_SIZE
     print ('Eb_No_dB=%4.2f, BER=%10.4e, Pe=%10.4e' % \
    (Eb_No_dB[loop], BER[loop], Pe[loop]))
#     plt.figure(1)
#     plt.plot(arange(100),s[0:100])
     loop += 1
#plt.semilogy(Eb_No_dB, Pe,'r',Eb_No_dB, BER,'s')
plt.figure(2)
plt.semilogy(Eb_No_dB, Pe, linewidth=2)
plt.semilogy(Eb_No_dB, BER,'-s')
plt.grid(True)
plt.legend(('analytical','simulation'))
plt.xlabel('Eb/No (dB)')
plt.ylabel('BER')
plt.show()

