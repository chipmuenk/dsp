# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 15:37:02 2016

@author: muenker
"""

def myabs(c):
    if c >= 0:
        return c
    else:
        return -c
        
for i in range(10):
    print("abs(%d) = %d" %(i - 5, myabs(i - 5)))