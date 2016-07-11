# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

EXPORT = False
#BASE_DIR = "/home/muenker/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
BASE_DIR = "D:/Daten/HM/dsvFPGA/Vorlesung/2016ss/nologo/img/"
FILENAME = "svg_exp_test"
FMT = ".svg"

fig1 = plt.figure(num=1)
ax1 = fig1.add_subplot(111)

x = np.linspace(-1, 1, 100) 
ax1.plot(x, x)   

ax1.set_xlim(0.3, 0.9) 
ax1.set_ylim(0.3, 0.9)  
ax1.set_xlabel("x label")
ax1.set_ylabel("y label")
ax1.set_title("Title")

if EXPORT:
    fig1.savefig(BASE_DIR + FILENAME + FMT)

plt.show()