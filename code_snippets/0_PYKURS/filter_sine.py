# -*- coding: utf-8 -*-
"""
Created on Tue Apr  5 16:40:46 2016

Beispielsfile aus dem Python/Numpy-Schnupperkurs zur Filterung im Zeitbereich

Signalgenerierung
-----------------
- Erzeuge einen Zeitvektor t 0 ... 10 ms (10e-3), in Schritten von 1 / 20 kHz
- Berechne die Sinusfunktion s an diesen Zeitpunkten, skaliere ihn mit der
  Amplitude 1.5 und addiere einen DC-Offset von 2
- Erzeuge ein Zufallssignal n (randn: normal- oder gaußverteilt) mit der 
  gleichen Länge len(t) = len(s) und der Varianz 0.5
- Addiere Sinus- und Zufallssignal (gleiche Länge!) zu einem verrauschten Sinus-
  signal sn



@author: Christian Münker
"""

# Importiere zusätzliche Module 
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

t = np.arange(0, 1e-2, 1/2e4)

s = 1.5 * np.sin(2*np.pi*1000 * t) + 2
n = 0.5 * np.random.randn(len(t))
sn = s + n

"""
Filterung
---------
Definiere die folgenden Filter über ihre Impulsantwort h:
- Moving Average Filter der Länge 4 als Liste [1, 1, 1, 1]. Da mit der Liste nicht
  gerechnet werden kann, wird sie in ein Numpy-Array umgewandelt
- Moving Average Filter der Länge 20 über den np.ones() Befehl. Um eine Grund-
  verstärkung von 1 zu erreichen, muss auch hier (elementweise) durch die Länge 
  dividiert werden.
- Tiefpass-Filter mit der Eckfrequenz 0.1 * fS/2 = 1 kHz und der Ordnung 40, 
  entworfen mit dem Remez oder Parks-McClellan Algorithmus (-> Kapitel 4)

Die letzte Zuweisung überschreibt die vorherigen, kommentieren Sie aus was nicht
benötigt wird.
"""
h = np.array([1,1,1,1])/4
h = np.ones(20)/20.
h = sig.fir_filter_design.remez(40,[0, 0.1, 0.2, 0.5], [1,0])

"""
Filtern Sie das verrauschte Sinus-Signal, indem Sie es mit der Impulsantwort
falten. Achtung: Bei der Faltung wird das Ausgangssignal um die Filterordnung 
also um len(h)-1 länger!
"""
y = np.convolve(sn, h)

fig1 = plt.figure(1)
ax11 = fig1.add_subplot(311)
ax12 = fig1.add_subplot(312)
ax13 = fig1.add_subplot(313)

print("t:",len(t), "y:",len(y))
ax11.plot(t, sn)
ax12.plot(t, y[len(h)-1:])
ax13.stem(h)

plt.show()