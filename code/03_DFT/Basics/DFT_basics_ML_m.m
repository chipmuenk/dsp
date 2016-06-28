% ueb_DFT_basics_ML_m.m  ============================================
% Matlab Musterlösung zu "Fourierreihe und synchrone DFT"
%
% Berechnung und Darstellung der DFT in Matlab
% 
% (c) 2013 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
%================================================================= 
set(0,'DefaultAxesColorOrder', [0.8 0 0.2; 0 1 0; 0 0 1], ...
      'DefaultAxesLineStyleOrder','-|--|:|-.');

set(0,'DefaultAxesUnits','normalized');      
set(0,'DefaultAxesFontSize',16);
set(0,'defaultTextFontSize',16);
set(0,'defaultLineMarkerSize', 6);

set(0,'defaultaxeslinewidth',2);
set(0,'defaultlinelinewidth',2);
close all; % alle Plot-Fenster schließen
clear all; % alle Variablen aus Workspace löschen
%
N_FFT = 3; 
f_a = 1e3; T_mess = 1. / f_a;
t = linspace(0,T_mess-T_mess/NFFT,N_FFT)
xn = 1 + 1 * cos(2*pi*t*f_a);
% calculate DFT and scale it with 1/N: 
Xn = fft(xn,length(xn))/length(xn);
%
%
f = linspace(0,1,length(xn));
for i=1:length(Xn)
  if abs(Xn(i))/max(abs(Xn(i)))<1e-10
    Xn(i) = 0;
  endif
endfor
figure(1); 
subplot(211);
stem(f,abs(Xn));
subplot(212);
stem(f,angle(Xn)/pi);