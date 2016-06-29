%===============================================
% DFT_period_m.m
%
% Faltung y[n] = x[n] * h[n] in Matlab
% 
% (c) 2013 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
%===============================================
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
f_S = 2e3; T_S = 1 / f_S; 
N_FFT = 128; t_max = N_FFT*T_S;
f_a = 1e3; 
t = 0:T_S:t_max;
y = sin(2*pi*t*f_a);
Sy = fft(y,N_FFT); 
f = 0:N_FFT-1;
figure(1); clf;
plot(t, y); grid on;
figure(2); clf;
plot(f,Sy); grid on;