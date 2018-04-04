% LTF_ML_Notchfilter_m.m ====================================================
%
% Matlab Musterlösung zur Aufgabe "Notchfilter"
%
% 
% (c) 2013-APR-26 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
%=======================================================================    
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
% Definiere Nullstellen auf EK:
N = [exp(j*0.3*2*pi); ...
	exp(-j*0.3*2*pi)];
% Pole: gleicher Winkel
P = 0.95 * N;
% "Ausmultiplizieren" von P/N -> Koeff. 
b = poly(N); a = poly(P); 
figure(1);
zplane(N,P);
figure(2);
subplot(211);
% Frequenzgang an 2048 Punkten:
[H,W] = freqz(b,a,2048); 
F = W / (2* pi);
plot(F, 20*log10(abs(H))); grid on;
subplot(212); 
plot(F,angle(H)); 
xlabel('F ->'); grid on;
% Testfreq. (normierte Kreisfreq.):
W_test = [0 0.29 0.3 0.31 0.5]*2*pi; 
%
% Frequenzgang bei Testfrequenzen:
[H_test, W_test]=freqz(b,a,W_test); 
H_test
abs(H_test)
20*log10(abs(H_test))
%