% DFT_period_ML_m.m ====================================================
%
% Matlab Musterlösung zu "DFT periodischer Signale mit Python / Matlab"
% Berechnung und Darstellung der DFT in Matlab
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
f_S = 5e3; T_S = 1 / f_S; 
N_FFT = 1000; T_mess = N_FFT * T_S; 
f_a = 1e3; f_b = 1.1e3; DC = 1;
t = [0:1:N_FFT-1]*T_S;
y = DC + 0.5 * sin(2*pi*t*f_a) ...
      + 0.2 * cos(2*pi*t*f_b);
fprintf('P = %g\n', sum(y.^2) ...
   * T_S / T_mess);
figure(1); clf;
subplot(212);
Sy = fft(y,N_FFT) / N_FFT;
f = linspace(-f_S/2, ...
    f_S/2 - f_S/N_FFT, N_FFT); 
Sy = fftshift(Sy); % center DC-comp.
stem(f,abs(Sy)); grid on; 
xlim([-2000,2000]);ylim([-0.1,1.1]);
xlabel('f [Hz]->');ylabel('Y(f) ->'); 
title('Zweiseitige DFT')
fprintf('P = %g\n', Sy * Sy');
subplot(222);
Sy = 2*fft(y,N_FFT)/N_FFT;
Sy(1) = Sy(1)/2;
f = linspace(0, f_S/2, N_FFT/2);
stem(f, abs(Sy(1:N_FFT/2)));
xlabel('f [Hz]->');ylabel('Y(f) ->');
axis([-100,2000,-0.1, 1.1]);
title('Einseitige DFT'); grid on;