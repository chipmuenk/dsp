%===============================================
% LTI_tricks_m.m
%
% Tricks: Logarithmierte Impulsantwort, Interpolation in Matlab
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

% -- Impulse response (lin / log) --
f1 = 50; Ts = 5e-3; 
n = [0:49]; % sample n
t = 0:0.1:49; % start/step/stop
xn = 1.5 + 0.5*cos(2.0*pi*f1*n*Ts);
b = [0.1, 0]; a = [1, -0.9];
%
[h, k] = impz(b, a, 30);
figure(1);
subplot(211); 
stem(k, h, 'r-');
ylabel('h[k] \rightarrow'); grid on;
title('Impulse Response h[n]');
subplot(212); 
stem(k, 20*log10(abs(h)), 'r-'); 
xlabel('k \rightarrow'); grid on;
ylabel('20 log h[k] \rightarrow');
% ------- Filtered signal ---
figure(2); 
yn = filter(b,a,xn);
yt = interp1(n, yn, t, 'cubic');
hold on; % don't overwrite plots
plot(t, yt,'color',[0.8,0,0],'LineWidth',3);
stem([0:length(yn)-1],yn,'b');
xlabel('n \rightarrow'); grid on;
ylabel('y[n] \rightarrow');
title('Filtered Signal');
%  