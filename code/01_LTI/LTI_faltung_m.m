%===============================================
% LTI_faltung_m.m
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
h = [0.25 0.5 0.25]; 
x = [1, 1, 1, 1, 1];
y = conv(x, h); % Faltung
n = 0:length(y)-1; 
figure(1);
stem(n, y); grid on;
xlabel('n ->'); 
ylabel('y[n] ->');
title('Faltung');
%