%===============================================
% ueb_LTI_F_IIR_allgemein_m.m
%
% Einfaches IIR-Filter im Zeit- und Frequenzbereich in Matlab
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
alpha = 0.9; f_S = 1; 
b = [1 0]; % z + 0
% b = [1 0 0]; % z^2 + 0
a = [1 -alpha]; % z - 0.9;
%a = [1 +alpha]; % z + 0.9;
%a = [1 0 -alpha]; % z^2 - 0.9;
%a = [1 0 +alpha]; % z^2 - 0.9; 
figure(1); 
zplane(b,a);  % P/N Diagramm
%
[H,F]=freqz(b,a,1024, f_S);
figure(2); 
plot(F,abs(H)); % Plotte H(f)
grid on; 
xlabel('F bzw. \Omega / 2 \pi'); 
ylabel('|H(F)|');
% N automatisch:
%[himp,t]=impz(b,a,[],f_S); 
[himp,t]=impz(b,a,20,f_S); 
figure(3); 
h1 = stem(t,himp);
set(h1,'Markersize',6,'Linewidth',1.5);
%
xlabel('n'); ylabel('h[n]'); 
grid on;