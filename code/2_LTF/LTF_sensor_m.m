%===============================================
% ueb_LTI_F_sensor_m.m
%
% Abgetastetes und gefiltertes "Sensorsignal" im Frequenzbereich
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
close all;
clear all;
% Variables
Ts = 1/200.0;
f1 = 50.0;
phi0  = 0;
Tmax = 6.0/f1;
N_Ts = Tmax / Ts;
%- input signal, filt. coeff.
n = 0:round(N_Ts)-1;
xn=1.5+0.5*cos(2*pi*f1*n*Ts+phi0);
b = ones(1,5); a = 1; 
%b = conv([1,1,1],[1,1,1]); a = 1;
%b = [1, 0]; a = [1, -0.9];
%
%----- P/Z-Plot -----
figure(1);
title('Pole/Zero-Plan')
zplane(b,a);
% ----- frequency response -----
figure(2);
[H, W] = freqz(b, a);
f = W  / (Ts * 2 * pi);
[Asig,w]=freqz(b,a,[f1*Ts*2*pi,0.5]);
H_mx = max(abs(H)); H = H / H_mx;
Asig = abs(Asig(1))/H_mx;
title('Frequency Response H(f)');
subplot(311);
plot(f,abs(H));
ylabel('|H(e^{j \Omega})| ->');
%
%
%
%
%
subplot(312);
plot(f, angle(H)/pi);
ylabel('\angle H(e^{j\Omega}) / \pi');
subplot(313);
[tau, w] = grpdelay(b,a, 2048, 200);
%
plot(w, tau);
xlabel('f \rightarrow');
ylabel('\tau_g(e^{j \Omega})/T_S ->');
%