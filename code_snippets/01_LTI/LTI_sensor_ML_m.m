% LTI_sensor_m.m  ============================================
% Matlab Musterloesung zu "Periodizität abgetasteter Signale"
%
% Abtastung und Filterung eines Sensorsignals in Matlab
% 
% (c) 2013 Christian Muenker - Files zur Vorlesung "DSV auf FPGAs"
%=================================================================
set(0,'DefaultAxesColorOrder', [0.8 0 0.2; 0 1 0; 0 0 1], ...
      'DefaultAxesLineStyleOrder','-|--|:|-.');

set(0,'DefaultAxesUnits','normalized');      
set(0,'DefaultAxesFontSize',14);
set(0,'defaultTextFontSize',14);
set(0,'defaultLineMarkerSize', 6);

set(0,'defaultaxeslinewidth', 2);
set(0,'defaultlinelinewidth', 2);
close all; % alle Plot-Fenster schlieszen
clear all; % alle Variablen aus Workspace loeschen

% Variables
Ts = 1/200.0;
f1 = 50.0;
phi0  = 0;
tstep = 1e-3;
Tmax = 6.0/f1;
N_Ts = Tmax / Ts;
%- Calculate input signals
t = 0:tstep:Tmax-tstep;
n = 0:round(N_Ts)-1;
xt=1.5+0.5*cos(2*pi*f1*t+phi0); 
xn=1.5+0.5*cos(2*pi*f1*n*Ts+phi0);
%xn = zeros(N_Ts); xn(0) = 1
%--- Plot input signals ---
figure(1);
xlabel('Time (s) ->')
ylabel('Amplitude (V) ->')
ttlstr=strcat('x[n] = 1.5 + ', ...
'0.5 cos[2 pi * 50 / 200 Hz n]');
%
title(ttlstr);
grid on; hold on;
plot(t, xt, 'b');
stem(n*Ts, xn, 'r');
ylim([-0.1 2.2]);
% line in DATA coordinates:
line([0 Tmax],[1.5 1.5]);
%
%--- Impulse response -----
figure(2);
%h=[1, 1, 1, 1, 1];
h=conv([1,1,1],[1,1,1]);
%h=[1,0.5, 0.25, 0.125, 0.0625]
stem([0:length(h)-1], h); 
xlabel('n ->');
ylabel('h[n] ->');
title('Impulse Response h[n]');
%--- Filtered signal ------
figure(3);
yn=conv(xn,h)/5;
%
stem([0:length(yn)-1], yn);
xlabel('n ->'); 
ylabel('y[n] ->');
title('Filtered Signal'); grid on;
% Print signal + filtered signal
str_n = sprintf('%6d', [0:11]);
disp(['n =    ', str_n]);
xn_s = sprintf('%6.3f', xn(1:12));
disp(['x[n] = ', xn_s]);
yn_s = sprintf('%6.3f', yn(1:12));
disp(['y[n] = ', yn_s]);
%