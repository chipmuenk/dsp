%===============================================
% Kap7_sampled_sine.m
%
% Plotte mehrere Sinusfunktionen unterschiedlicher Frequenz und Phase,
% die alle die gleiche abgetastete Sequenz liefern
%
% ToDo: DFT, Phasenspektren
% 
% (c) 2010 Christian Münker - Files zur Vorlesung "Signal Processing"
%===============================================
set(0,'DefaultAxesColorOrder', [0.8 0 0.2; 0 1 0; 0 0 1], ...
      'DefaultAxesLineStyleOrder','-|--|:|-.');

set(0,'DefaultAxesUnits','normalized');      
set(0,'DefaultAxesFontSize',16);;
set(0,'defaultTextFontSize',14);
set(0,'defaultLineMarkerSize', 8);

set(0,'defaultaxeslinewidth',2);
set(0,'defaultlinelinewidth',2);
close all; % alle Plot-Fenster schließen
clear all; % alle Variablen aus Workspace löschen


%% Mehrere Sinusfunktionen in einem Plot; (Sub)sampling 
figure(1); % neue Grafik
f_1 = 500; f_2 = 1000; f_3 = 2500; phi_1 =pi/5;phi_2 =- pi/5; phi_3 =-pi/5;
Np = 1.5; % Plotte Np Perioden mit f_1:
t_min = 0; t_max = t_min + Np / f_1; 
% Erzeuge Vektor mit Np* N + 1 aequidistanten Zeitpunkten (Np Perioden von f_1)
N = 120; % Anzahl Datenpunkte pro Periode von f_1
t  = linspace(t_min, t_max, Np*N+1); 
%
OSR = 1.5;  % Oversampling Ratio in Bezug auf f_1
NS = floor(N / (2 * OSR)); % Abtastung alle NS Zeitpunkte
t_S =  t(1:NS:Np*N); % Vektor mit Sampling-Zeitpunkten
f_S = 2 * f_1 * OSR;
%t_P =  t(1 : NS/8 : 2*N); 
x1=cos(f_1*2*pi*t + phi_1);
x2=cos(f_2*2*pi*t + phi_2);
x3=cos(f_3*2*pi*t + phi_3);
plot(t,x1,'Linewidth',3);
hold on; % ermoegliche mehrere Plots in einer Grafik
plot(t,x2,'Color', [0 0.4 0]);
plot(t,x3,'Color','b');

x1_S=x1(1:NS:Np*N);
x2_S=x2(1:NS:Np*N);
x3_S=x3(1:NS:Np*N);

h1=stem(t_S,x1_S,'Linewidth',2); 
%h2=stem(t_S,x2_S,'Linewidth',2,'color','r');
%h3=stem(t_S,x3_S,'Linewidth',2,'color','black'); 
%h = stem(t,cos(t),'fill','--');
%set(get(h2,'BaseLine'),'LineStyle',':')
%set(h2,'MarkerFaceColor','r');

%x3_P = interpft(x3_S,length(x3_S)*8); % Fourier-Interpolation um den Faktor 8
%plot (t_P,x3_P,'color','r');

axis([t_min t_max -1.2 1.2]);
grid on;
title_string='';
%title_string=sprintf('Abtastung: f_1 = %d Hz, f_2 = %d Hz, f_3 = %d Hz, f_S = %g Hz, ...
% \\phi_1 = %g\\pi,  \\phi_2 = %g\\pi, \\phi_3 = %g\\pi', f_1,f_2,f_3,f_S,phi_1 / pi, phi_2 / pi, phi_3 / pi);
 
title_string1=sprintf('Abtastung: f_1 = %d Hz, f_2 = %d Hz, f_3 = %d Hz, f_S = %g Hz, \\phi_1 = %g\\pi,  \\phi_2 = %g\\pi, \\phi_3 = %g\\pi', f_1,f_2,f_3,f_S,phi_1 / pi, phi_2 / pi, phi_3 / pi); 
title(title_string1);%, '...', 'Fontsize', 16);
xlabel('t in s');
ylabel('x in V');
hold off;
