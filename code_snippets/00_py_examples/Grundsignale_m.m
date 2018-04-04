%===============================================
% ueb_LTI_Grundsignale_m.m
%
% Beispiele für Darstellung von einfachen Funktionen in Matlab
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

%% Komplexe Exponentialschwingung y = exp( j \omega t)
t=0:0.01:2;
y=exp(j*2*pi*t);
figure;
plot3(real(y),imag(y),t,'Linewidth',2);
title('Komplexe Exponentialschwingung y = exp( j \omega t)');
xlabel('\Re \{\} ');
ylabel('\Im \{\}');
zlabel('t \rightarrow');
grid on;
axis square;

figure;
%% Rechteckimpuls rect(t/T_0)
T0=1;
t=-2:0.01:2;
x=(abs(t)<0.5*T0);
plot(t,x);
xlabel('t/T_0 \rightarrow');
ylabel('x(t) \rightarrow');
title('rect(x) - Funktion'); 
grid on;
axis([-2 2 -0.2 1.2]);

figure;
%% sin x / x  - Funktion (sinc - function)
T0=1;
t=-8:0.01:8 ;
f0=1/T0 ;
x=sin((pi*f0*t))./(pi*f0*t) ; % ./  : Elementweise Division
plot(t,x);
grid on;
xlabel('t/T_0 \rightarrow');
ylabel('x(t) \rightarrow');
title('sin(x)/x - Funktion');
% !! Warning: Divide by zero!!!
% Abhilfe? -> sinc( ) Funktion benutzen!
%=========================================================================

%% Dirac-Puls (-Kamm)
t=-3:3;
x=ones(1,length(t)); % 1 - Vektor der Größe 1 x 7 [ vgl. x=zeros(a,b) ];
figure;
stem(t,x,'^'); % "stem" = Stamm, Stengel
axis([-3.6 3.6 -.2 1.2]);
title('Periodische Diracfunktion');
xlabel('t/T_0 \rightarrow');
ylabel('x(t) \rightarrow');
% Formatierung:
text(-3.4,0.5,'...','Fontsize',18,'color','b'); % Pünktchen links
text(3.2,0.5,'...','Fontsize',18,'color','b'); % Pünktchen rechts
text(0.15,1.0 ,'(1)','Fontsize',14); % Dirac - Gewicht
