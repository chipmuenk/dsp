% ueb_FIL_intro_m.m ====================================================
%
% Demonstrate different filter design
% methods and compare results to specifications 
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
f_S = 400; % Samplingfrequenz 
f_DB = 40; % Eckfrequenzen DB
f_SB = 50; %   und SB
F_DB = f_DB/(f_S/2.); % Normierte Frequenzen
F_SB = f_SB/(f_S/2.); % bezogen auf f_S / 2
%
A_DB = 0.1; % max. Ripple im DB (log.)
A_DB_lin = (10^(A_DB/20.0)-1)/ ...
    (10^(A_DB/20.0)+1); % und lin.
A_SB = 40; % min. Daempfung im SB in dB
A_SB_lin = 10^(-A_SB/20.0); % und linear
%
L = 44; % Manuelle Vorgabe Filterordnung 
%%%%%%% FIR-Filterentwurf %%%%%%%%%%%%%%%
a = 1; 
%=== Windowed FIR / Least Square =========
F_c  = F_DB / (f_S/2);  % -6dB Frequenz
b = fir1(L, F_c, hamming(L+1));
%=== Frequency Sampling ==================
b = fir2(L, [0, F_DB, F_SB, 1], ...
    [1, 1, 0, 0], hamming(L+1));
%=== REMEZ / Parks-McClellan / Equiripple 
W_DB = 1; W_SB = 4; % manuelle Ordnung:
b = firpm(L, [0, F_DB, F_SB, 1],...
  [1, 1, 0, 0], [A_DB, A_SB], [W_DB, W_SB]);
% minimale Ordnung: 
[L_min, F , A, W] = firpmord([F_DB, F_SB],...
    [1, 0], [A_DB_lin, A_SB_lin], 2);
b = firpm(L_min, F, A, W);
%%%%%%%% IIR-Filterentwurf %%%%%%%
%% Butterworth-Filter
[L_b, F_b] = buttord(F_DB, F_SB, A_DB, A_SB);
[b, a] = butter(Lb, F_b);
%% Elliptisches Filter
[L_e,F_e] = ellipord(F_DB, F_SB, A_DB, A_SB);
[b, a] = ellip(L_e, A_DB, A_SB, F_e);
%===========================================
% Calculate H(w), w = 0 ... pi, 1024 Pts.
[H, w] = freqz(b, a, 1024); 
% Translate w to physical frequencies: 
f = w / (2 * pi) * f_S;     
%%%%%%%%%%%%%% Plot the Results %%%%%%%%%%%%
%% Pol/Nullstellenplan
figure(1);
[z, p, k] = zplane(b, a);
%% ----- Impulsantwort -----
figure(2); hold on; grid on;
[h, td] = impz(b, a, [], f_S);  %Impulsantwort / Koeffizienten
stem(td, h);
title('Impulsantwort h[n]');
%% ----- Linear frequency plot -----
figure(3); grid on; hold on;
plot(f, abs(H));
title('Betragsfrequenzgang');      
%% Log. Frequenzgang mit Spezifikationen
figure(5);
subplot (211);
plot(f,20 * log10(abs(H)), 'r'); grid on; hold on;
plot([0, f_DB],[-A_DB, -A_DB],'b--'); % untere Spec-Grenze
plot([f_DB, f_DB], [ -A_DB, -A_DB-10], 'b--'); %@ F_DB
if length(a) == 1
    plot([0, f_DB],[A_DB, A_DB], 'b--'); % obere Spec-Grenze
    axis([0, f_DB * 1.1, -A_DB*1.1, A_DB * 1.1]);
else
    plot([0, f_DB], [0, 0], 'b--'); % obere Spec-Grenze
    axis([0, f_DB * 1.1, -A_DB * 1.1, A_DB * 0.1]);
end
title('Betragsfrequenzgang in dB');
%
subplot(212); grid on; hold on;
plot(f,20 * log10(abs(H)), 'r');
plot([0,  f_DB],[-A_DB, -A_DB],'b--'); % untere Grenze DB
if length(a) == 1
    plot([0,  f_DB], [A_DB, A_DB],'b--'); % obere Grenze DB
else
    plot([0, f_DB], [0, 0], 'b--'); % obere Grenze DB
end
plot([f_SB, f_S/2.], [-A_SB, -A_SB], 'b--'); % obere Grenze SB
plot([f_DB, f_DB], [-A_DB, -A_DB-10], 'b--'); % @ F_DB
plot([f_SB, f_SB],[1, -A_SB],'b--'); % @ F_SB
%
%=========================================
%% Phasengang (voller Bereich, unwrapped)
figure(6); grid on; hold on;
plot(f,unwrap(angle(H))/pi);
% Ohne unwrap wird Phase auf +/- pi umgebrochen
title('Phasengang (normiert auf Vielfache von \pi)');
%% Groupdelay
figure(7);
[tau_g, w] = grpdelay(b,a);
plot(w, tau_g); grid on; hold on;
title('Group Delay \tau_g'); 
ylim([max(min(tau_g)-0.5,0), (max(tau_g) + 0.5)]);
