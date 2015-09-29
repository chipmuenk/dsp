%=========================================================================
% Kap3_Filterentwurf.m
%
% Demonstrate different filter design methods and compare results
% to specifications 
% 
% (c) 2011-04-05 Christian Münker - Files zur Vorlesung "DSV für FPGAs"
%=========================================================================

set(0,'DefaultAxesColorOrder', [0.8 0 0.2; 0 1 0; 0 0 1], ...
      'DefaultAxesLineStyleOrder','-|--|:|-.');

set(0,'DefaultAxesUnits','normalized');      
set(0,'DefaultAxesFontSize',16);
set(0,'defaultTextFontSize',16);
set(0,'DefaultTextFontName','Arial');
set(0,'DefaultAxesFontName','Arial');
set(0,'defaultLineMarkerSize', 3);

set(0,'defaultaxeslinewidth',2);
set(0,'defaultlinelinewidth',2);

close all; % alle Fenster schließen
clear all; % Variablen löschen

DEF_PRINT = 0;          % 1: Print Plots to PNG-Files
PRINT_PATH = ('D:/FH/Master/DSV/pueb3_Filterentwurf_'); % Path and base-name of Plot-Files
%
% Select which plots to show
SHOW_POLE_ZERO = 1;     % Pole-zero-plot
SHOW_LIN_H_f = 1;       % Linear plot of H(f)
SHOW_LOG_H_f = 1;       % Log. plot of H(f)
SHOW_LIN_LOG_H_f = 1;   % Lin./ log. plot of H(f)
SHOW_PHASE = 1;         % Phase response
SHOW_GRPDELAY = 1;      % Group delay
SHOW_IMPZ = 1;          % Impulse response
SHOW_TRAN_SIN = 1;      % Transient response to sine (interpolated)
SHOW_TRAN_SIN_INTP = 1; % Transient response to sine

DEF_FIR = 1; % 0: IIR transfer function 
             % 1: FIR transfer function
			
DEF_F_RANGE = 'f_S/2'; % select how to display the frequency axis:
%                       'F/2'   normalized frequency F = 0 ... 0.5 (f_S/2)
%                       'f_S/2' absolute frequency f = 0 ... f_S/2
 
N_FFT = 2048; % Länge FFT für freqz und grpdelay - Berechnung
f_S = 1000; T_S = 1/f_S;% Samplingfrequenz 
f_S2 = f_S/2; %Nyquistfrequenz = halbe Samplingfrequenz
f_DB = 20; %Grenzfrequenz Durchlassband
f_SB = 50; % Grenzfrequenz Stopband
f_sig = 10; % Testsignalfrequenz
%
f_notch = 1000; % Centerfrequenz für Notchfilter
notch_eps = 0.1; % relative Breite des Notchs
%
% Vorgabe von A_DB entweder linear _oder_ in dB
%A_DB_log = 0.086; % max. Ripple im Durchlassband in dB
A_DB_lin = 0.01; % max. Ripple im Durchlassband
A_SB = 40; % min. Sperrdämpfung im Stoppband in dB

L = 86; %Vorgabe Filterordnung (Simulation nur möglich wenn L geradzahlig!)
%
%
% relative Grenzfrequenzen; in Matlab / Octave normiert auf HALBE Abtastfrequenz!
F_DB = f_DB/f_S2;
F_SB = f_SB/f_S2;
F_DBSBrel = (f_SB + f_DB)/f_S; % Mittelwert von F_SB und F_DB
F_notch = f_notch/f_S2;
F_notchl = F_notch * notch_eps;
F_notch_u = F_notch / notch_eps;
F_sig = f_sig / f_S2;
%
if DEF_FIR
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% FIR-Filterentwurf
%
% Ergebnis ist jeweils Spaltenvektor mit Zählerkoeffizienten / Impulsantwort des FIR-Filters 
aa = 1; % Spaltenvektor der Nennerkoeffizienten = 1 bei FIR-Filtern
%
%===================================================
% Filterentwurf mit least-square / Fourier-Approximation:
%
% Angabe von Frequenz/Amplituden Punkten im Durchlassband und Sperrband, 
%    optional mit Gewichtung (hier: 1 im Durchlassband, 4 im Sperrband)
%	
%bb = firls(L, [0 F_DB F_SB 1 ],[1 1  0 0],[2 1]);
%
%===================================================
%% Filterentwurf mit gefensterter (Default: Hamming) Fourier-Approximation:
%
% Hier wird nur eine Grenzfrequenz spezifiziert, kein Übergangsbereich wie bei firls 
% -> u.U. ungünstig, da don't care - Bereich zwischen f_DB und f_SB nicht ausgenutzt wird!
% Mögliche Typen: 'low' (default), 'high', 'stop' (F muss zwei Elemente haben)
% Für 'stop' und Multiband-Filter wird ein Frequenzvektor F übergeben mit den Eckfrequenzen
% von Stop- und Durchlassbändern. F=[0.35 0.55] erzeugt z.B. Bandpass.
%
%bb = fir1(L, F_DBSBrel);
%bb = fir1(L, F_DBSBrel,'low', 'hann'); % optional anderer Fenstertyp

%===================================================
%% Filterentwurf mit Parks-McClellan / Remez / Equiripple - Methode
%
bb = remez(L, [0 F_DB  F_SB 1 ],[1 1  0 0],[1 1]); % Octave
%bb = firpm(L, [0 F_DB  F_SB 1 ],[1 1  0 0],[1 1]); % Matlab
%===================================================
% Filterentwurf mit Frequency Sampling
%
%bb = fir2(L, [0 F_notch_l F_notch F_notch_u 1 ],[1 1 0 1 1]); %(N, f, m)
%===================================================

% Achtung: zplane bei Octave erwartet Filter-Koeffizienten als _Zeilenvektor_ , 
% bb muss daher transponiert werden
% Wird zplane mit Spaltenvektoren aufgerufen, werden die Vektoren als Pole / Nullstellen
% interpretiert und dementsprechend falsch dargestellt!
bb = bb'; % nur Octave, für Matlab auskommentieren!

else

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% IIR-Filterentwurf 
%
% Hinweise: 
%- Toleranzband im DB ist bei IIR-Entwurf definiert zwischen 0 ... -A_DB
%- Filterentwurf über [bb,aa] = ... führt zu numerischen Problemen bei Filtern höherer
%   Ordnung (ca. L > 10, selbst Ausprobieren!) Alternative Form:
%   [z,p,g] = ... liefert Nullstellen, Pole und Gain
%
%===================================================
	if exist('A_DB_log','var')
		A_DB = A_DB_log;
	else
	  A_DB = 20*log10(A_DB_lin+1);
    end
%===================================================
% Butterworth-Filter
% Grenzfrequenz definiert -3dB Frequenz und muss hier daher manuell angepasst werden!
% -> ausprobieren für optimales Ergebnis oder Funktion buttord verwenden!
% Ergebnis ist Ordnung L und normierte -3dB Grenzfrequenz F_c
%L = 9; % manuelle Wahl
%[bb,aa] = butter(L, F_DB *1.07); % manuelle Wahl
[L,F_c] = buttord(F_DB, F_SB, A_DB, A_SB)
[bb,aa] = butter(L, F_c); 
%===================================================
% Bessel-Filter
% Grenzfrequenz definiert -3dB Frequenz und muss hier daher manuell angepasst werden!
% -> ausprobieren für optimales Ergebnis!
%[bb,aa] = maxflat(L, F_DB *1.07); % besself not working in Octave
%===================================================
% Elliptisches Filter:
% Spezifikation sind hier maximaler Ripple im Durchlass- und Sperrband
%L = 4; % manuelle Wahl
% Funktion ellipord liefert Abschätzung für Ordnung sowie die Eckfrequenz des DB
[L,FDB] = ellipord(F_DB, F_SB, A_DB, A_SB) 
%[bb,aa] = ellip(L, A_DB, A_SB, F_DB);
[bb,aa] = ellip(L, A_DB, A_SB, 23.5/f_S2);
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

end

if strcmp(DEF_F_RANGE,'f_S/2')
%    f_range = [0 f_S/2];
elseif strcmp(DEF_F_RANGE,'F/2')
      f_S = 1;
else
    break;  
end 
f_range = [0 f_S/2];

%
% Define x-axis labels
if f_S == 1
    my_x_axis_f = sprintf('Norm. Frequenz [\\Omega / 2 \\pi oder f / f_S]');
    my_x_axis_t = sprintf('Sample n');
else  
    my_x_axis_f = sprintf('Frequenz [Hz]');
    my_x_axis_t = sprintf('Zeit [s]');
end

[H,f]=freqz(bb,aa,N_FFT, f_S); % calculate H(f) along the upper half of unity circle

%
%======================================
f_g = [f_sig, f_DB, f_SB]; % Vektor mit Testfrequenzen
% Berechne Frequenzantwort bei Testfrequenzen und gebe sie aus:
H1 = 20*log10(abs(freqz(bb,aa,f_g,f_S))) 
%=======================================

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Plotten der Ergebnisse
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%=========================================
%% Pol/Nullstellenplan
%=========================================
if SHOW_POLE_ZERO 
	zplane(bb',aa'); 
	title('Pol/Nullstellenplan');
end
%
%=========================================
%% Impulsantwort
%=========================================

if SHOW_IMPZ
	figure(2);
	[h,td]=impz(bb,aa,[],f_S);%Impulsantwort / Koeffizienten
	h1=stem(td,h); grid on; %Impulsantwort normiert
	set(h1,'Markersize',3);
	xlabel(my_x_axis_t);
	ylabel('h[n]');
	title('Impulsantwort h[n]');
  if DEF_PRINT
		  print (strcat(PRINT_PATH,'impz.png'),'-dpng');
  end
end

%=========================================
%% Linear frequency plot
%=========================================
if SHOW_LIN_H_f
	figure(3);
	%[H,f]=freqz(bb,aa,N_FFT,f_S);
	plot(f,abs(H));grid on;
	axis([f_range 0 1.2]);
	title('Betragsfrequenzgang (linear)');
	xlabel(my_x_axis_f);
	ylabel('|H(f)|');
	if DEF_PRINT
		  print (strcat(PRINT_PATH,'lin.png'),'-dpng');
    end
end

%=========================================
%% Log. Frequenzgang mit Spezifikationen
%=========================================
if SHOW_LOG_H_f
	if exist('A_DB_log','var')
		A_DB = A_DB_log;
	else
	  A_DB = 20*log10(A_DB_lin+1);
    end
	if DEF_FIR 
		A_DBo = A_DB; % Bei FIR - Filtern ist obere Grenze +A_DB
	else
		A_DBo = A_DB/10; % Bei IIR - Filtern ist obere Grenze 0 dB
    end
	DEL_A_DB = A_DBo + A_DB;

	figure(4);
	subplot (2, 1, 1);
	plot(f,20*log10(abs(H)));grid on; hold on;
	plot([0 F_DB*f_S/2],[-A_DB -A_DB],'b:');
	plot([F_DB*f_S/2  F_DB*f_S/2],[ -A_DB -A_DB-10],'b:');
	plot([0 F_DB*f_S/2],[A_DBo A_DBo],'b:');
	%axis([0 F_DB*f_S/2 * 1.1 -A_DB-DEL_A_DB*1.05 A_DBo+DEL_A_DB*1.05]);
	axis([0 F_DB*f_S/2 * 1.1 -A_DB*1.1 A_DBo*1.1]);
	
	title('Betragsfrequenzgang in dB (Durchlassbereich)');
	%xlabel(my_x_axis_f);
	ylabel('|H(f)| in dB');
	%
	subplot (2, 1, 2);
	plot(f,20*log10(abs(H)));grid on; hold on;
	plot([0  F_DB*f_S/2],[-A_DB -A_DB],'b:');
	plot([0  F_DB*f_S/2],[A_DB A_DB],'b:');
	plot([F_SB*f_S/2  f_S/2],[-A_SB -A_SB],'b:');
	plot([F_DB*f_S/2  F_DB*f_S/2],[ -A_DB -A_DB-10],'b:');
	plot([F_SB*f_S/2  F_SB*f_S/2],[1 -A_SB],'b:');
	axis([f_range -80 1]);
	title('Betragsfrequenzgang in dB (Sperrbereich)');
	xlabel(my_x_axis_f);
	ylabel('|H(f)| in dB');
	if DEF_PRINT
		  print (strcat(PRINT_PATH,'log.png'),'-dpng');
    end
end

%=========================================
%% Lin. (DB) / log. (SB) Frequenzgang mit Spezifikationen
%=========================================
if SHOW_LIN_LOG_H_f
	if exist('A_DB_lin','var')
        A_DB = 1 + A_DB_lin;
    else
        A_DB = 10^(A_DB_log/20);
    end
    A_DBmin = 1- (A_DB -1)*1.1; 
	if DEF_FIR 
		A_DBo = A_DB; % Bei FIR - Filtern ist obere Grenze +A_DB
		A_DBmax = 1+ (A_DB -1)*1.1; 
	else
		A_DBo = 1; % Bei IIR - Filtern ist obere Grenze 1
		A_DBmax = 1+ (A_DB -1)/10;
    end
	%DEL_A_DB = A_DBo + A_DB;

	figure(5);
	subplot (2, 1, 1);
	plot(f,(abs(H)));grid on; hold on;
	plot([0 F_DB*f_S/2],[1-A_DB_lin 1-A_DB_lin],'b:');
	plot([F_DB*f_S/2  F_DB*f_S/2],[ 1-A_DB_lin 1-A_DB_lin*2],'b:');
	plot([0 F_DB*f_S/2],[A_DBo A_DBo],'b:');
	%axis([0 F_DB*f_S/2 * 1.1 -A_DB-DEL_A_DB*1.05 A_DBo+DEL_A_DB*1.05]);
	axis([0 F_DB*f_S/2 * 1.1 A_DBmin A_DBmax] );
	
	title('Betragsfrequenzgang (Durchlassbereich)');
	%xlabel(my_x_axis_f);
	ylabel('|H(f)|');
	%
	subplot (2, 1, 2);
	plot(f,20*log10(abs(H)));grid on; hold on;
	plot([0  F_DB*f_S/2],[-A_DB -A_DB],'b:');
	plot([0  F_DB*f_S/2],[A_DB A_DB],'b:');
	plot([F_SB*f_S/2  f_S/2],[-A_SB -A_SB],'b:');
	plot([F_DB*f_S/2  F_DB*f_S/2],[ -A_DB -A_DB-10],'b:');
	plot([F_SB*f_S/2  F_SB*f_S/2],[1 -A_SB],'b:');
	axis([f_range -80 1]);
	title('Betragsfrequenzgang in dB (Sperrbereich)');
	xlabel(my_x_axis_f);
	ylabel('|H(f)| in dB');
  if DEF_PRINT
		  print (strcat(PRINT_PATH,'linlog.png'),'-dpng');
  end
end


%=========================================
%% Phasengang (voller Bereich, unwrapped)
%=========================================%
if SHOW_PHASE
	figure(6);
	plot(f,unwrap(angle(H))/pi);grid on;
	% Ohne unwrap wird Phase auf +/- pi umgebrochen
	title('Phasengang (normiert auf Vielfache von \pi)');
	xlabel(my_x_axis_f);
	ylabel('\phi(f)/\pi');
end

%=========================================
%% Groupdelay
%=========================================
if SHOW_GRPDELAY
	figure(7);
	[tau_g,w] = grpdelay(bb,aa,N_FFT,f_S); % Octave benötigt Angabe von N_FFT
	plot(w, tau_g); grid on;
	axis([f_range max(min(tau_g)-0.5,0) max(tau_g) + 0.5]);
	title('Group Delay \tau_g');
	xlabel(my_x_axis_f);
	ylabel('\tau_g(f)/T_S');
end

%=========================================
%% Filterantwort auf sinusförmiges Eingangssignal
%=========================================      
if SHOW_TRAN_SIN
	figure(8);
	if DEF_FIR
		tmax =  L / f_S; % Berücksichtige Gruppenlaufzeit von ca. L/2 T_S
	else
		tmax = 3 * L / f_S;
    end

    tmax = max(tmax, 3/f_sig);
	
	t=(0:1/f_S:1.5*tmax)'; % mehr Berechnen als Zeichnen, da am Ende der Werte 
							% der interpolierte Verlauf fehlerhaft ist
	x = sin(2*pi*F_sig*f_S/2*t);
	y = filter(bb,aa,x);% Gefiltert
	ymax = max(max(abs(y)),1);
	xd = [zeros(floor(L/2),1); x(1:length(x)-floor(L/2))]; %Verzögerungsausgleich
	%x um taug verzögert;
	plot(t,y,'--o');  hold on;
	plot(t,xd,'b-o');
	grid on;
	title('Eingangs- und Ausgangssignal (zeitdiskret)'); 
	xlabel(my_x_axis_t);
	ylabel('xd und y');
	legend('Ausgangssignal', 'um L/2 verzögertes Eingangssignal');
	axis([0 tmax -1.4*ymax 1.4*ymax]);
end

%=========================================
%% Filterantwort auf sinusförmiges Eingangssignal, 
% interpolierte = geglättete Darstellung
%=========================================
if SHOW_TRAN_SIN_INTP
	figure(9);
	I=12;%Interpolationsfaktor
	xdi=interp(xd,I);%interpol. Signal xd
	yi=interp(y,I);%interpol. Signal y
	ti=(0:1:length(xdi)-1)/(I*f_S)';
	plot(ti,yi,'--'); hold on;
	plot(ti,xdi,'b-','Linewidth',2); 
	grid on;
	title('Eingangs- und Ausgangssignal (interpoliert)'); 
	xlabel(my_x_axis_t);
	ylabel('xd und y');
	legend('Ausgangssignal', 'um L/2 verzögertes Eingangssignal' );
	axis([0 tmax -1.4*ymax 1.4*ymax]); 
end
