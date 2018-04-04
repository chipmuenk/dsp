%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ueb_LTI_F_system_properties.m
%
% Pol-Nullstellenplan, Amplitudengang, Phasengang, 
% 3D-Übertragungsfunktion etc. eines zeitdiskreten Systems,
% definiert über:
%			- Zähler- und Nennerkoeffizienten (Polynomform)
%			- oder Nullstellen ("Wurzeln") von Zähler und Nenner (Produktform)
% Getestet mit Octave 3.2.4 und Matlab R2011 und 2012b
% 
% (c) 2013 Christian Münker - Files zur Vorlesung "DSV für FPGAs"
%===============================================

close all; % close all plot windows
clear all; % clear all variables from workspace

prog_ver = version; % get version number of Matlab / Octave
OCTAVE = size(ver('Octave'),1); % returns '0' for Matlab

% Set default graphics properties
set(0,'DefaultAxesUnits','normalized'); 
set(0,'DefaultAxesColorOrder', [0.8 0 0.2; 0 1 0; 0 0 1], ...
          'DefaultAxesLineStyleOrder','-|--|:|-.');        

set(0,'DefaultLineLineWidth',2);
set(0,'DefaultAxesFontSize',16);
set(0,'DefaultTextFontSize',16);
PN_SIZE = 8; % Markersize for Poles / Zeroes

if OCTAVE 
    set(0,'DefaultAxesLineWidth',2);
    set(0,'DefaultTextFontName','Arial');
    set(0,'DefaultAxesFontName','Arial');
    set(0,'defaultLineMarkerSize',10);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Begin of "user defined part" -- edit below
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DEF_PRINT = 0;          % 1: Print Plots to PNG-Files
PRINT_PATH = ('D:/Daten/ueb2-IIR2-ML_add_1_'); % Path and base-name of Plot-Files
%
% Select which plots to show
SHOW_POLE_ZERO = 1;     % Pole-zero-plot
SHOW_LIN_H_f = 1;       % Linear plot of H(f)
SHOW_LOG_H_f = 1;       % Log. plot of H(f)
SHOW_PHASE = 1;         % Phase response
SHOW_GRPDELAY = 0;      % Group delay
SHOW_IMPZ = 1;          % Impulse response
SHOW_3D_H_z = 1;        % 3D plot of |H(z)| with poles and zeros
SHOW_3D_LIN_H_F = 1;    % 3D plot of |H(f)| with poles and zeros
SHOW_3D_LOG_H_F = 0;    % 3D plot of log |H(f)| with poles and zeros
SHOW_LIN_H_F_REP = 0;   % % Linear plot of H(f) (repeated spectra)
SHOW_3D_LIN_H_F_REP = 0; % 3D plot of |H(f)| with repeated spectra !!! SLOW !!!
%====================================================
f_S = 1000; % Sampling frequency, only effective for plotting options 'f_S/2' or 'f_S'
%==================================================
DEF_PN = 0; % 0: Transfer function from coefficients (polynomial form)
            % 1: Transfer function from poles / zeros (product form)
DEF_FIR = 0; % 0: IIR transfer function 
             % 1: FIR transfer function
ROTATE_FILTER = 0; 	% 0: Use filter as defined 
					% else: rotate poles/zeroes in multiples of pi;
					% set ROTATE_FILTER = 1 for LP <-> HP transform (e^j pi)
					% set ROTATE_FILTER = 0.5 for LP -> BP transform (e^j pi/2)


%---------------------------
% Define 2D-Plotting Options
%---------------------------
DEF_F_RANGE = 'F/2'; % select how to display the frequency axis:
%                       'F/2'   normalized frequency F = 0 ... 0.5 (f_S/2)
%                       'F'     normalized frequency F = 0 ... 1 (f_S)
%                       '+-F'   normalized frequency F = -0.5 ... 0.5 (f_S)
%                       'f_S/2' absolute frequency f = 0 ... f_S/2
%                       'f_S'   absolute frequency f = 0 ... f_S%
%                       '+-f_S'   absolute frequency f = -f_S/2 ... f_S/2%
%----
N_imp_response = 0; % select number of samples for impulse response, 
					% N_imp_response = 0 -> automatic scaling
N_FFT = 1024; % FFT-Size for freqz-plots
zmin_dB = -70; % lower limit of log. display for 2D and 3D Plot
%---------------------------
% Define 3D-Plotting Options
%---------------------------
POLAR_SPEC = 1; % Plot circular range in 3D-Plot
FORCE_ZMAX = 0; % Enforce absolute limit for 3D-Plot
PLOT_MESH = 0;  % 3D plot of H(z) as mesh instead of surface
%
steps = 80;               % number of steps for x, y, r, phi
rmin = 0;    rmax = 1.0;  % polar range definition
%
xmin = -1.0; xmax = 1.0;  % cartesian range definition
ymin = -1.0; ymax = 1.0;
%
zmin =  0.0; zmax = 10.0; % zmax-setting is only used when FORCE_ZMAX = 1
zmax_rel = 5; % Max. displayed z - value relative to max|H(f)|
%
plevel_rel = 1.05; % height of plotted pole position relative to zmax
zlevel_rel = 0.2; % height of plotted zero position relative to zmax
%
%===================================================================
% Definition of H(z) via polynomial coefficients (DEF_PN = 0) or 
%                    via poles / zeroes (DEF_PN = 1)
%
% Attention: Coefficients have to be specified as row vectors: [a1 a2 a3],
%                  Poles and zeroes as column vectors: [z_01; z_02; z_03]
%------------------------------------------------------------------------
if DEF_FIR
%------------------------------------------------------------------------    
% FIR - Filter
%
%==================== Definition by numerator coefficients b ============
%
% Various linear-phase filters:
b = [-0.07 0.57 0.57 -0.07];
b = [-0.2 -0.2 0.762 -0.038 -0.617 0.031 0.489]
b = [1 2 3 2 1]
%b = [0.25 0.5 0.25];
%b = [0.25 -0.5 0.25];
%b = [1 -1]; % differentiator
%b = [1 0 -1]; % comb filter, N = 2
%b = [1 0 0 0 0 0 0 0 1]; % "shifted comb" filter, N = 8
%b = [1 0 0 0 0 0 0 0 -1]; % comb filter, N = 8
%b = [1 0 0 0 0 0 -1]; % comb filter, N = 6
%b = [1 0.62 1]; % simple FIR notch filter at F = 0.3, defined by coeff.
%
% Non linear-phase filters:
%b = [0.25 0.5 0.75];
%b = [0.5 1];
%
% Moving average filters:
%b = [1 1 1 1 1]/5; % order 4
%b = ones(1,32)/32; % order 31
%
% Filters designed by Matlab with order 14 
% Attention: Coefficient vectors need to be transposed (') (only for Octave?)!
%b = remez(14, [0 0.3 0.7 1], [1 1 0 0])'; % Halfband-filter, F_DB = 0.15, F_SB = 0.35
%b = firls(14, [0 0.3 0.7 1], [1 1 0 0])'; %F_G = 0.25
%b = firls(3, [0 0.3 0.7 1], [1 0.7 0 0])';
%b = fir1(8, 0.5,'high', 'hann')'; % optional anderer Fenstertyp

% Optional: Create translated filter
% Optional: Create rotated filter
if ROTATE_FILTER
    phi_rot = ROTATE_FILTER * pi; % "rotate" filter 
    for k = 1:length(b)
        if ROTATE_FILTER == 1 % LP <-> HP translation {1 -1 1 ...}
            b(k)=b(k)*(-1)^(k-1); % treat separately for higher precision
        else 
            b(k)=b(k)*exp(j*phi_rot*(k-1)); % general translation
        end
    end
end

% Optional: Cascade filter twice by convolving (Falten) coefficients
%b = conv(b,b); 
a = [1, zeros(1,length(b)-1)]; % create same number of poles at origin (FIR)
%
%==================== Definition by nulls / gain k_0 ===========================
% 
% Simple FIR notch filter at F = 0.3 
k_0 = 1.0; % scale factor
%nulls = [exp(j*pi*0.6)];%  (complex-valued)
% nulls = [exp(j*pi*0.6); exp(-j*pi*0.6)];% same, but real-valued
%
% DIY-FIR-Filter by zero placement
% real-valued (conjugate-complex zero pairs):
%nulls = [-1; (-1+j)/sqrt(2); (-1-j)/sqrt(2); j; -j; 0.7; 1/0.7];
% complex-valued:
nulls = [-1; (-1+j)/sqrt(2); j; 0.7; 1/0.7];
nulls = [1.25*exp(j * 7* pi / 8); 1.25*exp(-j * 7*pi /8); ...
         1.25*exp(j * 6* pi / 8); 1.25*exp(-j * 6*pi /8); ...
        0.8*exp(j * 7* pi / 8); 0.8*exp(-j * 7*pi /8); ...
         0.8*exp(j * 6* pi / 8); 0.8*exp(-j * 6*pi /8); ...
        exp(j * 2* pi / 3); exp(-j * 2*pi /3); -j ; j; exp(j * pi / 3); exp(-j * pi / 3); 1];
%k_0 = 1.0;
%
%
%nulls = [nulls; nulls]; % optional: repeat zeros to cascade filter twice
poles = zeros(length(nulls),1); % FIR: create equal number of poles 
                                % at the origin to obtain causal filter
%--------------------------------------------------------------------------------
else % if DEF_FIR
%--------------------------------------------------------------------------------
% IIR - Filter
%
%==================== Definition by num. / denom. coefficients b, a ============
%
% Integrator
b = [1 +1 0] % z

%a = [1 1] % z - 1
a = [1 0 -0.64]
%b = [1 -1.2  1]/128;
%a = [1 -1.8 0.81];
%
% 4th order IIR-lowpass filters with F_DB = 0.25 (=f_S/4), = 0.5 in Matlab
%[b,a] = cheby1(4, 1, 0.5); % Chebychev Type 1 Filter: Passband ripple 1 dB
%[b,a] = cheby2(4, 40, 0.5); % Chebychev Type 2 Filter: Stopband ripple -40 dB
%[b,a] = ellip(4, 1, 40, 0.5); % Elliptic / Cauer Filter: Stopband / passband ripple
%
% BP-filter with two distinct peaks (bad design)
%b = [0.032 -0.053 0.047 -0.053 0.032]; % ZÃ¤hlerpolynom-Koeffizienten
%a = [1.0 -2.742 +3.735 -2.578 0.885]; % Nennerpolynom-Koeffizienten
%
% Allpass:
%k_0 = 0.8;
%b = [k_0 1]; % numerator zk_0 + 1
%a = [1 k_0]; % denominator  z + k_0
%
%==================== Definition by poles / nulls / gain k_0 =====================
% DIY - IIR-Filter
k_0 = 0.4;
poles = [(-1+1i); (-1-1i); -0.5]; % real valued (conj. complex poles)
%poles = [(-1+j)/1.7; ( + j)/1.2]; % complex-valued (asymm. poles) %; -0.5
nulls = [1 -1]; % or [0 0 0] - no difference
%
% Notch Filter
%k_0 = 0.95
%nulls = [exp(j*pi*0.6); exp(-j*pi*0.6)];% notch filter at F = 0.3
%nulls = [exp(j*pi*0.6)]; % single asymm. null
%poles = k_0 * nulls; % poles at same angle as zeroes but with r = 0.95
%
% Allpass
%k_0 = 0.8; 
%poles = [-k_0]; 
%nulls = [-1/k_0];

% Optional: Create rotated filter
if ROTATE_FILTER
	phi_rot = ROTATE_FILTER * pi; % "rotate" filter 
    for k = 1:length(b)
        if ROTATE_FILTER == 1 % LP <-> HP translation {1 -1 1 ...}
            b(k)=b(k)*(-1)^(k-1); % treat separately for higher precision
        else 
            b(k)=b(k)*exp(j*phi_rot*(k-1)); % general translation
        end
    end
	for k = 1:length(a)
		if ROTATE_FILTER == 1 % LP <-> HP translation {1 -1 1 ...}
		a(k)=a(k)*(-1)^(k-1); % treat separately for higher precision
		else 
		a(k)=a(k)*exp(j*phi_rot*(k-1)); % general translation
		end
	end
end
end % if DEF_FIR

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% End of "user defined part" -- changes below at own risk :-)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Calculate limits etc. for 3D-Plots
dr = rmax / steps * 2; dphi = pi / steps; % grid size for polar range
dx = (xmax - xmin) / steps;  dy = (ymax - ymin) / steps; % grid size cartesian range

% Set frequency range for 2D-plots according to user input
% Default: DEF_F_RANGE = 'f_S/2'
shift_F = 0;
half_range = 1;
PLOT_WHOLE_F = 0;
f_range = [0 f_S/2];
if strcmp(DEF_F_RANGE,'f_S')
    PLOT_WHOLE_F = 1;
    f_range = [0 f_S];
elseif strcmp(DEF_F_RANGE,'+-f_S')
    PLOT_WHOLE_F = 1;
    f_range = [-f_S/2 f_S/2];
    shift_F = 1;
elseif strcmp(DEF_F_RANGE,'f_S/2')
    half_range = 0.5;
elseif strcmp(DEF_F_RANGE,'F/2')
    PLOT_WHOLE_F = 0;
    f_S = 1;
    f_range = [0 0.5];
	half_range = 0.5;
elseif strcmp(DEF_F_RANGE,'F')
    PLOT_WHOLE_F = 1;
    f_S = 1;
    f_range = [0 1];
elseif strcmp(DEF_F_RANGE,'+-F')
    PLOT_WHOLE_F = 1;
    f_S = 1;
    f_range = [-0.5 0.5];
    shift_F = 1;
else
    warning('Invalid input for DEF_F_RANGE, using DEF_F_RANGE = ''f_S/2'' !');
	half_range = 0.5;
end 
%
% Define x-axis labels
if f_S == 1
    my_x_axis_f = sprintf('Norm. Frequenz [F bzw. \\Omega / 2 \\pi]');
    my_x_axis_t = sprintf('Sample n');
else  
    my_x_axis_f = sprintf('Frequenz [Hz]');
    my_x_axis_t = sprintf('Zeit [s]');
end

if DEF_PN
    % Calculate polynomes b and a from poles and zeros:
	% Due to rounding errors, b and a may have very small imaginary parts
	% that would be zero with a precise calculation. These "false imaginary" 
	% parts have to be eliminated as functions like H_mag cannot handle 
	% complex numbers
	b = k_0 * poly(nulls); % numerator = zeroes' polynome
	a = poly(poles); % denominator = poles' polynome
	tol = 1e-10;
        n  = abs(imag(b))<tol; % check whether imag. part is smaller than tol.
        b(n) = real(b(n)); % set all imaginary parts below tol. to zero
        n = abs(imag(a))<tol; % same for denominator
        a(n) = real(a(n)); % 
	%
	% TODO: P/Z at infinity have to be corrected! 

else 
	% Calculate poles and zeroes from the roots of denominator and numerator polynomes
	%
	nulls=roots(b); % Zeroes =  numerator roots
	poles=roots(a); % Poles =  denominator roots
end 
		
% H=tf(b,a);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pole-Zero-Plot
%===============================================================
if SHOW_POLE_ZERO
    figure(1);
    if ~OCTAVE
        if DEF_PN 
             [HZ, HP, HI] = zplane(nulls,poles); % Plot poles/zeros directly 
        else [HZ, HP, HI] = zplane(b,a);         % or from coefficient polynomes
        end
        set(HZ, 'Color', 'b', 'LineWidth', 2, 'MarkerSize', PN_SIZE); 
        set(HP, 'Color', 'r', 'LineWidth', 2, 'MarkerSize', PN_SIZE); 
        set(HI, 'Color', 'k', 'LineWidth', 1); % unit circle
    else
    % Octave doesn't support handles with zplane
        if DEF_PN 
             zplane(nulls,poles); % Plot poles/zeros directly 
        else zplane(b,a);         % or from coefficient polynomes
        end
    end
    xlabel('Real Part'); ylabel('Imaginary Part');
    grid on;
	if DEF_PRINT
		print (strcat(PRINT_PATH, 'pn.png'),'-dpng');
	end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Berechnung von |H(f)|
%===============================================================

if PLOT_WHOLE_F
	% Attention: Octave needs N_FFT argument
	[H,F]=freqz(b,a,N_FFT,'whole', f_S); % calculate H(f) along the whole unity circle
else
    [H,F]=freqz(b,a,N_FFT, f_S); % calculate H(f) along the upper half of unity circle
end
H_org = H;

if shift_F 
   H = fftshift(H); % display spectrum centered at F = 0
   F = F - f_S/2;
end

H_max = max(abs(H)); H_max_dB = 20*log10(H_max);
H_min = min(abs(H)); H_min_dB = 20*log10(H_min);
min_dB = floor(max(zmin_dB,H_min_dB)/10)*10;
%
%===============================================================
%% Plot lin |H(f)|
%===============================================================
if SHOW_LIN_H_f
    figure(2);
    plot(F,abs(H)); grid on;
    title('Frequenzgang von H (lin. Maßstab)');
    xlabel(my_x_axis_f); 
    ylabel('|H(f)|');
    axis ([f_range 0 10]); axis autoy;
		if DEF_PRINT
		  print (strcat(PRINT_PATH, 'Hf.png'),'-dpng');
		end
end

%===============================================================
%% Plot log |H(f)|
%===============================================================
if SHOW_LOG_H_f
    figure(3);
    plot(F,20*log10(abs(H))); grid on;
    title('Frequenzgang von H (log. Maßstab)');
    xlabel(my_x_axis_f );
    ylabel('20 log |H(f)|');
    axis ([f_range min_dB H_max_dB]); %axis autox;
		if DEF_PRINT
		  print (strcat(PRINT_PATH,'Hf_log.png'),'-dpng');
		end
end

%===============================================================
%% Plot phi(H(f))
%===============================================================
if SHOW_PHASE
    figure(4);
    %phasengang= unwrap(atan2(imag(H),real(H)))/pi;
    phasengang=unwrap(angle(H))/pi;
    plot(F,phasengang); grid on;
    title('Phasengang von H');
    axis 'auto';
    xlabel(my_x_axis_f);
    ylabel('\phi_H (f) (rad / \pi)');
    axis ([f_range 0 3]); axis autoy;
		if DEF_PRINT
		  print (strcat(PRINT_PATH,'H_phi.png'),'-dpng');
		end
end

%===============================================================
%% Plot Group Delay
%===============================================================
if SHOW_GRPDELAY
    figure(5);
    if PLOT_WHOLE_F
	    [tau_g,w] = grpdelay(b,a,N_FFT,'whole'); 
    else
      [tau_g,w] = grpdelay(b,a,N_FFT); 
	  end
	  F = w/2/pi*f_S;
		if shift_F 
		   tau_g = fftshift(tau_g); % display spectrum centered at F = 0
		   F = F - f_S/2;
		end
    plot(F, tau_g); grid on;
    title('Group Delay \tau_g');
    axis 'auto';
    xlabel(my_x_axis_f );
    ylabel('\tau_g(f)/T_S');
    axis ([f_range 0 3]); axis autoy;
	  if DEF_PRINT
		  print (strcat(PRINT_PATH,'H_grp.png'),'-dpng');
		end
end

%===============================================================
%% Plot impulse response
%===============================================================
if SHOW_IMPZ
    figure(6);
	if N_imp_response
		[himp,t]=impz(b,a,N_imp_response,f_S);
	else
		[himp,t]=impz(b,a,[],f_S);
	end
    if (imag(himp) == 0)
       h1 = stem(t,himp);
	   set(h1,'Markersize',3,'Linewidth',2);
	   title('Impulsantwort von H');
       ylabel('h[n]');
       xlabel(my_x_axis_t);
       grid on;
    else
        subplot(2,1,1);
        h2 = stem(t,real(himp));
		set(h2,'Markersize',3,'Linewidth',2);
        title('Komplexe Impulsantwort von H');
        ylabel('re\{h[n]\}');
        grid on;
        subplot(2,1,2);
        h3 = stem(t,imag(himp));
		set(h3,'Markersize',3,'Linewidth',2,'color','b');
       ylabel('im\{h[n]\}');
       xlabel(my_x_axis_t);	
    grid on; 
    end
    if DEF_PRINT
        print (strcat(PRINT_PATH,'Himp.png'),'-dpng');
    end
end

%===============================================================
%% 3D-Surface Plot of |H(z)|
%===============================================================
if FORCE_ZMAX
    thresh = zmax
else
    thresh = zmax_rel * H_max; % calculate display thresh. from max. of H(f)
end

plevel = plevel_rel * thresh; % height of displayed pole position
zlevel = zlevel_rel * thresh; % height of displayed zero position

if POLAR_SPEC
    [r, phi] = meshgrid(rmin:dr:rmax, 0:dphi:2*pi); % polar grid
    [x, y] = pol2cart(phi,r); % x = r.*cos(phi); y = r.*sin(phi);
else
    [x,y] = meshgrid(xmin:dx:xmax, ymin:dy:ymax); % cartesian grid
end
z = x + j*y; % create coordinates for complex plane

phi_EK = linspace(0,2*pi,200); % 200 points from 0 ... 2 pi
xy_EK = exp(j*phi_EK); % calculate coordinates for unity circle
H_EK = H_mag(b,a,xy_EK,thresh); %|H| along the unity circle

if SHOW_3D_H_z
    figure(7);
    %
    colormap gray;  %hsv / gray / default / colorcube / bone 
    if PLOT_MESH
        g = meshz(x,y, H_mag(b,a,z,thresh)); %plot 3D-mesh of |H(z)| ; limit at |H(z)| = thresh
        %Alternatives:	mesh(c) - 3D-mesh of |H(z)| (combined with contour-plot) ; 
        %				meshz - 3D-mesh of |H(z)| with "curtains" at the rim
        % hidden off; % plot hidden lines
        set(g,'EdgeColor',[.4 .4 .4]); % medium gray color for mesh
    else
        g = surfl(x,y, H_mag(b,a,z,thresh)); %plot 3D-surface of |H(z)| ; limit at |H(z)| = thresh
        %Alternatives surf(l,c) - 3D-surface (with lighting / contourplot)
        % set(g,'FaceColor',[.4 .4 .4]);
        % set(g,'FaceColor','none'); % invisible surfaces
        % set(g,'FaceAlpha',0.8); % define transparency
        %
        %set(g,'EdgeColor',[.8 .8 .8]); % light gray color for edges between surfaces
        %set(g,'EdgeColor','none'); % invisible edges between surfaces
        %
        shading interp; % % interpolate surface elements (smooth surface)
        %shading facet; % show surface elements as facets
        %shading flat; % constant color for each surface element

        if ~OCTAVE
            lighting gouraud; %phong (slowest) / gouraud / flat / none
            material dull; %metal / shiny / dull
            camlight left;
        end;
    end;
    %  
    grid on;
    hold on; 
    % Plot 1.01*|H(f)| along the unity circle (|H(f)| < thresh),
    % increase value by 1% to improve visibility:
    plot3(real(xy_EK), imag(xy_EK), H_EK*1.01, 'Linewidth',2);
    % Plot unit circle:
    plot3(real(xy_EK), imag(xy_EK), zeros(1,length(xy_EK)), 'Color','k','Linewidth',2);
    % Plot the zeros at (x,y,0) with "stems":
    plot3(real(nulls), imag(nulls), ones(1,length(nulls))*zlevel, 'o', ...
         'Color','b','Markersize',PN_SIZE, 'Linewidth',2);
    for k = 1:length(nulls)
          line([real(nulls(k)) real(nulls(k))],[imag(nulls(k)) imag(nulls(k))],...
          [0 zlevel],'Linewidth',1, 'Color', 'b');
    end;
    % Plot the poles at |H(z_p)| = plevel with "stems"
    plot3(real(poles), imag(poles), H_mag(b,a,poles,plevel), 'x', 'Markersize',PN_SIZE,...
        'Linewidth',2);
    axis([xmin xmax ymin ymax zmin plevel]); 
    for k = 1:length(poles)
          line([real(poles(k)) real(poles(k))],[imag(poles(k)) imag(poles(k))],...
          [0 plevel],'Linewidth',1,'Color','r');
    end;
    title('3D-Darstellung von |H(z)|');
    xlabel('Re');
    ylabel('Im');
	  if DEF_PRINT
		  print (strcat(PRINT_PATH,'Hz_3d.png'),'-dpng');
		end
    hold off;
end;

%===============================================================
%% 3D-Plot of |H(f)| - linear scale
%===============================================================
if SHOW_3D_LIN_H_F
    figure(8);
	%plot ||H(f)| as line
    plot3(real(xy_EK), imag(xy_EK), H_EK, 'Linewidth',2); 
    hold on;
	%plot ||H(f)| as stems
    h=stem3(real(xy_EK), imag(xy_EK), H_EK ,'k'); 
    set(h,'marker','none','Linewidth',1); % remove stem markers
    plot3(real(xy_EK), imag(xy_EK), zeros(1,length(xy_EK)), 'k','Linewidth',2); % plot EK
    plot3(real(nulls), imag(nulls), zeros(1,length(nulls)), ...
        'o', 'Color', 'b', 'Markersize',PN_SIZE, 'Linewidth',2); % plot nulls
    plot3(real(poles), imag(poles), zeros(1,length(poles)),...
      'x', 'Markersize',PN_SIZE, 'Linewidth',2); % plot poles
    axis([xmin xmax ymin ymax 0 H_max*1.1]);
    title('3D-Darstellung von |H(j\Omega)|');
    xlabel('Re');
    ylabel('Im');
    grid on;
	  %set(gca, 'XTickLabel','','YTickLabel','','ZTickLabel',''); % turn off tick labels
		set(gca,'view',[35 20]); % set viewing angle
	  if DEF_PRINT
		  print (strcat(PRINT_PATH,'Hf_3d.png'),'-dpng');
		end
    hold off;
end

%===============================================================
%% 3D-Plot of log |H(f)|
%===============================================================
if SHOW_3D_LOG_H_F
    figure(9);
    % Plot log|H(f)|:
    plot3(real(xy_EK), imag(xy_EK), max(20*log10(H_EK), min_dB), 'Linewidth',2);
    hold on;
    % Plot thin vertical lines:
    h=stem3(real(xy_EK), imag(xy_EK), max(20*log10(H_EK), min_dB),'k');
    set(h,'marker','none','Linewidth',1);
    % Plot unit circle:
    plot3(real(xy_EK), imag(xy_EK), zeros(1,length(xy_EK)), 'Color', 'k','Linewidth',2);
    plot3(real(nulls), imag(nulls), zeros(1,length(nulls)), 'o', 'Color', 'b', 'Markersize',PN_SIZE, 'Linewidth',2);
    plot3(real(poles), imag(poles), zeros(1,length(poles)), 'x', 'Markersize',PN_SIZE, 'Linewidth',2);
    axis([xmin xmax ymin ymax min_dB max(0,H_max_dB)]);
    title('3D-Darstellung von 20 log |H(j\Omega)| in dB');
    xlabel('Re');
    ylabel('Im');
    grid on;
	  if DEF_PRINT
		  print (strcat(PRINT_PATH,'Hf_log_3d.png'),'-dpng');
		end
    hold off;
end

%===============================================================
%% Plot lin |H(f)| (repeated spectra)
%===============================================================
if SHOW_LIN_H_F_REP
    figure(10);
    % repeated spectra always need whole range of F, H
    [H,F]=freqz(b,a,N_FFT,'whole', f_S); 
    N_rep = 3; % Anzahl der Wiederholungen
    f_range_rep = f_range * N_rep;
    N_rep = ceil(N_rep / 2) * 2; % NÃ¤chste gerade Zahl, sonst wird
	                         % Spektrum bei shift_F = 1 um F = 1/2 verschoben
    f_rep = f_S*linspace(0,N_rep,N_rep*length(H)); % 
     if shift_F 
        H = fftshift(H); % center spectrum at f = 0
        f_rep = f_rep - N_rep/2 * f_S - f_S/2;
        F = F - f_S/2;
     end
     H_rep = repmat(H,N_rep,1); % Repeat spectrum N_rep times
	 
% Das Spektrum jenseits von f_S muss fÃ¼r die Darstellung kÃ¼nstlich wiederholt 
% die DFT, freqz etc. liefern nur Spektrumspunkte bis f_S
    plot(f_rep,abs(H_rep),'color',[.5 .5 .5]); grid on; hold on;
	plot((F(1:length(H)* half_range)), abs(H(1:length(H)*half_range)),'linewidth',3);

    title('Frequenzgang von H (lin. MaÃŸstab)');
    xlabel(my_x_axis_f); 
    ylabel('|H(f)|');
    axis ([f_range_rep 0 10]); axis autoy;
	y_max = get(gca, 'ylim');
    plot([f_range(1) f_range(1)]/half_range,[0 y_max(2)],'b:'); % Grenzen des Basisbands
    plot([f_range(2) f_range(2)]/half_range,[0 y_max(2)],'b:'); % Grenzen des Basisbands
	if DEF_PRINT
        print (strcat(PRINT_PATH, 'Hf_rep.png'),'-dpng');
	end
	hold off;
end % if SHOW_LIN_H_F_REP

%===============================================================
%% 3D-Plot of |H(f)| (repeated spectra)
%===============================================================
if SHOW_3D_LIN_H_F_REP
    figure(11);
	N_rep = 3; D_rep = 2; % number of stacked z-planes and distance
	
	H_EK_rep = repmat(H_EK, 1, N_rep);
	phi_EK = linspace(0,2*pi*N_rep,200*N_rep); % 200N points from 0 ... N 
	l = phi_EK/(2*pi);
	xy_EK = exp(j*phi_EK); % calculate coordinates for unity circle
	z_EK = (atan(500*(l-round(l)))/pi + round(l) - atan(500*l)/pi)*D_rep;
	plot3(real(xy_EK),imag(xy_EK),z_EK, 'k','Linewidth',3 ); 
	hold on;
	plot3([1 1], [0 0], [0 N_rep * D_rep],'Linewidth',2, 'color', [0.5 0.5 0.5]);
	plot3(real(nulls), imag(nulls), zeros(1,length(nulls)), ...
        'o', 'Color', 'b', 'Markersize',PN_SIZE, 'Linewidth',2); % plot nulls
	hold on;
    plot3(real(poles), imag(poles), zeros(1,length(poles)),...
      'x', 'Markersize',PN_SIZE, 'Linewidth',2); % plot poles
    axis([xmin xmax ymin ymax 0 N_rep*D_rep]);
    title('Wiederholspektren von |H(j\Omega)|');
    xlabel('Re');
    ylabel('Im');
	  zlabel(' |H(j\Omega)| ');

	plot3(real(xy_EK), imag(xy_EK), H_EK_rep + z_EK, 'Linewidth',2); 
	
    for k=1:length(xy_EK)	
		plot3([real(xy_EK(k)) real(xy_EK(k))], [imag(xy_EK(k)) imag(xy_EK(k))], ...
		   [z_EK(k) H_EK_rep(k) + z_EK(k)],'color',[0.5 0.5 0.5], 'Linewidth', 1); 
    end
	set(gca, 'ZTickLabelMode', 'manual', 'ZTickLabel', []); % no numbers @ z-axis
	%for k = 1:N_rep
		%t = linspace(0,2*pi,100);
		%patch(cos(t),sin(t),0,'r'); 
	%endfor
    grid on;
		if DEF_PRINT
		  print (strcat(PRINT_PATH,'Hz_3d_rep.png'),'-dpng');
		end
    hold off;
end % if SHOW_3D_LIN_H_F_REP

                                             
