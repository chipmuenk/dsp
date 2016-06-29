%===============================================
% plots_3D_m.m
%
% Beispiel für 3D-Surface Plot in Matlab
% 
% (c) 2013 Christian Münker - Files zur Vorlesung "DSV auf FPGAs"
%===============================================
clear all; clf;
x = -5:.25:5; y = x;
[x,y] = meshgrid(x);
R = sqrt(x.^2 + y.^2);
Z = sin(R);
surf(x,y,Z,gradient(Z));