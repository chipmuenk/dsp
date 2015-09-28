% Calculate magnitude of H(z) in polynomial form
%

function mag = H_mag(zaehler, nenner, z, lim)
mag = min(abs(polyval(zaehler,z))./abs(polyval(nenner,z)),lim);

