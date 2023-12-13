# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 10:29:42 2012
dsp_fpga_lib (Version 8)
@author: Muenker_2
"""
import sys
import string # needed for remezord?
import numpy as np
import numpy.ma as ma
from numpy import pi, asarray, absolute, sqrt, log10, arctan, ceil, hstack, mod

import scipy.signal as sig
from scipy import __version__ as sci_version
from scipy import special # needed for remezord
import scipy.spatial.distance as sc_dist
import matplotlib as mpl
import matplotlib.pyplot as plt
from  matplotlib import patches

__version__ = "0.9"
def versions():
    print("Python version:", ".".join(map(str, sys.version_info[:3])))
    print("Numpy:", np.__version__)
    print("Scipy:", sci_version)
    print("Matplotlib:", mpl.__version__, mpl.get_backend())

mpl_rc = {'lines.linewidth'           : 1.5,
          'lines.markersize'          : 8,  # markersize in points
          'text.color'                : 'black',
          'font.family'               : 'sans-serif',#'serif',
          'font.style'                : 'normal',
          #'mathtext.fontset'          : 'stixsans',#'stix',
          #'mathtext.fallback_to_cm'   : True,
          'mathtext.default'          : 'it',
          'font.size'                 : 12, 
          'legend.fontsize'           : 12, 
          'axes.labelsize'            : 12, 
          'axes.titlesize'            : 14, 
          'axes.linewidth'            : 1, # linewidth for coordinate system
          'axes.formatter.use_mathtext': True, # use mathtext for scientific notation.
          #
          'axes.grid'                 : True,
          'grid.color'                : '#222222',
          'axes.facecolor'            : 'white',
          'axes.labelcolor'           : 'black',
          'axes.edgecolor'            : 'black',
          'grid.linestyle'            : ':',
          'grid.linewidth'            : 0.5,
          #
          'xtick.top'                 : False,     # mpl >= 2.0
          'xtick.direction'           : 'out',
          'ytick.direction'           : 'out',
          'xtick.color'               : 'black',
          'ytick.color'               : 'black',
          #
          'figure.figsize'            : (7,4), # default figure size in inches
          'figure.facecolor'          : 'white',
          'figure.edgecolor'          : '#808080',
          'figure.dpi'                : 100,
          'savefig.dpi'               : 100,
          'savefig.facecolor'         : 'white',
          'savefig.edgecolor'         : 'white',
          'savefig.bbox'              : 'tight',
          'savefig.pad_inches'        : 0,
          'hatch.color'               : '#808080', # mpl >= 2.0
          'hatch.linewidth'           : 0.5,       # mpl >= 2.0
          'animation.html'          : 'jshtml'   # javascript, mpl >= 2.1 
           } 

mpl_rc_33 = {'mathtext.fallback'      : 'cm'} # new since mpl 3.3
mpl_rc_32 = {'mathtext.fallback_to_cm': True} # deprecated since mpl 3.3

if mpl.__version__ < "3.3":
    mpl_rc.update(mpl_rc_32) # lower than matplotlib 3.3
else:
    mpl_rc.update(mpl_rc_33)
          
plt.rcParams.update(mpl_rc) # define plot properties 

def H_mag(zaehler, nenner, z, lim):
    """ Calculate magnitude of H(z) or H(s) in polynomial form at the complex
    coordinate z = x, 1j * y (skalar or array)
    The result is clipped at lim."""
#    limvec = lim * np.ones(len(z))
    try: len(zaehler)
    except TypeError:
        z_val = abs(zaehler) # zaehler is a scalar
    else:
        z_val = abs(np.polyval(zaehler,z)) # evaluate zaehler at z
    try: len(nenner)
    except TypeError:
        n_val = nenner # nenner is a scalar
    else:
        n_val = abs(np.polyval(nenner,z))

    return np.minimum((z_val/n_val),lim)


#----------------------------------------------
# from scipy.sig.signaltools.py:
def cmplx_sort(p):
    "sort roots based on magnitude."
    p = np.asarray(p)
    if np.iscomplexobj(p):
        indx = np.argsort(abs(p))
    else:
        indx = np.argsort(p)
    return np.take(p, indx, 0), indx

# adapted from scipy.signal.signaltools.py:
# TODO:  comparison of real values has several problems (5 * tol ???)
def unique_roots(p, tol=1e-3, magsort = False, rtype='min', rdist='euclidian'):
    """
Determine unique roots and their multiplicities from a list of roots.

Parameters
----------
p : array_like
    The list of roots.
tol : float, default tol = 1e-3
    The tolerance for two roots to be considered equal. Default is 1e-3.
magsort: Boolean, default = False
    When magsort = True, use the root magnitude as a sorting criterium (as in
    the version used in numpy < 1.8.2). This yields false results for roots
    with similar magniudes (e.g. on the unit circle) but is signficantly
    faster for a large number of roots (factor 20 for 500 double roots.)
rtype : {'max', 'min, 'avg'}, optional
    How to determine the returned root if multiple roots are within
    `tol` of each other.
    - 'max' or 'maximum': pick the maximum of those roots (magnitude ?).
    - 'min' or 'minimum': pick the minimum of those roots (magnitude?).
    - 'avg' or 'mean' : take the average of those roots.
    - 'median' : take the median of those roots
dist : {'manhattan', 'euclid'}, optional
    How to measure the distance between roots: 'euclid' is the euclidian
    distance. 'manhattan' is less common, giving the
    sum of the differences of real and imaginary parts.

Returns
-------
pout : list
    The list of unique roots, sorted from low to high (only for real roots).
mult : list
    The multiplicity of each root.

Notes
-----
This utility function is not specific to roots but can be used for any
sequence of values for which uniqueness and multiplicity has to be
determined. For a more general routine, see `numpy.unique`.

Examples
--------
>>> vals = [0, 1.3, 1.31, 2.8, 1.25, 2.2, 10.3]
>>> uniq, mult = sp.signal.unique_roots(vals, tol=2e-2, rtype='avg')

Check which roots have multiplicity larger than 1:

>>> uniq[mult > 1]
array([ 1.305])

Find multiples of complex roots on the unit circle:
>>> vals = np.roots(1,2,3,2,1)
uniq, mult = sp.signal.unique_roots(vals, rtype='avg')

"""

    def manhattan(a,b):
        """
        Manhattan distance between a and b
        """
        return ma.abs(a.real - b.real) + ma.abs(a.imag - b.imag)

    def euclid(a,b):
        """
        Euclidian distance between a and b
        """
        return ma.abs(a - b)

    if rtype in ['max', 'maximum']:
        comproot = ma.max  # nanmax ignores nan's
    elif rtype in ['min', 'minimum']:
        comproot = ma.min  # nanmin ignores nan's
    elif rtype in ['avg', 'mean']:
        comproot = ma.mean # nanmean ignores nan's
#    elif rtype == 'median':
    else:
        raise TypeError(rtype)

    if rdist in ['euclid', 'euclidian']:
        dist_roots = euclid
    elif rdist in ['rect', 'manhattan']:
        dist_roots = manhattan
    else:
        raise TypeError(rdist)

    mult = [] # initialize list for multiplicities
    pout = [] # initialize list for reduced output list of roots
    p = np.atleast_1d(p) # convert p to at least 1D array
    tol = abs(tol)

    if len(p) == 0:  # empty argument, return empty lists
        return pout, mult

    elif len(p) == 1: # scalar input, return arg with multiplicity = 1
        pout = p
        mult = [1]
        return pout, mult

    else:
        sameroots = [] # temporary list for roots within the tolerance
        pout = p[np.isnan(p)].tolist() # copy nan elements to pout as list
        mult = len(pout) * [1] # generate a list with a "1" for each nan
        #p = ma.masked_array(p[~np.isnan(p)]) # delete nan elements, convert to ma
        p = np.ma.masked_where(np.isnan(p), p) # only masks nans, preferrable?

    if np.iscomplexobj(p) and not magsort:

        for i in range(len(p)): # p[i] is current root under test
            if not p[i] is ma.masked: # has current root been "deleted" yet?
                tolarr = dist_roots(p[i], p[i:]) < tol # test against itself and
                # subsequent roots, giving a multiplicity of at least one
                mult.append(np.sum(tolarr)) # multiplicity = number of "hits"
                sameroots = p[i:][tolarr]   # pick the roots within the tolerance
                p[i:] = ma.masked_where(tolarr, p[i:]) # and "delete" (mask) them
                pout.append(comproot(sameroots)) # avg/mean/max of mult. root

    else:
        p,indx = cmplx_sort(p)
        indx = -1
        curp = p[0] + 5 * tol # needed to avoid "self-detection" ?
        for k in range(len(p)):
            tr = p[k]
#            if dist_roots(tr, curp) < tol:
            if abs(tr - curp) < tol:
                sameroots.append(tr)
                curp = comproot(sameroots)  # not correct for 'avg'
                                            # of multiple (N > 2) root !
                pout[indx] = curp
                mult[indx] += 1
            else:
                pout.append(tr)
                curp = tr
                sameroots = [tr]
                indx += 1
                mult.append(1)

    return np.array(pout), np.array(mult)

##### original code ####
#    p = asarray(p) * 1.0
#    tol = abs(tol)
#    p, indx = cmplx_sort(p)
#    pout = []
#    mult = []
#    indx = -1
#    curp = p[0] + 5 * tol
#    sameroots = []
#    for k in range(len(p)):
#        tr = p[k]
#        if abs(tr - curp) < tol:
#            sameroots.append(tr)
#            curp = comproot(sameroots)
#            pout[indx] = curp
#            mult[indx] += 1
#        else:
#            pout.append(tr)
#            curp = tr
#            sameroots = [tr]
#            indx += 1
#            mult.append(1)
#    return array(pout), array(mult)

#------------------------------------------------------------------------------


def zplane(b=None, a=1, z=None, p=None, k=1,  pn_eps=1e-3, analog=False,
          plt_ax = None, plt_poles=True, style='square', anaCircleRad=0, lw=2,
          mps = 10, mzs = 10, mpc = 'r', mzc = 'b', plabel = '', zlabel = ''):
    """
    Plot the poles and zeros in the complex z-plane either from the
    coefficients (`b,`a) of a discrete transfer function `H`(`z`) (zpk = False)
    or directly from the zeros and poles (z,p) (zpk = True).

    When only b is given, an FIR filter with all poles at the origin is assumed.

    Parameters
    ----------
    b :  array_like
         Numerator coefficients (transversal part of filter)
         When b is not None, poles and zeros are determined from the coefficients
         b and a

    a :  array_like (optional, default = 1 for FIR-filter)
         Denominator coefficients (recursive part of filter)

    z :  array_like, default = None
         Zeros
         When b is None, poles and zeros are taken directly from z and p

    p :  array_like, default = None
         Poles

    analog : boolean (default: False)
        When True, create a P/Z plot suitable for the s-plane, i.e. suppress
        the unit circle (unless anaCircleRad > 0) and scale the plot for
        a good display of all poles and zeros.

    pn_eps : float (default : 1e-2)
         Tolerance for separating close poles or zeros

    plt_ax : handle to axes for plotting (default: None)
        When no axes is specified, the current axes is determined via plt.gca()

    plt_poles : Boolean (default : True)
        Plot poles. This can be used to suppress poles for FIR systems
        where all poles are at the origin.

    style : string (default: 'square')
        Style of the plot, for style == 'square' make scale of x- and y-
        axis equal.

    mps : integer  (default: 10)
        Size for pole marker

    mzs : integer (default: 10)
        Size for zero marker

    mpc : char (default: 'r')
        Pole marker colour

    mzc : char (default: 'b')
        Zero marker colour

    lw : integer (default:  2)
        Linewidth for unit circle

    plabel, zlabel : string (default: '')
        This string is passed to the plot command for poles and zeros and
        can be displayed by legend()


    Returns
    -------
    z, p, k : ndarray


    Notes
    -----
    """
    # TODO:
    # - polar option
    # - add keywords for color of circle -> **kwargs
    # - add option for multi-dimensional arrays and zpk data

    # make sure that all inputs are arrays
    b = np.atleast_1d(b)
    a = np.atleast_1d(a)
    z = np.atleast_1d(z) # make sure that p, z  are arrays
    p = np.atleast_1d(p)

    if b.any(): # coefficients were specified
        if len(b) < 2 and len(a) < 2:
            logger.error('No proper filter coefficients: both b and a are scalars!')
            return z, p, k

        # The coefficients are less than 1, normalize the coefficients
        if np.max(b) > 1:
            kn = np.max(b)
            b = b / float(kn)
        else:
            kn = 1.

        if np.max(a) > 1:
            kd = np.max(a)
            a = a / abs(kd)
        else:
            kd = 1.

        # Calculate the poles, zeros and scaling factor
        p = np.roots(a)
        z = np.roots(b)
        k = kn/kd
    elif not (len(p) or len(z)): # P/Z were specified
        logger.error('Either b,a or z,p must be specified!')
        return z, p, k

    # find multiple poles and zeros and their multiplicities
    if len(p) < 2: # single pole, [None] or [0]
        if not p or p == 0: # only zeros, create equal number of poles at origin
            p = np.array(0,ndmin=1) #
            num_p = np.atleast_1d(len(z))
        else:
            num_p = [1.] # single pole != 0
    else:
        #p, num_p = sig.signaltools.unique_roots(p, tol = pn_eps, rtype='avg')
        p, num_p = unique_roots(p, tol = pn_eps, rtype='avg')
#        p = np.array(p); num_p = np.ones(len(p))
    if len(z) > 0:
        z, num_z = unique_roots(z, tol = pn_eps, rtype='avg')
#        z = np.array(z); num_z = np.ones(len(z))
        #z, num_z = sig.signaltools.unique_roots(z, tol = pn_eps, rtype='avg')
    else:
        num_z = []

    ax = plt_ax#.subplot(111)
    if analog == False:
        # create the unit circle for the z-plane
        uc = patches.Circle((0,0), radius=1, fill=False,
                            color='grey', ls='solid', zorder=1)
        ax.add_patch(uc)
        if style == 'square':
            #r = 1.1
            #ax.axis([-r, r, -r, r]) # overridden by next option
            ax.axis('equal')
    #    ax.spines['left'].set_position('center')
    #    ax.spines['bottom'].set_position('center')
    #    ax.spines['right'].set_visible(True)
    #    ax.spines['top'].set_visible(True)

    else: # s-plane
        if anaCircleRad > 0:
            # plot a circle with radius = anaCircleRad
            uc = patches.Circle((0,0), radius=anaCircleRad, fill=False,
                                color='grey', ls='solid', zorder=1)
            ax.add_patch(uc)
        # plot real and imaginary axis
        ax.axhline(lw=2, color = 'k', zorder=1)
        ax.axvline(lw=2, color = 'k', zorder=1)

    # Plot the zeros
    ax.scatter(z.real, z.imag, s=mzs*mzs, zorder=2, marker = 'o',
               facecolor = 'none', edgecolor = mzc, lw = lw, label=zlabel)
    # and print their multiplicity
    for i in range(len(z)):
        if num_z[i] > 1:
            ax.text(np.real(z[i]), np.imag(z[i]),'  (' + str(num_z[i]) +')',
                            va = 'top', color=mzc)
    if plt_poles:
        # Plot the poles
        ax.scatter(p.real, p.imag, s=mps*mps, zorder=2, marker='x',
                   color=mpc, lw=lw, label=plabel)
        # and print their multiplicity
        for i in range(len(p)):
            if num_p[i] > 1:
                ax.text(np.real(p[i]), np.imag(p[i]), '  (' + str(num_p[i]) +')',
                                va = 'bottom', color=mpc)

# =============================================================================
#            # increase distance between ticks and labels
#            # to give some room for poles and zeros
#         for tick in ax.get_xaxis().get_major_ticks():
#             tick.set_pad(12.)
#             tick.label1 = tick._get_text1()
#         for tick in ax.get_yaxis().get_major_ticks():
#             tick.set_pad(12.)
#             tick.label1 = tick._get_text1()
#
# =============================================================================
    xl = ax.get_xlim(); Dx = max(abs(xl[1]-xl[0]), 0.05)
    yl = ax.get_ylim(); Dy = max(abs(yl[1]-yl[0]), 0.05)
    ax.set_xlim((xl[0]-Dx*0.05, max(xl[1]+Dx*0.05,0)))
    ax.set_ylim((yl[0]-Dy*0.05, yl[1] + Dy*0.05))

    return z, p, k

#------------------------------------------------------------------------------
def zplane_bak(b=None, a=1, z=None, p=None, k=1,  pn_eps=1e-3, analog=False,
              plt_ax = None, plt_poles=True, style='square', anaCircleRad=0, lw=2,
              mps = 10, mzs = 10, mpc = 'r', mzc = 'b', plabel = '', zlabel = ''):
        """
        Plot the poles and zeros in the complex z-plane either from the
        coefficients (`b,`a) of a discrete transfer function `H`(`z`) (b specified)
        or directly from the zeros and poles (z,p specified).

        When only b is given, an FIR filter with all poles at the origin is assumed.

        Parameters
        ----------
        b :  array_like
             Numerator coefficients (transversal part of filter)
             When b is not None, poles and zeros are determined from the coefficients
             b and a

        a :  array_like (optional, default = 1 for FIR-filter)
             Denominator coefficients (recursive part of filter)

        z :  array_like, default = None
             Zeros
             When b is None, poles and zeros are taken directly from z and p

        p :  array_like, default = None
             Poles

        analog : boolean (default: False)
            When True, create a P/Z plot suitable for the s-plane, i.e. suppress
            the unit circle (unless anaCircleRad > 0) and scale the plot for
            a good display of all poles and zeros.

        pn_eps : float (default : 1e-2)
             Tolerance for separating close poles or zeros

        plt_ax : handle to axes for plotting (default: None)
            When no axes is specified, the current axes is determined via plt.gca()

        plt_poles : Boolean (default : True)
            Plot poles. This can be used to suppress poles for FIR systems
            where all poles are at the origin.

        style : string (default: 'square')
            Style of the plot, for style == 'square' make scale of x- and y-
            axis equal.

        mps : integer  (default: 10)
            Size for pole marker

        mzs : integer (default: 10)
            Size for zero marker

        mpc : char (default: 'r')
            Pole marker colour

        mzc : char (default: 'b')
            Zero marker colour

        lw : integer (default:  2)
            Linewidth for unit circle

        plabel, zlabel : string (default: '')
            This string is passed to the plot command for poles and zeros and
            can be displayed by legend()


        Returns
        -------
        z, p, k : ndarray


        Notes
        -----
        """
        # TODO:
        # - polar option
        # - add keywords for color of circle -> **kwargs
        # - add option for multi-dimensional arrays and zpk data

        # make sure that all inputs are arrays
        b = np.atleast_1d(b)
        a = np.atleast_1d(a)
        z = np.atleast_1d(z) # make sure that p, z  are arrays
        p = np.atleast_1d(p)

        if b.any(): # coefficients were specified
            if len(b) < 2 and len(a) < 2:
                logger.error('No proper filter coefficients: both b and a are scalars!')
                return z, p, k

            # The coefficients are less than 1, normalize the coefficients
            if np.max(b) > 1:
                kn = np.max(b)
                b = b / float(kn)
            else:
                kn = 1.

            if np.max(a) > 1:
                kd = np.max(a)
                a = a / abs(kd)
            else:
                kd = 1.

            # Calculate the poles, zeros and scaling factor
            p = np.roots(a)
            z = np.roots(b)
            k = kn/kd
        elif not (len(p) or len(z)): # P/Z were specified
            print('Either b,a or z,p must be specified!')
            return z, p, k

        # find multiple poles and zeros and their multiplicities
        if len(p) < 2: # single pole, [None] or [0]
            if not p or p == 0: # only zeros, create equal number of poles at origin
                p = np.array(0,ndmin=1) #
                num_p = np.atleast_1d(len(z))
            else:
                num_p = [1.] # single pole != 0
        else:
            #p, num_p = sig.signaltools.unique_roots(p, tol = pn_eps, rtype='avg')
            p, num_p = unique_roots(p, tol = pn_eps, rtype='avg')
    #        p = np.array(p); num_p = np.ones(len(p))
        if len(z) > 0:
            z, num_z = unique_roots(z, tol = pn_eps, rtype='avg')
    #        z = np.array(z); num_z = np.ones(len(z))
            #z, num_z = sig.signaltools.unique_roots(z, tol = pn_eps, rtype='avg')
        else:
            num_z = []

        if not plt_ax:
            ax = plt.gca()# fig.add_subplot(111)
        else:
            ax = plt_ax
            
        if analog == False:
            # create the unit circle for the z-plane
            uc = patches.Circle((0,0), radius=1, fill=False,
                                color='grey', ls='solid', zorder=1)
            ax.add_patch(uc)
            if style == 'square':
                r = 1.1
                ax.axis([-r, r, -r, r])
                ax.axis('equal')
        #    ax.spines['left'].set_position('center')
        #    ax.spines['bottom'].set_position('center')
        #    ax.spines['right'].set_visible(True)
        #    ax.spines['top'].set_visible(True)

        else: # s-plane
            if anaCircleRad > 0:
                # plot a circle with radius = anaCircleRad
                uc = patches.Circle((0,0), radius=anaCircleRad, fill=False,
                                    color='grey', ls='solid', zorder=1)
                ax.add_patch(uc)
            # plot real and imaginary axis
            ax.axhline(lw=2, color = 'k', zorder=1)
            ax.axvline(lw=2, color = 'k', zorder=1)

        # Plot the zeros
        ax.scatter(z.real, z.imag, s=mzs*mzs, zorder=2, marker = 'o',
                   facecolor = 'none', edgecolor = mzc, lw = lw, label=zlabel)
        # and print their multiplicity
        for i in range(len(z)):
            if num_z[i] > 1:
                ax.text(np.real(z[i]), np.imag(z[i]),'  (' + str(num_z[i]) +')',
                                va = 'top', color=mzc)
        if plt_poles:
            # Plot the poles
            ax.scatter(p.real, p.imag, s=mps*mps, zorder=2, marker='x',
                       color=mpc, lw=lw, label=plabel)
            # and print their multiplicity
            for i in range(len(p)):
                if num_p[i] > 1:
                    ax.text(np.real(p[i]), np.imag(p[i]), '  (' + str(num_p[i]) +')',
                                    va = 'bottom', color=mpc)

            # increase distance between ticks and labels
            # to give some room for poles and zeros
        for tick in ax.get_xaxis().get_major_ticks():
            tick.set_pad(12.)
            tick.label1 = tick._get_text1()
        for tick in ax.get_yaxis().get_major_ticks():
            tick.set_pad(12.)
            tick.label1 = tick._get_text1()

        xl = ax.get_xlim(); Dx = max(abs(xl[1]-xl[0]), 0.05)
        yl = ax.get_ylim(); Dy = max(abs(yl[1]-yl[0]), 0.05)
        ax.set_xlim((xl[0]-Dx*0.05, max(xl[1]+Dx*0.05,0)))
        ax.set_ylim((yl[0]-Dy*0.05, yl[1] + Dy*0.05))

        return z, p, k

#------------------------------------------------------------------------------
def impz(b, a=1, FS=1, N=1, step=False):
    """
    Calculate impulse response of a discrete time filter, specified by
    numerator coefficients b and denominator coefficients a of the system
    function H(z).
    
    When only b is given, the impulse response of the transversal (FIR)
    filter specified by b is calculated.
    
    Parameters
    ----------
    b :  array_like
         Numerator coefficients (transversal part of filter)
    
    a :  array_like (optional, default = 1 for FIR-filter)
         Denominator coefficients (recursive part of filter)
    
    FS : float (optional, default: FS = 1)
         Sampling frequency.
    
    N :  float (optional, default N=1 for automatic calculation)
         Number of calculated points.
         Default: N = len(b) for FIR filters, N = 100 for IIR filters

    step: boolean (optional, default: step=False)
         plot step response instead of impulse response
    
    Returns
    -------
    hn : ndarray with length N (see above)
    td : ndarray containing the time steps with same
    
    
    Examples
    --------
    >>> b = [1,2,3] # Coefficients of H(z) = 1 + 2 z^2 + 3 z^3
    >>> h, n = dsp_lib.impz(b)
    """
    try: len(a) #len_a = len(a)
    except TypeError:
         # a has len = 1 -> FIR-Filter
        impulse = np.repeat(0.,len(b)) # create float array filled with 0.
        try: len(b)
        except TypeError:
            print('No proper filter coefficients: len(a) = len(b) = 1 !')
    else:
        try: len(b)
        except TypeError: b = [b,] # convert scalar to array with len = 1
        impulse = np.repeat(0.,100)  # IIR-Filter
    if N > 1:
        impulse = np.repeat(0.,N)
    impulse[0] =1.0 # create dirac impulse
    hn = np.array(sig.lfilter(b,a,impulse)) # calculate impulse response
    td = np.arange(len(hn)) / FS

    if step:
        hn = np.cumsum(hn) # integrate impulse response to get step response
    return hn, td

#==================================================================
def grpdelay(b, a=1, nfft=512, whole='none', analog=False, Fs=2.*pi):
#==================================================================
    """
    Calculate group delay of a discrete time filter, specified by
    numerator coefficients `b` and denominator coefficients `a` of the system
    function `H` ( `z`).
    
    When only `b` is given, the group delay of the transversal (FIR)
    filter specified by `b` is calculated.
    
    Parameters
    ----------
    b :  array_like
         Numerator coefficients (transversal part of filter)
    
    a :  array_like (optional, default = 1 for FIR-filter)
         Denominator coefficients (recursive part of filter)
    
    whole : string (optional, default : 'none')
         Only when whole = 'whole' calculate group delay around
         the complete unit circle (0 ... 2 pi)
    
    N :  integer (optional, default: 512)
         Number of FFT-points
    
    FS : float (optional, default: FS = 2*pi)
         Sampling frequency.
    
    
    Returns
    -------
    tau_g : ndarray
        The group delay
    
    
    w : ndarray
        The angular frequency points where the group delay was computed
    
    Notes
    -----
    The group delay :math:`\\tau_g(\\omega)` of discrete and continuous time
    systems is defined by
    
    .. math::
    
        \\tau_g(\\omega) = -  \\phi'(\\omega)
            = -\\frac{\\partial \\phi(\\omega)}{\\partial \\omega}
            = -\\frac{\\partial }{\\partial \\omega}\\angle H( \\omega)
    
    A useful form for calculating the group delay is obtained by deriving the
    *logarithmic* frequency response in polar form as described in [JOS]_ for
    discrete time systems:
    
    .. math::
    
        \\ln ( H( \\omega))
          = \\ln \\left({H_A( \\omega)} e^{j \\phi(\\omega)} \\right)
          = \\ln \\left({H_A( \\omega)} \\right) + j \\phi(\\omega)
    
          \\Rightarrow \\; \\frac{\\partial }{\\partial \\omega} \\ln ( H( \\omega))
          = \\frac{H_A'( \\omega)}{H_A( \\omega)} +  j \\phi'(\\omega)
    
    where :math:`H_A(\\omega)` is the amplitude response. :math:`H_A(\\omega)` and
    its derivative :math:`H_A'(\\omega)` are real-valued, therefore, the group
    delay can be calculated from
    
    .. math::
    
          \\tau_g(\\omega) = -\\phi'(\\omega) =
          -\\Im \\left\\{ \\frac{\\partial }{\\partial \\omega}
          \\ln ( H( \\omega)) \\right\\}
          =-\\Im \\left\\{ \\frac{H'(\\omega)}{H(\\omega)} \\right\\}
    
    The derivative of a polynome :math:`P(s)` (continuous-time system) or :math:`P(z)`
    (discrete-time system) w.r.t. :math:`\\omega` is calculated by:
    
    .. math::
    
        \\frac{\\partial }{\\partial \\omega} P(s = j \\omega)
        = \\frac{\\partial }{\\partial \\omega} \\sum_{k = 0}^N c_k (j \\omega)^k
        =  j \\sum_{k = 0}^{N-1} (k+1) c_{k+1} (j \\omega)^{k}
        =  j P_R(s = j \\omega)
    
        \\frac{\\partial }{\\partial \\omega} P(z = e^{j \\omega T})
        = \\frac{\\partial }{\\partial \\omega} \\sum_{k = 0}^N c_k e^{-j k \\omega T}
        =  -jT \\sum_{k = 0}^{N} k c_{k} e^{-j k \\omega T}
        =  -jT P_R(z = e^{j \\omega T})
    
    where :math:`P_R` is the "ramped" polynome, i.e. its `k` th coefficient is
    multiplied by `k` resp. `k` + 1.
    
    yielding:
    
    .. math::
    
        \\tau_g(\\omega) = -\\Im \\left\\{ \\frac{H'(\\omega)}{H(\\omega)} \\right\\}
        \\quad \\text{ resp. } \\quad
        \\tau_g(\\omega) = -\\Im \\left\\{ \\frac{H'(e^{j \\omega T})}
                        {H(e^{j \\omega T})} \\right\\}
    
    
    where::
    
                        (H'(e^jwT))       (    H_R(e^jwT))        (H_R(e^jwT))
        tau_g(w) = -im  |---------| = -im |-jT ----------| = T re |----------|
                        ( H(e^jwT))       (    H(e^jwT)  )        ( H(e^jwT) )
    
    where :math:`H(e^{j\\omega T})` is calculated via the DFT at NFFT points and
    the derivative
    of the polynomial terms :math:`b_k z^-k` using :math:`\\partial / \\partial w b_k e^-jkwT` = -b_k jkT e^-jkwT.
    This is equivalent to muliplying the polynome with a ramp `k`,
    yielding the "ramped" function H_R(e^jwT).
    
    
    
    For analog functions with b_k s^k the procedure is analogous, but there is no
    sampling time and the exponent is positive.
    
    
    
    .. [JOS] Julius O. Smith III, "Numerical Computation of Group Delay" in
        "Introduction to Digital Filters with Audio Applications",
        Center for Computer Research in Music and Acoustics (CCRMA),
        Stanford University, http://ccrma.stanford.edu/~jos/filters/Numerical_Computation_Group_Delay.html, referenced 2014-04-02,
    
    .. [Lyons] Richard Lyons, "Understanding Digital Signal Processing", 3rd Ed.,
        Prentice Hall, 2010.
    
    Examples
    --------
    >>> b = [1,2,3] # Coefficients of H(z) = 1 + 2 z^2 + 3 z^3
    >>> tau_g, td = dsp_lib.grpdelay(b)
    
    
    """
## If the denominator of the computation becomes too small, the group delay
## is set to zero.  (The group delay approaches infinity when
## there are poles or zeros very close to the unit circle in the z plane.)
##
## Theory: group delay, g(w) = -d/dw [arg{H(e^jw)}],  is the rate of change of
## phase with respect to frequency.  It can be computed as:
##
##               d/dw H(e^-jw)
##        g(w) = -------------
##                 H(e^-jw)
##
## where
##         H(z) = B(z)/A(z) = sum(b_k z^k)/sum(a_k z^k).
##
## By the quotient rule,
##                    A(z) d/dw B(z) - B(z) d/dw A(z)
##        d/dw H(z) = -------------------------------
##                               A(z) A(z)
## Substituting into the expression above yields:
##                A dB - B dA
##        g(w) =  ----------- = dB/B - dA/A
##                    A B
##
## Note that,
##        d/dw B(e^-jw) = sum(k b_k e^-jwk)
##        d/dw A(e^-jw) = sum(k a_k e^-jwk)
## which is just the FFT of the coefficients multiplied by a ramp.
##
## As a further optimization when nfft>>length(a), the IIR filter (b,a)
## is converted to the FIR filter conv(b,fliplr(conj(a))).
    if whole !='whole':
        nfft = 2*nfft
    nfft = int(nfft)
#
    w = Fs * np.arange(0, nfft)/nfft # create frequency vector

    try: len(a)
    except TypeError:
        a = 1; oa = 0 # a is a scalar or empty -> order of a = 0
        c = b
        try: len(b)
        except TypeError: print('No proper filter coefficients: len(a) = len(b) = 1 !')
    else:
        oa = len(a)-1               # order of denom. a(z) resp. a(s)
        c = np.convolve(b,a[::-1])  # a[::-1] reverses denominator coeffs a
                                    # c(z) = b(z) * a(1/z)*z^(-oa)
    try: len(b)
    except TypeError: b=1; ob=0     # b is a scalar or empty -> order of b = 0
    else:
        ob = len(b)-1             # order of b(z)

    if analog:
        a_b = np.convolve(a,b)
        if ob > 1:
            br_a = np.convolve(b[1:] * np.arange(1,ob), a)
        else:
            br_a = 0
        ar_b = np.convolve(a[1:] * np.arange(1,oa), b)

        num = np.fft.fft(ar_b - br_a, nfft)
        den = np.fft.fft(a_b,nfft)
    else:
        oc = oa + ob                  # order of c(z)
        cr = c * np.arange(0,oc+1) # multiply with ramp -> derivative of c wrt 1/z

        num = np.fft.fft(cr,nfft) #
        den = np.fft.fft(c,nfft)  #
#
    minmag = 10. * np.spacing(1) # equivalent to matlab "eps"
    polebins = np.where(abs(den) < minmag)[0] # find zeros of denominator
#    polebins = np.where(abs(num) < minmag)[0] # find zeros of numerator
    if np.size(polebins) > 0:  # check whether polebins array is empty
        print('*** grpdelay warning: group delay singular -> setting to 0 at:')
        for i in polebins:
            print ('f = {0} '.format((Fs*i/nfft)))
            num[i] = 0
            den[i] = 1

    if analog:
        tau_g = np.real(num / den)
    else:
        tau_g = np.real(num / den) - oa
#
    if whole !='whole':
        nfft = nfft//2
        tau_g = tau_g[0:nfft]
        w = w[0:nfft]

    return tau_g, w

def grp_delay_ana(b, a, w):
    """
    Calculate the group delay of an anlog filter.
    """
    w, H = sig.freqs(b, a, w)
    H_angle = np.unwrap(np.angle(H))
#    tau_g = np.zeros(len(w)-1)
    tau_g = (H_angle[1:]-H_angle[:-1])/(w[0]-w[1])
    return tau_g, w[:-1]


# ------------------------------------------------------------------------------
def div_safe(num, den, n_eps=1, i_scale=1, verbose=False):

    """
    Perform elementwise array division after treating singularities, meaning:
    - check whether denominator (`den`) coefficients approach zero
    - check whether numerator (`num`) or denominator coefficients are non-finite, i.e.
      one of `nan`, `ìnf` or `ninf`.

    At each singularity, replace denominator coefficient by `1` and numerator
    coefficient by `0`

    Parameters
    ----------
    num : array_like
        numerator coefficients
    den : array_like
        denominator coefficients

    n_eps : float
        n_eps * machine resolution is the limit for the denominator below which
        the ratio is set to zero. The machine resolution in numpy is given by
        `np.spacing(1)`, the distance to the nearest number which is equivalent
        to matlab's "eps".

    i_scale : float
        The scale for the index `i` for `num, den` for printing the index of
        the singularities.

    verbose : bool, optional
        whether to print  The default is False.

    Returns
    -------
    ratio : array_like
            The ratio of num and den (zero at singularities)

    """
    singular = np.where(~np.isfinite(den) | ~np.isfinite(num) |
                        (abs(den) < n_eps * np.spacing(1)))[0]

    if verbose and np.any(singular):
        print('div_safe singularity -> setting to 0 at:')
        for i in singular:
            print('i = {0} '.format(i * i_scale))

    num[singular] = 0
    den[singular] = 1

    return num / den


# ------------------------------------------------------------------------------
def validate_sos(sos):
    """
    Helper to validate a SOS input

    Copied from `scipy.signal._filter_design._validate_sos()`
    """
    sos = np.atleast_2d(sos)
    if sos.ndim != 2:
        raise ValueError('sos array must be 2D')
    n_sections, m = sos.shape
    if m != 6:
        raise ValueError('sos array must be shape (n_sections, 6)')
    if not (sos[:, 3] == 1).all():
        raise ValueError('sos[:, 3] should be all ones')
    return sos, n_sections


# ------------------------------------------------------------------------------
def group_delay(b, a=1, nfft=512, whole=False, analog=False, verbose=True,
                fs=2.*pi, sos=False, alg="auto", n_eps=100):
    """
Calculate group delay of a discrete time filter, specified by
numerator coefficients `b` and denominator coefficients `a` of the system
function `H` ( `z`).

When only `b` is given, the group delay of the transversal (FIR)
filter specified by `b` is calculated.

Parameters
----------
b :  array_like
     Numerator coefficients (transversal part of filter)

a :  array_like (optional, default = 1 for FIR-filter)
     Denominator coefficients (recursive part of filter)

whole : boolean (optional, default : False)
     Only when True calculate group delay around
     the complete unit circle (0 ... 2 pi)

verbose : boolean (optional, default : True)
    Print warnings about frequency points with undefined group delay (amplitude = 0)
    and the time used for calculating the group delay

nfft :  integer (optional, default: 512)
     Number of FFT-points

fs : float (optional, default: fs = 2*pi)
     Sampling frequency.

alg : str (default: "scipy")
      The algorithm for calculating the group delay:
          - "scipy" The algorithm used by scipy's grpdelay,
          - "jos": The original J.O.Smith algorithm; same as in "scipy" except that
            the frequency response is calculated with the FFT instead of polyval
          - "diff": Group delay is calculated by differentiating the phase
          - "Shpakh": Group delay is calculated from second-order sections

n_eps : integer (optional, default : 100)
        Minimum value in the calculation of intermediate values before tau_g is set
        to zero.

Returns
-------
tau_g : ndarray
        group delay

w : ndarray
    angular frequency points where group delay was computed

Notes
=======

The following explanations follow [JOS]_.

**Definition and direct calculation ('diff')**

The group delay :math:`\\tau_g(\\omega)` of discrete time (DT) and continuous time
(CT) systems is the rate of change of phase with respect to angular frequency.
In the following, derivative is always meant w.r.t. :math:`\\omega`:

.. math::

    \\tau_g(\\omega)
        = -\\frac{\\partial }{\\partial \\omega}\\angle H( \\omega)
        = -\\frac{\\partial \\phi(\\omega)}{\\partial \\omega}
        = -  \\phi'(\\omega)

With numpy / scipy, the group delay can be calculated directly with

.. code-block:: python

    w, H = sig.freqz(b, a, worN=nfft, whole=whole)
    tau_g = -np.diff(np.unwrap(np.angle(H)))/np.diff(w)

The derivative can create numerical problems for e.g. phase jumps at zeros of
frequency response or when the complex frequency response becomes very small e.g.
in the stop band.

This can be avoided by calculating the group delay from the derivative of the
*logarithmic* frequency response in polar form (amplitude response and phase):

.. math::

    \\ln ( H( \\omega))
      = \\ln \\left({H_A( \\omega)} e^{j \\phi(\\omega)} \\right)
      = \\ln \\left({H_A( \\omega)} \\right) + j \\phi(\\omega)

      \\Rightarrow \\; \\frac{\\partial }{\\partial \\omega} \\ln ( H( \\omega))
      = \\frac{H_A'( \\omega)}{H_A( \\omega)} +  j \\phi'(\\omega)

where :math:`H_A(\\omega)` is the amplitude response. :math:`H_A(\\omega)` and
its derivative :math:`H_A'(\\omega)` are real-valued, therefore, the group
delay can be calculated by separating real and imginary components (and discarding
the real part):

.. math::

    \\begin{align}
    \\Re \\left\\{\\frac{\\partial }{\\partial \\omega} \\ln ( H( \\omega))\\right\\} &= \\frac{H_A'( \\omega)}{H_A( \\omega)} \\\
    \\Im \\left\\{\\frac{\\partial }{\\partial \\omega} \\ln ( H( \\omega))\\right\\} &= \\phi'(\\omega)
    \\end{align}

and hence

.. math::

      \\tau_g(\\omega) = -\\phi'(\\omega) =
      -\\Im \\left\\{ \\frac{\\partial }{\\partial \\omega}
      \\ln ( H( \\omega)) \\right\\}
      =-\\Im \\left\\{ \\frac{H'(\\omega)}{H(\\omega)} \\right\\}

Note: The last term contains the complex response :math:`H(\omega)`, not the
amplitude response :math:`H_A(\omega)`!

In the following, it will be shown that the derivative of birational functions
(like DT and CT filters) can be calculated very efficiently and from this the group
delay.


**J.O. Smith's basic algorithm for FIR filters ('scipy')**

An efficient form of calculating the group delay of FIR filters based on the
derivative of the logarithmic frequency response has been described in [JOS]_
and [Lyons]_ for discrete time systems.

A FIR filter is defined via its polyome :math:`H(z) = \\sum_k b_k z^{-k}` and has
the following derivative:

.. math::

    \\frac{\\partial }{\\partial \\omega} H(z = e^{j \\omega T})
    = \\frac{\\partial }{\\partial \\omega} \\sum_{k = 0}^N b_k e^{-j k \\omega T}
    =  -jT \\sum_{k = 0}^{N} k b_{k} e^{-j k \\omega T}
    =  -jT H_R(e^{j \\omega T})

where :math:`H_R` is the "ramped" polynome, i.e. polynome :math:`H` multiplied
with a ramp :math:`k`, yielding

.. math::

    \\tau_g(e^{j \\omega T}) = -\\Im \\left\\{ \\frac{H'(e^{j \\omega T})}
                    {H(e^{j \\omega T})} \\right\\}
                    = -\\Im \\left\\{ -j T \\frac{H_R(e^{j \\omega T})}
                    {H(e^{j \\omega T})} \\right\\}
                    = T \\, \\Re \\left\\{\\frac{H_R(e^{j \\omega T})}
                    {H(e^{j \\omega T})} \\right\\}

scipy's grpdelay directly calculates the complex frequency response
:math:`H(e^{j\\omega T})` and its ramped function at the frequency points using
the polyval function.

When zeros of the frequency response are on or near the data points of the DFT, this
algorithm runs into numerical problems. Hence, it is neccessary to check whether
the magnitude of the denominator is less than e.g. 100 times the machine eps.
In this case, :math:`\\tau_g` is set to zero.

**J.O. Smith's basic algorithm for IIR filters ('scipy')**

IIR filters are defined by

.. math::

        H(z) = \\frac {B(z)}{A(z)} = \\frac {\\sum b_k z^k}{\\sum a_k z^k},

their group delay can be calculated numerically via the logarithmic frequency
response as well.

The derivative  of :math:`H(z)` w.r.t. :math:`\\omega` is calculated using the
quotient rule and by replacing the derivatives of numerator and denominator
polynomes with their ramp functions:

.. math::

    \\begin{align}
    \\frac{H'(e^{j \\omega T})}{H(e^{j \\omega T})}
    &= \\frac{\\left(B(e^{j \\omega T})/A(e^{j \\omega T})\\right)'}{B(e^{j \\omega T})/A(e^{j \\omega T})}
    = \\frac{B'(e^{j \\omega T}) A(e^{j \\omega T}) - A'(e^{j \\omega T})B(e^{j \\omega T})}
    { A(e^{j \\omega T}) B(e^{j \\omega T})}  \\\\
    &= \\frac {B'(e^{j \\omega T})} { B(e^{j \\omega T})}
      - \\frac { A'(e^{j \\omega T})} { A(e^{j \\omega T})}
    = -j T \\left(\\frac { B_R(e^{j \\omega T})} {B(e^{j \\omega T})} - \\frac { A_R(e^{j \\omega T})} {A(e^{j \\omega T})}\\right)
    \\end{align}

This result is substituted once more into the log. derivative from above:

.. math::

    \\begin{align}
    \\tau_g(e^{j \\omega T})
    =-\\Im \\left\\{ \\frac{H'(e^{j \\omega T})}{H(e^{j \\omega T})} \\right\\}
    &=-\\Im \\left\\{
        -j T \\left(\\frac { B_R(e^{j \\omega T})} {B(e^{j \\omega T})}
                    - \\frac { A_R(e^{j \\omega T})} {A(e^{j \\omega T})}\\right)
                     \\right\\} \\\\
        &= T \\Re \\left\\{\\frac { B_R(e^{j \\omega T})} {B(e^{j \\omega T})}
                    - \\frac { A_R(e^{j \\omega T})} {A(e^{j \\omega T})}
         \\right\\}
    \\end{align}


If the denominator of the computation becomes too small, the group delay
is set to zero.  (The group delay approaches infinity when
there are poles or zeros very close to the unit circle in the z plane.)

**J.O. Smith's algorithm for CT filters**

The same process can be applied for CT systems as well: The derivative of a CT
polynome :math:`P(s)` w.r.t. :math:`\\omega` is calculated by:

.. math::

    \\frac{\\partial }{\\partial \\omega} P(s = j \\omega)
    = \\frac{\\partial }{\\partial \\omega} \\sum_{k = 0}^N c_k (j \\omega)^k
    =  j \\sum_{k = 0}^{N-1} (k+1) c_{k+1} (j \\omega)^{k}
    =  j P_R(s = j \\omega)

where :math:`P_R` is the "ramped" polynome, i.e. its `k` th coefficient is
multiplied by the ramp `k` + 1, yielding the same form as for DT systems (but
the ramped polynome has to be calculated differently).

.. math::

    \\tau_g(\\omega) = -\\Im \\left\\{ \\frac{H'(\\omega)}{H(\\omega)} \\right\\}
                     = -\\Im \\left\\{j \\frac{H_R(\\omega)}{H(\\omega)} \\right\\}
                     = -\\Re \\left\\{\\frac{H_R(\\omega)}{H(\\omega)} \\right\\}


**J.O. Smith's improved algorithm for IIR filters ('jos')**

J.O. Smith gives the following speed and accuracy optimizations for the basic
algorithm:

    * convert the filter to a FIR filter with identical phase and group delay
      (but with different magnitude response)

    * use FFT instead of polyval to calculate the frequency response

The group delay of an IIR filter :math:`H(z) = B(z)/A(z)` can also
be calculated from an equivalent FIR filter :math:`C(z)` with the same phase
response (and hence group delay) as the original filter. This filter is obtained
by the following steps:

* The zeros of :math:`A(z)` are the poles of :math:`1/A(z)`, its phase response is
  :math:`\\angle A(z) = - \\angle 1/A(z)`.

* Transforming :math:`z \\rightarrow 1/z` mirrors the zeros at the unit circle,
  correcting the negative phase response. This can be performed numerically by "flipping"
  the order of the coefficients and multiplying by :math:`z^{-N}` where :math:`N`
  is the order of :math:`A(z)`. This operation also conjugates the coefficients (?)
  which mirrors the zeros at the real axis. This effect has to be compensated,
  yielding the polynome :math:`\\tilde{A}(z)`. It is the "flip-conjugate" or
  "Hermitian conjugate" of :math:`A(z)`.

  Frequently (e.g. in the scipy and until recently in the Matlab implementation)
  the conjugate operation is omitted which gives wrong results for complex
  coefficients.

* Finally, :math:`C(z) = B(z) \\tilde{A}(z)`:

.. math::

    C(z) = B(z)\\left[ z^{-N}{A}^{*}(1/z)\\right] = B(z)\\tilde{A}(z)

where

.. math::

    \\begin{align}
    \\tilde{A}(z) &=  z^{-N}{A}^{*}(1/z) = {a}^{*}_N + {a}^{*}_{N-1}z^{-1} + \ldots + {a}^{*}_1 z^{-(N-1)}+z^{-N}\\\\
    \Rightarrow \\tilde{A}(e^{j\omega T}) &=  e^{-jN \omega T}{A}^{*}(e^{-j\omega T}) \\\\
    \\Rightarrow \\angle\\tilde{A}(e^{j\omega T}) &= -\\angle A(e^{j\omega T}) - N\omega T
    \\end{align}


In Python, the coefficients of :math:`C(z)` are calculated efficiently by
convolving the coefficients of :math:`B(z)` and  :math:`\\tilde{A}(z)`:

.. code-block:: python

    c = np.convolve(b, np.conj(a[::-1]))

where :math:`b` and :math:`a` are the coefficient vectors of the original
numerator and denominator polynomes. The actual group delay is then calculated
from the equivalent FIR filter as described above.

Calculating the frequency response with the `np.polyval(p,z)` function at the
`NFFT` frequency points along the unit circle, :math:`z = \\exp(-j \\omega)`,
seems to be numerically less robust than using the FFT for the same task, it
is also much slower.

This measure fixes already most of the problems described for narrowband IIR
filters in scipy issues [SC9310]_ and [SC1175]_. In my experience, these problems
occur for all narrowband IIR response types.

**Shpak algorithm for IIR filters**

The algorithm described above is numerically efficient but not robust for
narrowband IIR filters. Especially for filters defined by second-order sections,
it is recommended to calculate the group delay using the D. J. Shpak's algorithm.

Code is available at [ENDO5828333]_ (GPL licensed) or at [SPA]_ (MIT licensed).

This algorithm sums the group delays of the individual sections which is much
more robust as only second-order functions are involved. However, converting `(b,a)`
coefficients to SOS coefficients introduces inaccuracies.

References
```````````

.. [JOS] https://ccrma.stanford.edu/%7Ejos/fp/Numerical_Computation_Group_Delay.html or

         https://www.dsprelated.com/freebooks/filters/Numerical_Computation_Group_Delay.html

.. [Lyons] https://www.dsprelated.com/showarticle/69.php

.. [SC1175] https://github.com/scipy/scipy/issues/1175

.. [SC9310] https://github.com/scipy/scipy/issues/9310

.. [SPA] https://github.com/spatialaudio/group-delay-of-filters

.. [ENDO5828333] https://gist.github.com/endolith/5828333

.. [OCTAVE] https://sourceforge.net/p/octave/mailman/message/9298101/

Examples
--------
>>> b = [1,2,3] # Coefficients of H(z) = 1 + 2 z^2 + 3 z^3
>>> tau_g, td = pyfda_lib.grpdelay(b)
"""

    if not whole:
        nfft = 2*nfft
    w = fs * np.arange(0, nfft)/nfft  # create frequency vector

    tau_g = np.zeros_like(w)  # initialize tau_g

    # ----------------
    if alg == 'auto':
        if sos:
            alg = "shpak"
            if verbose:
                print("Filter in SOS format, using Shpak algorithm for group delay.")

        elif not np.isscalar(a):
            alg = 'jos'  # TODO: use 'shpak' here as well?
            if verbose:
                print("IIR filter, using J.O. Smith's algorithm for group delay.")
        else:
            alg = 'jos'
            if verbose:
                print("FIR filter, using J.O. Smith's algorithm for group delay.")

    if sos and alg != "shpak":
        b, a = sig.sos2tf(b)

    # ---------------------
    if alg == 'diff':
        w, H = sig.freqz(b, a, worN=nfft, whole=whole)
        # np.spacing(1) is equivalent to matlab "eps"
        singular = np.absolute(H) < n_eps * 10 * np.spacing(1)
        H[singular] = 0
        tau_g = -np.diff(np.unwrap(np.angle(H)))/np.diff(w)
        # differentiation returns one element less than its input
        w = w[:-1]

    # ---------------------
    elif alg == 'jos':
        b, a = map(np.atleast_1d, (b, a))  # when scalars, convert to 1-dim. arrays

        c = np.convolve(b, a[::-1])     # equivalent FIR polynome
                                        # c(z) = b(z) * a(1/z)*z^(-oa)
        cr = c * np.arange(c.size)      # multiply with ramp -> derivative of c wrt 1/z

        den = np.fft.fft(c, nfft)   # FFT = evaluate polynome around unit circle
        num = np.fft.fft(cr, nfft)  # and ramped polynome at NFFT equidistant points

        # Check for singularities i.e. where denominator (`den`) coefficients
        # approach zero or numerator (`num`) or denominator coefficients are
        # non-finite, i.e. `nan`, `ìnf` or `ninf`.
        singular = np.where(~np.isfinite(den) | ~np.isfinite(num) |
                            (abs(den) < n_eps * np.spacing(1)))[0]

        with np.errstate(invalid='ignore', divide='ignore'):
            # element-wise division of numerator and denominator FFTs
            tau_g = np.real(num / den) - a.size + 1

        # set group delay = 0 at each singularity
        tau_g[singular] = 0

        if verbose and np.any(singular):
            print('singularity -> setting to 0 at:')
            for i in singular:
                print('i = {0} '.format(i * fs/nfft))

        if not whole:
            nfft = nfft//2
            tau_g = tau_g[0:nfft]
            w = w[0:nfft]

        # if analog: # doesnt work yet
        #     a_b = np.convolve(a,b)
        #     if ob > 1:
        #         br_a = np.convolve(b[1:] * np.arange(1,ob), a)
        #     else:
        #         br_a = 0
        #     ar_b = np.convolve(a[1:] * np.arange(1,oa), b)

        #     num = np.fft.fft(ar_b - br_a, nfft)
        #     den = np.fft.fft(a_b,nfft)

    # ---------------------
    elif alg == "scipy":  # implementation as in scipy.signal
        w = np.atleast_1d(w)
        b, a = map(np.atleast_1d, (b, a))
        c = np.convolve(b, a[::-1])    # coefficients of equivalent FIR polynome
        cr = c * np.arange(c.size)     # and of the ramped polynome
        z = np.exp(-1j * w)            # complex frequency points around the unit circle
        den = np.polyval(c[::-1], z)   # evaluate polynome
        num = np.polyval(cr[::-1], z)  # and ramped polynome

        # Check for singularities i.e. where denominator (`den`) coefficients
        # approach zero or numerator (`num`) or denominator coefficients are
        # non-finite, i.e. `nan`, `ìnf` or `ninf`.
        singular = np.where(~np.isfinite(den) | ~np.isfinite(num) |
                            (abs(den) < n_eps * np.spacing(1)))[0]

        with np.errstate(invalid='ignore', divide='ignore'):
            # element-wise division of numerator and denominator FFTs
            tau_g = np.real(num / den) - a.size + 1

        # set group delay = 0 at each singularity
        tau_g[singular] = 0

        if verbose and np.any(singular):
            print('singularity -> setting to 0 at:')
            for i in singular:
                print('i = {0} '.format(i * fs/nfft))

        if not whole:
            nfft = nfft//2
            tau_g = tau_g[0:nfft]
            w = w[0:nfft]

    # ---------------------
    elif alg.lower() == "shpak":  # Use Shpak's algorith
        if sos:
            w, tau_g = sos_group_delayz(b, w, fs=fs)
        else:
            w, tau_g = group_delayz(b, a, w, fs=fs)

    else:
        print('Unknown algorithm "{0}"!'.format(alg))
        tau_g = np.zeros_like(w)

    return w, tau_g


# ------------------------------------------------------------------------------
def group_delayz(b, a, w, plot=None, fs=2*np.pi):
    """
    Compute the group delay of digital filter.

    Parameters
    ----------
    b : array_like
        Numerator of a linear filter.
    a : array_like
        Denominator of a linear filter.
    w : array_like
        Frequencies in the same units as `fs`.
    plot : callable
        A callable that takes two arguments. If given, the return parameters
        `w` and `gd` are passed to plot.
    fs : float, optional
        The angular sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed, in the same units as `fs`.
    gd : ndarray
        The group delay in seconds.
    """
    b, a = map(np.atleast_1d, (b, a))
    if len(a) == 1:
        # scipy.signal.group_delay returns gd in samples thus scaled by 1/fs
        gd = sig.group_delay((b, a), w=w, fs=fs)[1]
    else:
        sos = sig.tf2sos(b, a)
        gd = sos_group_delayz(sos, w, plot, fs)[1]
    if plot is not None:
        plot(w, gd)
    return w, gd


# ------------------------------------------------------------------------------
#
# The *_group_delayz routines and subroutines have been copied from
#
# https://github.com/spatialaudio/group-delay-of-filters
#
# committed by Nara Hahn under MIT license
#
# ------------------------------------------------------------------------------
def sos_group_delayz(sos, w, plot=None, fs=2*np.pi):
    """
    Compute group delay of digital filter in SOS format.

    Parameters
    ----------
    sos : array_like
        Array of second-order filter coefficients, must have shape
        ``(n_sections, 6)``. Each row corresponds to a second-order
        section, with the first three columns providing the numerator
        coefficients and the last three providing the denominator
        coefficients.
    w : array_like
        Frequencies in the same units as `fs`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `gd` are passed to plot.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    sos, n_sections = validate_sos(sos)
    if n_sections == 0:
        raise ValueError('Cannot compute group delay with no sections')
    gd = 0
    for biquad in sos:
        gd += quadfilt_group_delayz(biquad[:3], w, fs)[1]
        gd -= quadfilt_group_delayz(biquad[3:], w, fs)[1]
    if plot is not None:
        plot(w, gd)
    return w, gd


def quadfilt_group_delayz(b, w, fs=2*np.pi):
    """
    Compute group delay of 2nd-order digital filter.

    Parameters
    ----------
    b : array_like
        Coefficients of a 2nd-order digital filter.
    w : array_like
        Frequencies in the same units as `fs`.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    W = 2 * pi * w / fs
    c1 = np.cos(W)
    c2 = np.cos(2*W)
    u0, u1, u2 = b**2  # b[0]**2, b[1]**2, b[2]**2
    v0, v1, v2 = b * np.roll(b, -1)  # b[0]*b[1], b[1]*b[2], b[2]*b[0]
    num = (u1+2*u2) + (v0+3*v1)*c1 + 2*v2*c2
    den = (u0+u1+u2) + 2*(v0+v1)*c1 + 2*v2*c2

    ratio = div_safe(num, den, n_eps=100, verbose=False)

    return w, 2 * pi / fs * ratio


def zpk_group_delay(z, p, k, w, plot=None, fs=2*np.pi):
    """
    Compute group delay of digital filter in zpk format.

    Parameters
    ----------
    z : array_like
        Zeroes of a linear filter
    p : array_like
        Poles of a linear filter
    k : scalar
        Gain of a linear filter
    w : array_like
        Frequencies in the same units as `fs`.
    plot : callable, optional
        A callable that takes two arguments. If given, the return parameters
        `w` and `gd` are passed to plot.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    gd = 0
    for z_i in z:
        gd += zorp_group_delayz(z_i, w)[1]
    for p_i in p:
        gd -= zorp_group_delayz(p_i, w)[1]
    if plot is not None:
        plot(w, gd)
    return w, gd


def zorp_group_delayz(zorp, w, fs=1):
    """
    Compute group delay of digital filter with a single zero/pole.

    Parameters
    ----------
    zorp : complex
        Zero or pole of a 1st-order linear filter
    w : array_like
        Frequencies in the same units as `fs`.
    fs : float, optional
        The sampling frequency of the digital system.

    Returns
    -------
    w : ndarray
        The frequencies at which `gd` was computed.
    gd : ndarray
        The group delay in seconds.
    """
    W = 2 * pi * w / fs
    r, phi = np.abs(zorp), np.angle(zorp)
    r2 = r**2
    cos = np.cos(W - phi)
    return w, 2 * pi * (r2 - r*cos) / (r2 + 1 - 2*r*cos)

#==================================================================
def format_ticks(xy, scale, format="%.1f"):
#==================================================================
    """
    Reformat numbers at x or y - axis. The scale can be changed to display
    e.g. MHz instead of Hz. The number format can be changed as well.
    
    Parameters
    ----------
    xy : string, either 'x', 'y' or 'xy'
         select corresponding axis (axes) for reformatting
    
    scale :  real,
    
    format : string,
             define C-style number formats
    
    Returns
    -------
    nothing
    
    
    Examples
    --------
    >>> format_ticks('x',1000.)
    Scales all numbers of x-Axis by 1000, e.g. for displaying ms instead of s.
    >>> format_ticks('xy',1., format = "%.2f")
    Two decimal places for numbers on x- and y-axis
    """
    if xy == 'x' or xy == 'xy':
        locx,labelx = plt.xticks() # get location and content of xticks
        plt.xticks(locx, map(lambda x: format % x, locx*scale))
    if xy == 'y' or xy == 'xy':
        locy,labely = plt.yticks() # get location and content of xticks
        plt.yticks(locy, map(lambda y: format % y, locy*scale))
        
#==================================================================
def lim_eps(a, eps):
#==================================================================
    """
    Return min / max of an array a, increased by eps*(max(a) - min(a)). 
    Handy for nice looking axes labeling.
    """
    mylim = (min(a) - (max(a)-min(a))*eps, max(a) + (max(a)-min(a))*eps)
    return mylim
      


#========================================================

abs = absolute

def oddround(x):
    """Return the nearest odd integer from x."""

    return x-mod(x,2)+1

def oddceil(x):
    """Return the smallest odd integer not less than x."""

    return oddround(x+1)

def remlplen_herrmann(fp,fs,dp,ds):
    """
    Determine the length of the low pass filter with passband frequency
    fp, stopband frequency fs, passband ripple dp, and stopband ripple ds.
    fp and fs must be normalized with respect to the sampling frequency.
    Note that the filter order is one less than the filter length.
    
    Uses approximation algorithm described by Herrmann et al.:
    
    O. Herrmann, L.R. Raviner, and D.S.K. Chan, Practical Design Rules for
    Optimum Finite Impulse Response Low-Pass Digital Filters, Bell Syst. Tech.
    Jour., 52(6):769-799, Jul./Aug. 1973.
    """

    dF = fs-fp
    a = [5.309e-3,7.114e-2,-4.761e-1,-2.66e-3,-5.941e-1,-4.278e-1]
    b = [11.01217, 0.51244]
    Dinf = log10(ds)*(a[0]*log10(dp)**2+a[1]*log10(dp)+a[2])+ \
           a[3]*log10(dp)**2+a[4]*log10(dp)+a[5]
    f = b[0]+b[1]*(log10(dp)-log10(ds))
    N1 = Dinf/dF-f*dF+1

    return int(oddround(N1))

def remlplen_kaiser(fp,fs,dp,ds):
    """
    Determine the length of the low pass filter with passband frequency
    fp, stopband frequency fs, passband ripple dp, and stopband ripple ds.
    fp and fs must be normalized with respect to the sampling frequency.
    Note that the filter order is one less than the filter length.
    
    Uses approximation algorithm described by Kaiser:
    
    J.F. Kaiser, Nonrecursive Digital Filter Design Using I_0-sinh Window
    function, Proc. IEEE Int. Symp. Circuits and Systems, 20-23, April 1974.
    """
    
    dF = fs-fp
    N2 = (-20*log10(sqrt(dp*ds))-13.0)/(14.6*dF)+1.0
    
    return int(oddceil(N2))
    
def remlplen_ichige(fp,fs,dp,ds):
    """
    Determine the length of the low pass filter with passband frequency
    fp, stopband frequency fs, passband ripple dp, and stopband ripple ds.
    fp and fs must be normalized with respect to the sampling frequency.
    Note that the filter order is one less than the filter length.
    Uses approximation algorithm described by Ichige et al.:
    K. Ichige, M. Iwaki, and R. Ishii, Accurate Estimation of Minimum
    Filter Length for Optimum FIR Digital Filters, IEEE Transactions on
    Circuits and Systems, 47(10):1008-1017, October 2000.
    """

    dF = fs-fp
    v = lambda dF,dp:2.325*((-log10(dp))**-0.445)*dF**(-1.39)
    g = lambda fp,dF,d:(2.0/pi)*arctan(v(dF,dp)*(1.0/fp-1.0/(0.5-dF)))
    h = lambda fp,dF,c:(2.0/pi)*arctan((c/dF)*(1.0/fp-1.0/(0.5-dF)))
    Nc = ceil(1.0+(1.101/dF)*(-log10(2.0*dp))**1.1)
    Nm = (0.52/dF)*log10(dp/ds)*(-log10(dp))**0.17
    N3 = ceil(Nc*(g(fp,dF,dp)+g(0.5-dF-fp,dF,dp)+1.0)/3.0)
    DN = ceil(Nm*(h(fp,dF,1.1)-(h(0.5-dF-fp,dF,0.29)-1.0)/2.0))
    N4 = N3+DN

    return int(N4)

def remezord(freqs,amps,rips,Hz=1,alg='ichige'):
    """
    Filter parameter selection for the Remez exchange algorithm.

    Calculate the parameters required by the Remez exchange algorithm to
    construct a finite impulse response (FIR) filter that approximately
    meets the specified design.
    
    Parameters
    ----------
    
        freqs : list
            A monotonic sequence of band edges specified in Hertz. All elements
            must be non-negative and less than 1/2 the sampling frequency as
            given by the Hz parameter. The band edges "0" and "f_S / 2" do not
            have to be specified, hence  2 * number(amps) - 2 freqs are needed.
    
        amps : list
            A sequence containing the amplitudes of the signal to be
            filtered over the various bands, e.g. 1 for the passband, 0 for the
            stopband and 0.42 for some intermediate band.
    
        rips : list
            A list with the peak ripples (linear, not in dB!) for each band. For
            the stop band this is equivalent to the minimum attenuation.
    
        Hz : float
            Sampling frequency
    
        alg : string
            Filter length approximation algorithm. May be either 'herrmann',
            'kaiser' or 'ichige'. Depending on the specifications, some of
            the algorithms may give better results than the others.
    
    Returns
    -------
    
    numtaps,bands,desired,weight -- See help for the remez function.
    
    Examples
    --------
            We want to design a lowpass with the band edges of 40 resp. 50 Hz and a
            sampling frequency of 200 Hz, a passband peak ripple of 10%
            and a stop band ripple of 0.01 or 40 dB.
        >>> (L, F, A, W) = dsp.remezord([40, 50], [1, 0], [0.1, 0.01], Hz = 200) 

    Notes
    -----

    Supplies remezord method according to Scipy Ticket #475
    http://projects.scipy.org/scipy/ticket/475
    https://github.com/scipy/scipy/issues/1002
    https://github.com/thorstenkranz/eegpy/blob/master/eegpy/filter/remezord.py
    """

    # Make sure the parameters are floating point numpy arrays:
    freqs = asarray(freqs,'d')
    amps = asarray(amps,'d')
    rips = asarray(rips,'d')

    # Scale ripples with respect to band amplitudes:
    rips /= (amps+(amps==0.0))

    # Normalize input frequencies with respect to sampling frequency:
    freqs /= Hz

    # Select filter length approximation algorithm:
    if alg == 'herrmann':
        remlplen = remlplen_herrmann
    elif alg == 'kaiser':
        remlplen = remlplen_kaiser
    elif alg == 'ichige':
        remlplen = remlplen_ichige
    else:
        raise ValueError('Unknown filter length approximation algorithm.')

    # Validate inputs:
    if any(freqs > 0.5):
        raise ValueError('Frequency band edges must not exceed the Nyquist frequency.')
    if any(freqs < 0.0):
        raise ValueError('Frequency band edges must be nonnegative.')
    if any(rips <= 0.0):
        raise ValueError('Ripples must be nonnegative and non-zero.')
    if len(amps) != len(rips):
        raise ValueError('Number of amplitudes must equal number of ripples.')
    if len(freqs) != 2*(len(amps)-1):
        raise ValueError('Number of band edges must equal 2*(number of amplitudes-1)')


    # Find the longest filter length needed to implement any of the
    # low-pass or high-pass filters with the specified edges:
    f1 = freqs[0:-1:2]
    f2 = freqs[1::2]
    L = 0
    for i in range(len(amps)-1):
        L = max((L,
                 remlplen(f1[i],f2[i],rips[i],rips[i+1]),
                 remlplen(0.5-f2[i],0.5-f1[i],rips[i+1],rips[i])))

    # Cap the sequence of band edges with the limits of the digital frequency
    # range:
    bands = hstack((0.0,freqs,0.5))

    # The filter design weights correspond to the ratios between the maximum
    # ripple and all of the other ripples:
    weight = max(rips)/rips

    return [L,bands,amps,weight]

#######################################
# If called directly, do some example #
#######################################
if __name__=='__main__':
    pass
