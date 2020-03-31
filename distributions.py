#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

DISTRIBUTION FUNCTIONS (MASS AND SIZE) AND INTEGRATIONS

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
import math
import numpy as np
from numba import njit

#%% DISTRIBUTIONS

@njit()
def pdf_expo(x, x_mean_inv):
    """Exponential probability density function
    
    Integral ( f(x) dx ) = 1
    
    Parameters
    ----------
    x: float
        Argument
    x_mean_inv: float
        Inverse of the expected value (1/x_mean), where x_mean is the
        expected value and the standard deviation.
    
    Returns
    -------
        float
        Exponential probability density function evaluated at 'x'
    
    """
    
    return x_mean_inv * np.exp(- x * x_mean_inv)

# antiderivative of expo pdf
def prob_exp(x,k):
    """Antiderivative of the exponential probability density function
    
    Parameters
    ----------
    x: float
        Argument
    k: float
        Inverse of the expected value (1/x_mean), where x_mean is the
        expected value and the standard deviation.
    
    Returns
    -------
        float
        Antiderivative of the expo. PDF evaluated at 'x'
    
    """
    
    return 1.0 - np.exp(-k*x)

two_pi_sqrt = math.sqrt(2.0 * math.pi)
@njit()
def pdf_lognormal(x, mu_log, sigma_log):
    """Lognormal probability density function
    
    Integral ( f(x) dx ) = 1.
    About conversions: Let mu be the geometric mean. Then
    arithm. expect. value = mu * exp( 0.5 * sigma_log**2) and
    standard deviation = x_mean * sqrt( exp( sigma_log**2 ) - 1 )
    
    Parameters
    ----------
    x: float
        Argument
    mu_log:
        log. nat. of the geometric mean
    sigma_log: float
        log. nat. of the geometric standard deviation (also called 'mode')
    
    Returns
    -------
        float
        Lognormal probability density function evaluated at 'x'
    
    """
    
    return np.exp( -0.5*( ( np.log( x ) - mu_log ) / sigma_log )**2 ) \
           / (x * two_pi_sqrt * sigma_log)

def dst_normal(x, par):
    """Gaussian (normal) probability density function
    
    Integral ( f(x) dx ) = 1.
    
    Parameters
    ----------
    x: float
        Argument
    par: ndarray
        par[0] = expected value
        par[1] = standard deviation
    
    Returns
    -------
        float
        Normal probability density function evaluated at 'x'
    
    """
    
    return np.exp( -0.5 * ( ( x - par[0] ) / par[1] )**2 ) \
           / (two_pi_sqrt * par[1])

def conc_per_mass_expo_np(m, DNC, DNC_over_LWC):
    """Exponential number concentration density distribution per mass
    
    Also called 'mass distribution'
    f_m(m) = DNC^2/LWC * exp(-m/m_avg), where LWC = liquid water content.
    Integral f_m(m) dm = DNC = droplet number concentration (1/m^3)
    m_avg = M/N = LWC/DNC, where
    M = total droplet mass in dV, N = tot. # of droplets in dV

    Parameters
    ----------
    m: float
        Mass
    DNC: float
        Droplet number concentration (1/m^3)
    DNC_over_LWC: float
        Droplet number concentration over liquid water content
        LWC in mass/volume
    par: ndarray
        par[0] = expected value
        par[1] = standard deviation
    
    Returns
    -------
        float
        Mass distribution evaluated at 'm'
    
    """
    
    return DNC * DNC_over_LWC * np.exp(-DNC_over_LWC * m)
conc_per_mass_expo = njit()(conc_per_mass_expo_np)

two_pi_sqrt = np.sqrt(2.0*np.pi)
def conc_per_mass_lognormal_np(m, DNC, mu_log, sigma_log):
    """Lognormal number concentration density distribution per mass
    
    Also called 'mass distribution'
    Integral f_m(m) dm = DNC = droplet number concentration (1/m^3)

    Parameters
    ----------
    m: float
        Mass
    DNC: float
        Droplet number concentration (1/m^3)
    mu_log:
        log. nat. of the geometric mean
    sigma_log: float
        log. nat. of the geometric standard deviation (also called 'mode')
    
    Returns
    -------
        float
        Mass distribution evaluated at 'm'
    
    """
    
    return DNC * np.exp( -0.5*( ( np.log( m ) - mu_log ) / sigma_log )**2 ) \
           / (m * two_pi_sqrt * sigma_log)
conc_per_mass_lognormal = njit()(conc_per_mass_lognormal_np)

#%% MOMENTS

def compute_moments_expo(n, DNC0, LWC0):
    """Computes moments of the exponential mass distribution
    
    Parameters
    ----------
    n: int
        Moment order
    DNC0: float
        Droplet number concentration (1/m^3)
    LWC0: float
        Liquid water content in mass/volume
    
    Returns
    -------
        float
        Moment of order 'n' of the exponential mass distribution
    
    """
    
    if n == 0:
        return DNC0
    elif n == 1:
        return LWC0
    else:    
        return math.factorial(n) * DNC0 * (LWC0/DNC0)**n

def moments_analytical_expo(n, DNC, DNC_over_LWC):
    """Computes moments of the exponential mass distribution
    
    Parameters
    ----------
    n: int
        Moment order
    DNC: float
        Droplet number concentration (1/m^3)
    DNC_over_LWC: float
        Droplet number concentration over liquid water content
        LWC in mass/volume
    
    Returns
    -------
        float
        Moment of order 'n' of the exponential mass distribution
    
    """
    
    if n == 0:
        return DNC
    else:
        LWC_over_DNC = 1.0 / DNC_over_LWC
        return math.factorial(n) * DNC * LWC_over_DNC**n

def moments_analytical_lognormal_m(n, DNC, mu_m_log, sigma_m_log):
    """Computes moments of the lognormal mass distribution
    
    Parameters
    ----------
    n: int
        Moment order
    DNC: float
        Droplet number concentration (1/m^3)
    mu_m_log:
        log. nat. of the geometric mean
    sigma_m_log: float
        log. nat. of the geometric standard deviation (also called 'mode')
    
    Returns
    -------
        float
        Moment of order 'n' of the lognormal mass distribution
    
    """
    
    if n == 0:
        return DNC
    else:
        return DNC * np.exp(n * mu_m_log + 0.5 * n*n * sigma_m_log*sigma_m_log)

def moments_analytical_lognormal_R(n, DNC, mu_R_log, sigma_R_log):
    """Computes moments of the lognormal radius distribution
    
    Parameters
    ----------
    n: int
        Moment order
    DNC: float
        Droplet number concentration (1/m^3)
    mu_R_log:
        log. nat. of the geometric mean
    sigma_R_log: float
        log. nat. of the geometric standard deviation (also called 'mode')
    
    Returns
    -------
        float
        Moment of order 'n' of the lognormal radius distribution
    
    """
    
    if n == 0:
        return DNC
    else:
        return DNC * np.exp(n * mu_R_log + 0.5 * n*n * sigma_R_log*sigma_R_log)

#%% NUMERICAL INTEGRATION

def num_int_np(func, x0, x1, steps=1E5):
    """Simple (1st order) numerical integration of a function over an interval
    
    Interval [x0, x1]
    
    Parameters
    ----------
    func: function
        Function of a single argument, f(x), to be integrated
    x0: int
        Lower border
    x1: int
        Upper border
    steps: int or float
        Number of steps for the numerical integration
    
    Returns
    -------
    intl: float    
        Result of the numerical integration
    
    """
    
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    while (x < x1):
        intl += dx * func(x)
        x += dx
    return intl
# num_int = njit()(num_int_np)
