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

from microphysics import compute_mass_from_radius

#%% DISTRIBUTIONS

# exponential prob. dens. fct. such that int ( f(x) dx ) = 1
# x_mean_inv = 1.0/x_avg, where x_mean is the mean (and the STD)
@njit()
def pdf_expo(x, x_mean_inv):
    return x_mean_inv * np.exp(- x * x_mean_inv)

# antiderivative of expo pdf
def prob_exp(x,k):
    return 1.0 - np.exp(-k*x)

def compute_moments_expo(n, DNC0, LWC0):
    if n == 0:
        return DNC0
    elif n == 1:
        return LWC0
    else:    
        return math.factorial(n) * DNC0 * (LWC0/DNC0)**n

# log-normal prob. dens. fct. (monomodal) such that int ( f(x) dx ) = 1
# mu_log is the ln() of the geometric mean "mu" (also "mode") of the dst
# sigma_log is the ln() of the geometric STD "sigma" of the dst
# it is x_mean := arithm. expect(x) = mu * exp( 0.5 * sigma_log**2)
# and x_std = SD(x) = x_mean * sqrt( exp( sigma_log**2 ) - 1 )
two_pi_sqrt = np.sqrt(2.0 * np.pi)
@njit()
def pdf_lognormal(x, mu_log, sigma_log):
    return np.exp( -0.5*( ( np.log( x ) - mu_log ) / sigma_log )**2 ) \
           / (x * two_pi_sqrt * sigma_log)

# normal prob. dens. fct. (monomodal) such that int ( f(x) dx ) = 1
two_pi_sqrt = math.sqrt(2.0 * math.pi)
def dst_normal(x, par):
    return np.exp( -0.5 * ( ( x - par[0] ) / par[1] )**2 ) \
           / (two_pi_sqrt * par[1])

# exponential number concentration density per mass in different form
# such that int f_m(m) dm = DNC = droplet number concentration (1/m^3)
# f_m(m) = 1/LWC * exp(-m/m_avg)
# LWC = liquid water content (kg/m^3)
# m_avg = M/N = LWC/DNC
# where M = total droplet mass in dV, N = tot. # of droplets in dV
# in this function f_m(m) = conc_per_mass(m, LWC_inv, DNC_over_LWC)
# DNC_over_LWC = 1/m_avg
# m in kg
# function moments checked versus analytical values via numerical integration
def conc_per_mass_expo_np(m, DNC, DNC_over_LWC): # = f_m(m)
    return DNC * DNC_over_LWC * np.exp(-DNC_over_LWC * m)
conc_per_mass_expo = njit()(conc_per_mass_expo_np)

# lognormal number concentration density per mass in different form
# such that int f_m(m) dm = DNC = droplet number concentration (1/m^3)
# mu = geometric expect.value of the PDF = "mode"
# sigma = geometric standard dev. of the PDF
# PDF for DNC = 1.0 tested: numeric integral = 1.0
two_pi_sqrt = np.sqrt(2.0*np.pi)
def conc_per_mass_lognormal_np(x, DNC, mu_log, sigma_log):
    return DNC * np.exp( -0.5*( ( np.log( x ) - mu_log ) / sigma_log )**2 ) \
           / (x * two_pi_sqrt * sigma_log)
conc_per_mass_lognormal = njit()(conc_per_mass_lognormal_np)

#%% NUMERICAL INTEGRATION

def num_int_np(func, x0, x1, steps=1E5):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    while (x < x1):
        intl += dx * func(x)
        x += dx
    return intl
# num_int = njit()(num_int_np)

def dst_expo_np(x,k):
    return np.exp(-x*k) * k
dst_expo = njit()(dst_expo_np)

def num_int_expo_np(x0, x1, k, steps=1E5):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = dst_expo(x,k)
    while (x < x1):
        f2 = dst_expo(x + 0.5*dx, k)
        f3 = dst_expo(x + dx, k)
        intl += 0.1666666666667 * dx * (f1 + 4 * f2 + f3)
        x += dx
        f1 = f3
    return intl
num_int_expo = njit()(num_int_expo_np)

def num_int_expo_mean_np(x0, x1, k, steps=1E5):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = dst_expo(x,k) * x
    while (x < x1):
        f2 = dst_expo(x + 0.5*dx, k) * (x + 0.5*dx)
        f3 = dst_expo(x + dx, k) * (x + dx)
        intl += 0.1666666666667 * dx * (f1 + 4 * f2 + f3)
        x += dx
        f1 = f3
    return intl
num_int_expo_mean = njit()(num_int_expo_mean_np)

def num_int_lognormal_np(x0, x1, par, steps=1E6):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = conc_per_mass_lognormal(x,par[0],par[1],par[2])
    # cnt = 0
    while (x < x1):
        f2 = conc_per_mass_lognormal(x + 0.5*dx, par[0],par[1],par[2])
        f3 = conc_per_mass_lognormal(x + dx, par[0],par[1],par[2])
        intl += 0.1666666666667 * dx * (f1 + 4.0 * f2 + f3)
        x += dx
        f1 = f3
    return intl
num_int_lognormal = njit()(num_int_lognormal_np)

def num_int_lognormal_mean_np(x0, x1, par, steps=1E6):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = conc_per_mass_lognormal(x,par[0],par[1],par[2]) * x
    # cnt = 0
    while (x < x1):
        f2 = conc_per_mass_lognormal(x + 0.5*dx, par[0],par[1],par[2]) \
             * (x + 0.5*dx)
        f3 = conc_per_mass_lognormal(x + dx, par[0],par[1],par[2]) \
             * (x + dx)
        intl += 0.1666666666667 * dx * (f1 + 4.0 * f2 + f3)
        x += dx
        f1 = f3
    return intl
num_int_lognormal_mean = njit()(num_int_lognormal_mean_np)

# x0 and x1 in microns
def num_int_lognormal_mean_mass_R_np(x0, x1, par, steps=1E6):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = conc_per_mass_lognormal(x,par[0],par[1],par[2]) \
         * 1.0E-18*compute_mass_from_radius(x, par[3])
    # cnt = 0
    while (x < x1):
        f2 = conc_per_mass_lognormal(x + 0.5*dx, par[0],par[1],par[2]) \
             * 1.0E-18*compute_mass_from_radius(x+0.5*dx, par[3])
        f3 = conc_per_mass_lognormal(x + dx, par[0],par[1],par[2]) \
             * 1.0E-18*compute_mass_from_radius(x+dx, par[3])
        intl += 0.1666666666667 * dx * (f1 + 4.0 * f2 + f3)
        x += dx
        f1 = f3
    return intl
num_int_lognormal_mean_mass_R = njit()(num_int_lognormal_mean_mass_R_np)

def num_int_impl_right_border(func, x0, intl_value, dx, cnt_lim=1E7):
    dx0 = dx
    x = x0
    intl = 0.0
    cnt = 0
    f1 = func(x)
    print('dx0 =', dx0)
    while (intl < intl_value and cnt < cnt_lim):
        f2 = func(x + 0.5*dx)
        f3 = func(x + dx)
        intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4 * f2 + f3)
        x += dx
        f1 = f3
        cnt += 1
    return x - dx + dx * (intl_value - intl_bef)/(intl - intl_bef)
    
def num_int_expo_impl_right_border_np(x0, intl_value, dx, k, cnt_lim=1E7):
    # dx0 = dx
    x = x0
    intl = 0.0
    cnt = 0
    f1 = dst_expo(x,k)
    while (intl < intl_value and cnt < cnt_lim):
        f2 = dst_expo(x + 0.5*dx, k)
        f3 = dst_expo(x + dx, k)
        intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4 * f2 + f3)
        x += dx
        f1 = f3
        cnt += 1
    return x - dx + dx * (intl_value - intl_bef)/(intl - intl_bef)
num_int_expo_impl_right_border = njit()(num_int_expo_impl_right_border_np)

def compute_right_border_impl_exp(x0, intl_value, k):
    return -1.0/k * np.log( np.exp(-k * x0) - intl_value )