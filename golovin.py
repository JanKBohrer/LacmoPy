#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 19:10:17 2019

@author: bohrer
"""

#from scipy import special
#import scipy.special
import numpy as np

import mpmath as mpm
mpm.mp.dps = 25; mpm.mp.pretty = True

# init distr is n(x,0) = n0/x0 * exp(-x/x0)
# thus int n(x,0) dx = n0
#def dist_vs_time_golo_exp2(x,t,x0,n0,b):
#    t = np.where(t<1E-12,1E-12,t)
#    x_x0 = x/x0
#    x = np.where(x_x0 < 1E-8, x0 * 1E-8, x)
#    x_x0 = x/x0
#    T = 1. - np.exp(-b*n0*x0*t)
#    T_sqrt = np.sqrt(T)
#    Bessel_term = np.where(special.iv(1, 2. * x_x0 * T_sqrt) > 1E240,
#                           1E240,special.iv(1, 2. * x_x0 * T_sqrt))
#    return n0 / x0 * Bessel_term \
#           * (1. - T) * np.exp(-1. * x_x0 * (1. + T)) / (x_x0 * T_sqrt)

def dist_vs_time_golo_exp(x,t,x0,n0,b):
    res = np.zeros(len(x), dtype=np.float128)
    for n in range( len(x)):
        t = np.where(t<1E-12,1E-12,t)
        x_x0 = x[n]/x0
        x = np.where(x_x0 < 1E-8, x0 * 1E-8, x)
        x_x0 = x[n]/x0
        T = 1. - np.exp(-b*n0*x0*t)
        T_sqrt = np.sqrt(T)
        z = 2. * x_x0 * T_sqrt
        res[n] = n0 / x0 * (1. - T) * 2.\
                 * mpm.exp( -0.5 * (1. + T) * z / T_sqrt )\
                 * mpm.besseli(1,z) / z
    return res

def dst_m_expo(x,x0,n0):
    return n0 / x0 * np.exp(-x/x0)

def compute_moments_Golovin(t, n, DNC, LWC, b):
    if n == 0:
        mom = DNC * np.exp(-b*LWC*t)
    elif n == 1:
        mom = LWC
    elif n == 2:
        mom = 2* LWC * LWC / DNC * np.exp(2*b*LWC*t)
    elif n == 3:
        mom =  LWC**3 / (DNC**2)\
               * (12 * np.exp(4*b*LWC*t) - 6 * np.exp(3*b*LWC*t))
    return mom
