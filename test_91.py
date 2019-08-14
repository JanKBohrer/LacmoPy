#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:16:21 2019

@author: bohrer
"""

import numpy as np
import matplotlib.pyplot as plt

import constants as c

import microphysics as mp

from scipy.optimize import curve_fit

from numba import vectorize, njit


T = 298.
#D_s_list = np.array([6,8,10,20,40,60]) * 1E-3
#R_s_list = D_s_list * 0.5

D_s = 10E-3 # mu = 10 nm
R_s = 0.5 * D_s
m_s = mp.compute_mass_from_radius_jit(R_s, c.mass_density_AS_dry)

w_s = np.logspace(-2., np.log10(0.78), 100)
rho_p = mp.compute_density_AS_solution(w_s, T) 
m_p = m_s/w_s
R_p = mp.compute_radius_from_mass_jit(m_p, rho_p)

sigma_p = mp.compute_surface_tension_AS(w_s, T)

S_eq = \
    mp.compute_kelvin_raoult_term_mf_AS(w_s,
                                     m_s,
                                     T,
                                     rho_p,
                                     sigma_p)
S_eq = \
    mp.compute_equilibrium_saturation_AS(w_s, R_p, rho_p, T, sigma_p)    
fig, ax = plt.subplots()
ax.plot(R_p, S_eq)    