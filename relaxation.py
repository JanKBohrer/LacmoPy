#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:40:20 2020

@author: bohrer
"""

import numpy as np
from numba import njit

# ICMW 2012 Test case 1 (Muhlbauer 2013, used in Arabas 2015)
# returns relax_time(z) profile in seconds
def compute_relaxation_time_profile(z):
    return 300 * np.exp(z/200)

# field is an 2D array(x,z)
# profile0 is an 1D profile(z)
# t_relax is the relaxation time profile(z)
# dt is the time step
# return: relaxation source term profile(z) for one time step dt
@njit()    
def compute_relaxation_term(field, profile0, t_relax, dt):
#    return dt * (profile0 - np.average(field, axis = 0)) / t_relax
    return dt * (profile0 - np.sum(field, axis = 0) / field.shape[0]) / t_relax
    