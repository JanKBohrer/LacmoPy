#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
some basic algebra
"""

import math
from numba import njit

# par[0] belongs to the largest exponential x^(N-1) for par[i], i = 0, .., N 
@njit()
def compute_polynom(par, x):
    res = par[0] * x + par[1]
    for a in par[2:]:
        res = res * x + a
    return res

# VECTOR LENGTH
# vector must be a list or an array of more than one component
# the components may be scalars or np.arrays
@njit()
def vector_length(vector):
    r = 0.0
    for el in vector:
        r += el*el
    return math.sqrt(r)

# MAGNITUDE OF THE DEVIATION BETWEEN TWO VECTORS
# dev = sqrt( (v1 - v2)**2 )
# v1 and v2 may be lists or np.arrays of the same dimensions > 1
@njit()
def deviation_magnitude_between_vectors(v1, v2):
    dev = 0.0
    
    for i, comp1 in enumerate(v1):
        dev += (v2[i] - comp1)*(v2[i] - comp1)
    
    return math.sqrt(dev)