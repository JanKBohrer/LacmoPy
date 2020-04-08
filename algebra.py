#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
some basic algebra
"""

from numba import njit

# par[0] belongs to the largest exponential x^(N-1) for par[i], i = 0, .., N 
@njit()
def compute_polynom(par, x):
    """Computes the polynom p(x)

    p(x) = par[0]*x^(N-1) + par[1]*x^(N-2) + ... + par[N]

    Parameters
    ----------
    par: ndarray, dtype=float
        1D array with the polynom parameters. par[n], n = 0, ..., N.
        The order of the polynom is N-1
    x: float
        Argument of the polynom function
    
    Returns
    -------
    res: float
        Polynom evaluated at x
    
    """
    
    res = par[0] * x + par[1]
    for a in par[2:]:
        res = res * x + a
    return res
