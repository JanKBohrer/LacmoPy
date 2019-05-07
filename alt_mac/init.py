#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 14:26:01 2019

@author: bohrer
"""

# create a number of superparticles
# needs:
# 2D grid with fixed cells -> parameters of the grid
# parameters of the distribution -> N modes, mu_i, sigma_i
# number of super-particles per cell no_ppc
# returns:
# 2D array with positions pos[i,j]: pos[0] = x[j], pos[1] = z[j] 
# 2D array with positions pos[i,j]: pos[0] = x[j], pos[1] = z[j] 
# array with masses masses[i,j], i = 0,1,2,...Ns: m[0] = m_w[j], j = 0...Np-1, m[1] = m_s[j]
# 

def init(grid, dist_paras, no_ppc):
    
    # go through cells:
    # number of particles per cell is fix
    # the total number per real particles is also given
    # so the total number of particles per cell is given
    # the multiplicities have to be chosen randomly such that the statistical properties of the 
    # overall distribution is fullfilled
    
    
    return pos, masses