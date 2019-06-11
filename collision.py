#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:10:24 2019

@author: bohrer

this module contains functions which are needed to describe the collision
of warm cloud droplets
"""

import numpy as np
from numba import njit
import math

import constants as c
from microphysics import compute_radius_from_mass_jit
from microphysics import compute_mass_from_radius
from grid import bilinear_weight


#%% PERMUTATION
# returns a permutation of the list of integers [0,1,2,..,N-1] by Shima method
@njit()
def generate_permutation(N):
    permutation = np.zeros(N, dtype=np.int64)
    for n_next in range(1, N):
        q = np.random.randint(0,n_next+1)
        if q==n_next:
            permutation[n_next] = n_next
        else:
            permutation[n_next] = permutation[q]
            permutation[q] = n_next
    return permutation

#%% COLLECTION, COLLISION, COALESCENCES EFFICIENCIES
# E_collection = E_collision * E_coalescence

### Hall 1980: Collision Efficiency table -> assume E_coalescence = 1.0 f.a. R

# in Hall_E_col:
# row =  collector drop radius (radius of larger drop)
# column = ratio of R_small/R_large "collector ratio"  
Hall_E_col = np.load("Hall_collision_efficiency.npy")
Hall_R_col = np.load("Hall_collector_radius.npy")
Hall_R_col_ratio = np.load("Hall_radius_ratio.npy")
@njit()
def linear_weight(i, weight, f):
    return f[i+1]*weight + f[i]*(1.0-weight)
@njit()
def compute_E_col_Hall(R_i, R_j):
    if R_i <= 0.0 or R_j <= 0.0:
        return 0.0
    if R_i < R_j:
        R_col = R_j
        R_ratio = R_i/R_j
    else:
        R_col = R_i
        R_ratio = R_j/R_i
    if R_col > 300.0:
        return 1.0
    else:
        # NOTE that ind_col is for index of R_collection,
        # which indicates the row of Hall_E_col, and NOT the coloumn
        ind_col = int(R_col/10.0)
        ind_ratio = int(R_ratio/0.05)
        if ind_col == Hall_R_col.shape[0]-1:
            if ind_ratio == Hall_R_col_ratio.shape[0]-1:
                return 1.0
            else:
                weight = (R_ratio - ind_ratio * 0.05) / 0.05
                return linear_weight(ind_ratio, weight, Hall_E_col[ind_col])
        elif ind_ratio == Hall_R_col_ratio.shape[0]-1:
            weight = (R_col - ind_col * 10.0) / 10.0
            return linear_weight(ind_col, weight, Hall_E_col[:,ind_ratio])
        else:
            weight_1 = (R_col - ind_col * 10.0) / 10.0
            weight_2 = (R_ratio - ind_ratio * 0.05) / 0.05
#        print(R_col, R_ratio)
#        print(ind_col, ind_ratio, weight_1, weight_2)
            return bilinear_weight(ind_col, ind_ratio,
                                   weight_1, weight_2, Hall_E_col)
        # E_col = bilinear_weight(ind_1, ind_2, weight_1, weight_2, Hall_E_col)

#%% COLLECTION KERNELS

from atmosphere import compute_viscosity_air
viscosity_air_NTP = compute_viscosity_air(300.0)

### terminal velocity dependent on the droplet radius 
# (under some assumptions for the environment variables (T, p) and density of
# water for the droplets)
# radius R in mu       
#def compute_terminal_velocity(R):
#    k_0 = 4.5 * viscosity_air_NTP / (c.mass_density_water_liquid_NTP * R * R)
fac_Cd_Re_sq = 32.0\
           * (c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP)\
           * c.mass_density_air_dry_NTP * c.earth_gravity\
           / (3.0 * viscosity_air_NTP * viscosity_air_NTP)
def compute_Cd_Re_sq(R):
    return fac_Cd_Re_sq * R * R * R
fac_v_from_Re = viscosity_air_NTP/(2.0*c.mass_density_water_liquid_NTP)
# for intermediate 35 mu <= R < 300 mu
def solve_v_iteratively_long(R):
    if R <= 100:
        Re = 10
    else:
        Re = 100
    R *= 1.0E-6    
    Cd_Re_sq = compute_Cd_Re_sq(R)
    Cd_Re_sq_star = 100.0 * Cd_Re_sq
    
    while abs(Cd_Re_sq_star - Cd_Re_sq) / Cd_Re_sq > 1.0E-3:
        ln_Re = math.log(Re)
        y = -2.44461E-2 * ln_Re - 6.98404E-3 * ln_Re*ln_Re\
            + 8.88634E-4 * ln_Re**3
        bracket_term = 24.0 * (1.0 + 0.10229 * Re**(0.94015 + y))
        Cd_Re_sq_star = Re * bracket_term
        Re = Re - (Cd_Re_sq_star - Cd_Re_sq) / bracket_term
    print(Re)
    return 1.0E2 * fac_v_from_Re * Re / R 

# par[0] belongs to the largest exponential x^(n-1) for par[i], i = 0, .., n 
@njit()
def compute_polynom(par,x):
    res = par[0] * x + par[1]
    for a in par[2:]:
        res = res * x + a
    return res

# import matplotlib.pyplot as plt
# x = np.linspace(-10.0,10.0,100)
# par = (0.1,0.2,0.05,0.04)
# y1 = compute_polynom(par, x)
# y2 = np.polyval(par,x)
# plt.plot(x, (y1-y2)/y1*1E25)
# # plt.plot(x, y2)

# terminal velocity as fct of R by Long 1974b
# "five sections":
# junction points = 15, 35, 300, 800
# NOTE: "assumes p = 1013 mb, T = 20 C, RH = 100 %
v_35 = 13.721114748125
v_300 = 244.10941058000004
fac_vT_1 = 2.0 * c.earth_gravity\
            * (c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP) \
            / (9.0 * viscosity_air_NTP)
@njit()
def compute_terminal_velocity_long(R):
    if R < 15:
        R *= 1.0E-6
        v = fac_vT_1 * R * R * 1.0E2
    elif R < 35:
        R *= 1.0E-4
        v = compute_polynom((-1.917038E13,
                        2.432383E11,
                        -1.263519E9,
                        4.243126E6,
                        -3.417072E3,
                        1.443646), R)
#        v = np.polyval((-1.917038E13,
#                        2.432383E11,
#                        -1.263519E9,
#                        4.243126E6,
#                        -3.417072E3,
#                        1.443646), R)
    elif R < 300:
#        v = solve_v_iteratively_Long(R)
        v = (v_300 - v_35) / (300 - 35) * (R-35) + v_35
    elif R < 800:
        R *=1.0E-4
        v = compute_polynom( (-1.135044E8, 3.971905E7, -5.064069E6,
                         2.559450E5, 2.865716E3, 3.510302E1), R )
#        v = np.polyval( (-1.135044E8, 3.971905E7, -5.064069E6,
#                         2.559450E5, 2.865716E3, 3.510302E1), R )
    else:
        R *= 1.0E-4
        D = compute_polynom( (2.036791E5, -3.815343E4,4.516634E3,
                         -8.020389E1, 1.44274121E1,1.0), (R-0.045))
#        D = np.polyval( (2.036791E5, -3.815343E4,4.516634E3,
#                         -8.020389E1, 1.44274121E1,1.0), (R-0.045))
        v = -(554.5/D) + 921.5 - (4.0E-3 / (R - 0.021))
    return v * 0.01

#v_35 = compute_terminal_velocity_Long(35)
#v_300 = compute_terminal_velocity_Long(300)

#v = []
##R = np.arange(1,35,1)
##R = np.arange(1,1200,1)
#R = np.arange(1,1200,1)
#for R_ in R:
#    v.append(compute_terminal_velocity_Long(R_))
#
#import matplotlib.pyplot as plt
#plt.plot(R,v)

# P_ij = dt/dV K_ij
# K_ij = b * (m_i + m_j)
# b = 1.5 m^3/(kg s) (Unterstrasser 2017, p. 1535)
@njit()
def kernel_golovin(m_i, m_j):
    return 1.5E-18 * (m_i + m_j)

# R_i, R_j in mu
# 1E-12 to get m^3/s for the kernel
@njit()
def kernel_hydro(R_i, R_j, E_col, v_dif):
    return math.pi * (R_i + R_j)**2 * E_col * v_dif * 1.0E-12

# Hall 1980
# use hydrodynamic kernel with terminal velocity dependent on R_i
# and tabulated collection efficiencies from the paper
@njit()
def kernel_hall(m_i, m_j):
    R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
    R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
    E_col = compute_E_col_Hall(R_i, R_j)
    dv = compute_terminal_velocity_long(R_i)\
         - compute_terminal_velocity_long(R_j)
    return kernel_hydro(R_i, R_j, E_col, abs(dv))

# Long 1974 
# Kernel (two versions (?) he says: might be approx by ... OR ... (?))
# First:
# R < 50 mu: 9.44E9 * (Vi^2 + Vj^2)
# R > 50 mu: 5.78E3 * (Vi + Vj)
# Second:
# 1.1E10 * Vi^2 (no Vj)
# 6.33E3 Vi     (no Vj)
# here R = radius of LARGER drop, Vi = volume of larger drop in cm^3
# implement first version:
# R_i in mu
# m_i in fg
four_pi_over_three = 4.0/3.0*math.pi
@njit()
def kernel_long(m_i, m_j):
    R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
    R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
    # R_max = max(R_i, R_j):
    if R_i <= 50 and R_j <= 50:
        # return 9.44E-15 * (m_i*m_i + m_j*m_j)\
        #        / (c.mass_density_water_liquid_NTP
        #           * c.mass_density_water_liquid_NTP)
        # return 9.44E-19 * (m_i*m_i + m_j*m_j)\
        #        / (c.mass_density_water_liquid_NTP
        #           * c.mass_density_water_liquid_NTP)
        # return 9.44E9*1.0E-24*(m_i*m_i + m_j*m_j)\
        #        / (c.mass_density_water_liquid_NTP
        #           * c.mass_density_water_liquid_NTP)
        return 9.44E9 * 1.0E-24 * 1.0E-6 * four_pi_over_three**2 * (R_i**6 + R_j**6)
    else:
        # return 5.78E-9 * (m_i + m_j) / c.mass_density_water_liquid_NTP
        # return 5.78E-12 * (m_i + m_j) / c.mass_density_water_liquid_NTP
        # return 5.78E3 * 1.0E-12 (m_i + m_j) / c.mass_density_water_liquid_NTP
        return 5.78E3 * 1.0E-12 * 1.0E-6 * four_pi_over_three * (R_i**3 + R_j**3)

#m_i = 1E6
# # m_j = 1E7
#R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
# # R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
#print(R_i)
# # print(R_j)
#
#kernel_lon = []
#kernel_hal = []
#kernel_gol = []
#
# # print(kernel_long(m_i,m_j))
#R_range = np.arange(10,1000,1)
#m_range = compute_mass_from_radius(R_range, c.mass_density_water_liquid_NTP)
#for m_j in m_range:
#    kernel_gol.append(kernel_golovin(m_i, m_j))
#    kernel_lon.append(kernel_long(m_i, m_j))
#    kernel_hal.append(kernel_hall(m_i, m_j))
#
#kernel_gol = np.array(kernel_gol)
#kernel_lon = np.array(kernel_lon)
#kernel_hal = np.array(kernel_hal)
#
#import matplotlib.pyplot as plt
#
## R_j = np.arange(10,100,10)
## R_range
#plt.plot(m_range, kernel_gol,"o")
#plt.plot(m_range, kernel_hal,"x")
#plt.plot(m_range, kernel_lon,"d")
#plt.xscale("log")
#plt.yscale("log")
#m_50 = compute_mass_from_radius(50, c.mass_density_water_liquid_NTP)
##plt.vlines(m_50,
##           0.0, kernel_long(m_i,m_50))

#%% SHIMA alg 01 ALPHA
    
@njit()
def collision_step_golovin(xis, masses, dV, dt):
    # xis is a 1D array of multiplicities
    # elements of xis can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    # no_spct = 5940
    # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
    ### 1. make a random permutation of I by
    permutation = generate_permutation(no_spct)
    # print("permutation len")
    # print(len(permutation))
    
    ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
    no_pairs = no_spct//2
    cand_i = permutation[0:no_pairs]
    cand_j = permutation[no_pairs:2*no_pairs]
    # map onto the original xis-list, to get the indices for that list
    
    cand_i = indices[cand_i]
    cand_j = indices[cand_j]
    
    # print("cand_i len")
    # print(len(cand_i))
    # print("cand_j len")
    # print(len(cand_j))
    
    ### 3. for each pair (i,j) generate uniform float random number from [0,1)
    rnd01 = np.random.rand(no_pairs)
    # print("len(rnd01)")
    # print(len(rnd01))
    for pair_n in range(no_pairs):
        i = cand_i[pair_n]
        j = cand_j[pair_n]
        # i_done.append(i)
        # j_done.append(j)
        pair_kernel =\
            kernel_golovin(masses[i], masses[j])
        if xis[i] >= xis[j]:
            xis_max = xis[i]
            ind_max = i
            xis_min = xis[j]
            ind_min = j
        else:
            xis_max = xis[j]
            ind_max = j
            xis_min = xis[i]
            ind_min = i
            
        # xis_max = max(xis[i], xis[j])
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xis_i, xis_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xis_max\
                    * 0.5 * no_spct * (no_spct - 1) / no_pairs
        # then evaluate
        # beta
        # = [p] + 1; if rnd01 < p - [p]
        # = [p]    ; if rnd01 >= p - [p]
        pair_prob_int = int(pair_prob)
        if rnd01[pair_n] < pair_prob - pair_prob_int:
            g = pair_prob_int + 1
        else: g = pair_prob_int
        # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
        if g > 0:
            # print("collision for pair", i, j)
            # xis_min = min(xis[i], xis[j])
            g = min(g,int(xis_max/xis_min))
            if xis_max - g*xis_min > 0:
                # print("case 1")
                xis[ind_max] -= g * xis_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xis[ind_max] = int(0.5 * xis_min)
                xis[ind_min] -= int(0.5 * xis_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]

### Coll step for Hall Kernel
@njit()
def collision_step_hall(xis, masses, dV, dt):
    # xis is a 1D array of multiplicities
    # elements of xis can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    # no_spct = 5940
    # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
    ### 1. make a random permutation of I by
    permutation = generate_permutation(no_spct)
    # print("permutation len")
    # print(len(permutation))
    
    ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
    no_pairs = no_spct//2
    cand_i = permutation[0:no_pairs]
    cand_j = permutation[no_pairs:2*no_pairs]
    # map onto the original xis-list, to get the indices for that list
    
    cand_i = indices[cand_i]
    cand_j = indices[cand_j]
    
    # print("cand_i len")
    # print(len(cand_i))
    # print("cand_j len")
    # print(len(cand_j))
    
    ### 3. for each pair (i,j) generate uniform float random number from [0,1)
    rnd01 = np.random.rand(no_pairs)
    # print("len(rnd01)")
    # print(len(rnd01))
    for pair_n in range(no_pairs):
        i = cand_i[pair_n]
        j = cand_j[pair_n]
        # i_done.append(i)
        # j_done.append(j)
        pair_kernel =\
            kernel_hall(masses[i], masses[j])
        if xis[i] >= xis[j]:
            xis_max = xis[i]
            ind_max = i
            xis_min = xis[j]
            ind_min = j
        else:
            xis_max = xis[j]
            ind_max = j
            xis_min = xis[i]
            ind_min = i
            
        # xis_max = max(xis[i], xis[j])
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xis_i, xis_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xis_max\
                    * 0.5 * no_spct * (no_spct - 1) / no_pairs
        # then evaluate
        # beta
        # = [p] + 1; if rnd01 < p - [p]
        # = [p]    ; if rnd01 >= p - [p]
        pair_prob_int = int(pair_prob)
        if rnd01[pair_n] < pair_prob - pair_prob_int:
            g = pair_prob_int + 1
        else: g = pair_prob_int
        # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
        if g > 0:
            # print("collision for pair", i, j)
            # xis_min = min(xis[i], xis[j])
            g = min(g,int(xis_max/xis_min))
            if xis_max - g*xis_min > 0:
                # print("case 1")
                xis[ind_max] -= g * xis_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xis[ind_max] = int(0.5 * xis_min)
                xis[ind_min] -= int(0.5 * xis_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]

@njit()
def collision_step_long(xis, masses, dV, dt):
    # xis is a 1D array of multiplicities
    # elements of xis can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    # no_spct = 5940
    # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
    ### 1. make a random permutation of I by
    permutation = generate_permutation(no_spct)
    # print("permutation len")
    # print(len(permutation))
    
    ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
    no_pairs = no_spct//2
    cand_i = permutation[0:no_pairs]
    cand_j = permutation[no_pairs:2*no_pairs]
    # map onto the original xis-list, to get the indices for that list
    
    cand_i = indices[cand_i]
    cand_j = indices[cand_j]
    
    # print("cand_i len")
    # print(len(cand_i))
    # print("cand_j len")
    # print(len(cand_j))
    
    ### 3. for each pair (i,j) generate uniform float random number from [0,1)
    rnd01 = np.random.rand(no_pairs)
    # print("len(rnd01)")
    # print(len(rnd01))
    for pair_n in range(no_pairs):
        i = cand_i[pair_n]
        j = cand_j[pair_n]
        # i_done.append(i)
        # j_done.append(j)
        pair_kernel =\
            kernel_long(masses[i], masses[j])
        if xis[i] >= xis[j]:
            xis_max = xis[i]
            ind_max = i
            xis_min = xis[j]
            ind_min = j
        else:
            xis_max = xis[j]
            ind_max = j
            xis_min = xis[i]
            ind_min = i
            
        # xis_max = max(xis[i], xis[j])
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xis_i, xis_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xis_max\
                    * 0.5 * no_spct * (no_spct - 1) / no_pairs
        # then evaluate
        # beta
        # = [p] + 1; if rnd01 < p - [p]
        # = [p]    ; if rnd01 >= p - [p]
        pair_prob_int = int(pair_prob)
        if rnd01[pair_n] < pair_prob - pair_prob_int:
            g = pair_prob_int + 1
        else: g = pair_prob_int
        # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
        if g > 0:
            # print("collision for pair", i, j)
            # xis_min = min(xis[i], xis[j])
            g = min(g,int(xis_max/xis_min))
            if xis_max - g*xis_min > 0:
                # print("case 1")
                xis[ind_max] -= g * xis_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xis[ind_max] = int(0.5 * xis_min)
                xis[ind_min] -= int(0.5 * xis_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]


# coalescence events with wating time distribution:
# orig. by Gillespie 1972, 1975 for water droplets (direct = all weights = 1)
# "Efficient algorithm" by Goodson and Kraft 2002 (also direct, no multiplic.)
# with "stochastic weights": Patterson (WIAS) 2011, 2012
# IDEA: (Gillespie 1975 J. atm. sci.)
# joint PDF: P(tau,i,j) = prob, at time t, that the next coalescence will
# occur in the time interval (t+tau, t+tau+dtau)
# AND will be between particles "i" and "j":
# P(\tau,i,j) = C_ij exp [ -\sum_{k=1}^{N-1} \sum_{l=k+1}^N C_kl \tau ]
# We have int_0^inf dtau \sum_{i=1}^{N-1} \sum_{j=i+1}^N P(tau,i,j) = 1
# i.e. a discrete prob. in (i,j) and a PDF in tau
# he defines: C_kl dtau = prob, that particles k and l will coalesce in
# the next infinit. time interval dtau
# this is equivalent to
# the kernel K_kl divided by the cell volume dV C_kl = K_kl / dV
# using the "Full-conditional method" from Gillespie 1975, p. 1981, sec 4a.
# (no majorant or other techniques):
# P(tau,i,j) = P1(tau) * P2(i|tau) * P3(j|tau,i)
# C_i = sum_{j=i+1}^N C_ij (i = 1,...,N-1)
# C_0 = sum_{i=1}^{N-1} C_i
# 1. choose waiting time tau from PDF
# P1(tau) = C_0 exp(-C_0 tau)
# 2. generate "i" from discrete PDF P2(i):
# P2(i) = C_i/C_0
# 3. generate "j" from discrete PDF P3(j|tau,i) = P3(j|i):
# P3(j|i) = C_ij/Ci


def collision_step_tau_golovin(xis, masses, dV, dt):
    indices = np.nonzero(xis)[0]
    # N = no_spct ("number of super particles per cell total")
    # no_spct = len(indices)
    no_spct = indices.shape[0]
    
    # 2. choose waiting time tau from PDF P1(tau) = C_0 exp(-C_0 tau)
    # on the way: store all C_i = sum_{j=i+1}^N C_ij (i = 1,...,N-1)
    # the number of elements in this list is N-1, if N = no_spct
    C_is = np.zeros(no_spct-1, dtype=np.float64)
    # this does not work, we need C_0 first
    # C_ijs = np.zeros(no_spct, dtype=np.float64) # the 0th index is not used
    for i in range(no_spct-1):
        ind_i = indices(i)
        # C_i = 0.0
        for j in range(i+1, no_spct):
            ind_j = indices(j)
            C_is[i] += kernel_golovin(masses[ind_i], masses[ind_j])
    rnd01 = np.random.rand()
    # 2. get the first index of the tuple (i,j)
    

#%%

# dV = 1.0E-6
# dt = 1.0
# # dt = 1.0
# store_every = 600
# t_end = 3600.0

# no_spc = 40

# # droplet concentration
# #n = 100 # cm^(-3)
# n0 = 297.0 # cm^(-3)
# # liquid water content (per volume)
# LWC0 = 1.0E-6 # g/cm^3
# # total number of droplets
# no_rpc = int(n0 * dV * 1.0E6)
# print("no_rpc=", no_rpc)

# # we start with a monomodal exponential distribution
# # mean droplet mass
# mu = 1.0E15*LWC0 / n0
# print("mu_m=", mu)
# from microphysics import compute_radius_from_mass
# import constants as c
# mu_R = compute_radius_from_mass(mu, c.mass_density_water_liquid_NTP)
# print("mu_R=", mu_R)
# total_mass_in_cell = dV*LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg

# masses = np.random.exponential(mu, no_rpc) # in mu
# xi = np.ones(no_rpc, dtype = np.int64)

# collision_step(xi, masses, dV, dt)



#%%

def simulate_collisions_np(xi, masses, dV, dt, t_end, store_every,
                           kernel="golovin"):
    if kernel == "long": collision_step = collision_step_long
    elif kernel == "golovin": collision_step = collision_step_golovin
    elif kernel == "hall": collision_step = collision_step_hall
    times = np.zeros(1+int(t_end/(store_every * dt)), dtype = np.float64)
    concentrations = np.zeros(1+int( math.ceil(t_end / (store_every * dt)) ),
                              dtype = np.float64)
    masses_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
                              dtype = np.float64)
    xis_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
                              dtype = np.int64)
    cnt_c = 0 

    for cnt, t in enumerate(np.arange(0.0,t_end,dt)):
        # print("sim at", t)
        if cnt%store_every == 0:
            times[cnt_c] = t
            concentrations[cnt_c] = np.sum(xi)/dV
            masses_vs_time[cnt_c] = masses
            xis_vs_time[cnt_c] = xi
            cnt_c += 1
        collision_step(xi, masses, dV, dt)  
        
    times[cnt_c] = t_end
    concentrations[cnt_c] = np.sum(xi)/dV
    masses_vs_time[cnt_c] = masses
    xis_vs_time[cnt_c] = xi
    
    return times, concentrations, masses_vs_time, xis_vs_time
# simulate_collisions = njit()(simulate_collisions_np)



#%% SHIMA ALGORITHM
# @njit()
# def collision_step(xi, masses, dV, dt):
#     # xi is a 1D array of multiplicities
#     # elements of xi can be 0, indicating vanished droplets
#     # look for the indices, which are not zero
#     indices = np.nonzero(xi)[0]
#     # N = no_spct ("number of super particles per cell total")
#     # no_spct = len(indices)
#     # no_spct = indices.shape[0]
#     no_spct = 5940
#     # Let I = [0,1,2,3,...,N-1] be the list of all EXISTING particle indices
    
#     ### 1. make a random permutation of I by
#     permutation = generate_permutation(no_spct)
#     # print("permutation len")
#     # print(len(permutation))
    
#     ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
#     no_pairs = no_spct//2
#     cand_i = permutation[0:no_pairs]
#     cand_j = permutation[no_pairs:2*no_pairs]
#     # map onto the original xi-list, to get the indices for that list
    
#     cand_i = indices[cand_i]
#     cand_j = indices[cand_j]
    
#     # print("cand_i len")
#     # print(len(cand_i))
#     # print("cand_j len")
#     # print(len(cand_j))
    
#     ### 3. for each pair (i,j) generate uniform float random number from [0,1)
#     rnd01 = np.random.rand(no_pairs)
#     # print("len(rnd01)")
#     # print(len(rnd01))
#     for pair_n in range(no_pairs):
#         i = cand_i[pair_n]
#         j = cand_j[pair_n]
#         # i_done.append(i)
#         # j_done.append(j)
#         pair_kernel =\
#             kernel_golovin(masses[i], masses[j])
#         if xi[i] >= xi[j]:
#             xi_max = xi[i]
#             ind_max = i
#             xi_min = xi[j]
#             ind_min = j
#         else:
#             xi_max = xi[j]
#             ind_max = j
#             xi_min = xi[i]
#             ind_min = i
            
#         # xi_max = max(xi[i], xi[j])
#         # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
#         # where N_s = no_spct and
#         # P_ij,s = max(xi_i, xi_j) P_ij
#         # P_ij = K_ij dt/dV
#         pair_prob = pair_kernel * dt/dV * xi_max\
#                     * 0.5 * no_spct * (no_spct - 1) / no_pairs
#         # then evaluate
#         # beta
#         # = [p] + 1; if rnd01 < p - [p]
#         # = [p]    ; if rnd01 >= p - [p]
#         pair_prob_int = int(pair_prob)
#         if rnd01[pair_n] < pair_prob - pair_prob_int:
#             g = pair_prob_int + 1
#         else: g = pair_prob_int
#         # print(pair_n, pair_prob, pair_prob_int, rnd01[pair_n], g)
#         if g > 0:
#             # print("collision for pair", i, j)
#             # xi_min = min(xi[i], xi[j])
#             g = min(g,int(xi_max/xi_min))
#             if xi_max - g*xi_min > 0:
#                 # print("case 1")
#                 xi[ind_max] -= g * xi_min
#                 masses[ind_min] += g * masses[ind_max]
#             else:
#                 # print("case 2")
#                 xi[ind_max] = int(0.5 * xi_min)
#                 xi[ind_min] -= int(0.5 * xi_min)
#                 masses[ind_min] += g * masses[ind_max]
#                 masses[ind_max] = masses[ind_min]

# # runtime tests njit/no jit:
# # -> without jit is slightly faster in this case ;)
# # no_spct= 297000
# # mu_m= 3367003.367003367
# # mu_R= 9.303487789072296
# # masses.shape= (297000,)
# # m_max= 48429996.65590665
# # simulate_collisions_np: repeats = 5 no reps =  10
# # simulate_collisions: repeats = 5 no reps =  10
# # best =  6.037e+05 us; worst =  1.163e+06 us; mean = 7.166e+05 +- 2.5e+05 us
# # best =  6.13e+05 us; worst =  1.236e+06 us; mean = 7.385e+05 +- 2.78e+05 us
# def simulate_collisions_np(xi, masses, dV, dt, t_end, store_every):
#     times = np.zeros(1+int(t_end/(store_every * dt)), dtype = np.float64)
#     concentrations = np.zeros(1+int( math.ceil(t_end / (store_every * dt)) ),
#                               dtype = np.float64)
#     masses_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
#                               dtype = np.float64)
#     xis_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
#                               dtype = np.float64)
#     cnt_c = 0 

#     for cnt, t in enumerate(np.arange(0.0,t_end,dt)):
#         if cnt%store_every == 0:
#             times[cnt_c] = t
#             concentrations[cnt_c] = np.sum(xi)/dV
#             masses_vs_time[cnt_c] = masses
#             xis_vs_time[cnt_c] = xi
#             cnt_c += 1
#         collision_step(xi, masses, dV, dt)  
        
#     times[cnt_c] = t_end
#     concentrations[cnt_c] = np.sum(xi)/dV
#     masses_vs_time[cnt_c] = masses
#     xis_vs_time[cnt_c] = xi
    
#     return times, concentrations, masses_vs_time, xis_vs_time
# simulate_collisions = njit()(simulate_collisions_np)
