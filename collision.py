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

#%% GOLOVIN KERNEL

# P_ij = dt/dV K_ij
# K_ij = b * (m_i + m_j)
# b = 1.5 m^3/(kg s) (Unterstrasser 2017, p. 1535)
@njit()
def kernel_golovin(m_i, m_j):
    return 1.5E-18 * (m_i + m_j)

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
        return 9.44E9 * 1.0E-24 * four_pi_over_three**2 * (R_i**6 + R_j**6)
    else:
        # return 5.78E-9 * (m_i + m_j) / c.mass_density_water_liquid_NTP
        # return 5.78E-12 * (m_i + m_j) / c.mass_density_water_liquid_NTP
        # return 5.78E3 * 1.0E-12 (m_i + m_j) / c.mass_density_water_liquid_NTP
        return 5.78E3 * 1.0E-12 * four_pi_over_three * (R_i**3 + R_j**3)
           
# m_i = 1E6
# # m_j = 1E7
# R_i = compute_radius_from_mass_jit(m_i, c.mass_density_water_liquid_NTP)
# # R_j = compute_radius_from_mass_jit(m_j, c.mass_density_water_liquid_NTP)
# print(R_i)
# # print(R_j)

# kernel_lon = []
# kernel_gol = []

# # print(kernel_long(m_i,m_j))
# R_range = np.arange(10,100,1)
# m_range = compute_mass_from_radius(R_range, c.mass_density_water_liquid_NTP)
# for m_j in m_range:
#     kernel_gol.append(kernel_golovin(m_i, m_j))
#     kernel_lon.append(kernel_long(m_i, m_j))

# kernel_gol = np.array(kernel_gol)
# kernel_lon = np.array(kernel_lon)

# import matplotlib.pyplot as plt

# # R_j = np.arange(10,100,10)
# # R_range
# plt.plot(m_range, kernel_gol,"o")
# plt.plot(m_range, kernel_lon,"x")
# m_50 = compute_mass_from_radius(50, c.mass_density_water_liquid_NTP)
# plt.vlines(m_50,
#            0.0, kernel_long(m_i,m_50))

#%% SHIMA alg 01 ALPHA
    
@njit()
def collision_step_golovin(xi, masses, dV, dt):
    # xi is a 1D array of multiplicities
    # elements of xi can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xi)[0]
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
    # map onto the original xi-list, to get the indices for that list
    
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
        if xi[i] >= xi[j]:
            xi_max = xi[i]
            ind_max = i
            xi_min = xi[j]
            ind_min = j
        else:
            xi_max = xi[j]
            ind_max = j
            xi_min = xi[i]
            ind_min = i
            
        # xi_max = max(xi[i], xi[j])
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xi_i, xi_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xi_max\
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
            # xi_min = min(xi[i], xi[j])
            g = min(g,int(xi_max/xi_min))
            if xi_max - g*xi_min > 0:
                # print("case 1")
                xi[ind_max] -= g * xi_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xi[ind_max] = int(0.5 * xi_min)
                xi[ind_min] -= int(0.5 * xi_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]

@njit()
def collision_step_long(xi, masses, dV, dt):
    # xi is a 1D array of multiplicities
    # elements of xi can be 0, indicating vanished droplets
    # look for the indices, which are not zero
    indices = np.nonzero(xi)[0]
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
    # map onto the original xi-list, to get the indices for that list
    
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
        if xi[i] >= xi[j]:
            xi_max = xi[i]
            ind_max = i
            xi_min = xi[j]
            ind_min = j
        else:
            xi_max = xi[j]
            ind_max = j
            xi_min = xi[i]
            ind_min = i
            
        # xi_max = max(xi[i], xi[j])
        # now we need p = P_ij,s N_s*(N_s-1)/2 / (no_pairs),
        # where N_s = no_spct and
        # P_ij,s = max(xi_i, xi_j) P_ij
        # P_ij = K_ij dt/dV
        pair_prob = pair_kernel * dt/dV * xi_max\
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
            # xi_min = min(xi[i], xi[j])
            g = min(g,int(xi_max/xi_min))
            if xi_max - g*xi_min > 0:
                # print("case 1")
                xi[ind_max] -= g * xi_min
                masses[ind_min] += g * masses[ind_max]
            else:
                # print("case 2")
                xi[ind_max] = int(0.5 * xi_min)
                xi[ind_min] -= int(0.5 * xi_min)
                masses[ind_min] += g * masses[ind_max]
                masses[ind_max] = masses[ind_min]

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
