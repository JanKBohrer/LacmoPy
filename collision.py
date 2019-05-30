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


#%% DISTRIBUTIONS

def dst_expo(x,k):
    return np.exp(-x*k) * k

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

#%% SHIMA alg 01 ALPHA
    
@njit()
def collision_step(xi, masses, dV, dt):
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

def simulate_collisions_np(xi, masses, dV, dt, t_end, store_every):
    times = np.zeros(1+int(t_end/(store_every * dt)), dtype = np.float64)
    concentrations = np.zeros(1+int( math.ceil(t_end / (store_every * dt)) ),
                              dtype = np.float64)
    masses_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
                              dtype = np.float64)
    xis_vs_time = np.zeros((1+int(t_end/(store_every * dt)), len(masses)),
                              dtype = np.float64)
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
simulate_collisions = njit()(simulate_collisions_np)

#%% GENERATE SIP ENSEMBLE

# no_spc is the intended number of super particles per cell,
# this will right on average, but will vary due to the random assigning 
# process of the xi_i
# m0, m1, dm: parameters for the numerical integration to find the cutoffs
# and for the numerical integration of integrals
# dV = volume size of the cell (in m^3)
# n0: initial particle number distribution (in 1/m^3)
# IN WORK: make the bin size smaller for low masses to get
# finer resolution here...
# -> "steer against" the expo PDF for low m -> something in between
# we need higher resolution in very small radii (Unterstrasser has
# R_min = 0.8 mu
# at least need to go down to 1 mu (resolution there...)
from init import compute_quantiles
def generate_SIP_ensemble(dst, par, no_spc, no_rpc, total_mass_in_cell,
                          p_min, p_max,
                          m0, m1, dm, seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    if par is not None:
        func = lambda x : dst(x, par)
    else: func = dst
    
    # define m_low, m_high by probability threshold p_min
    # m_thresh = [m_low, m_high] 
    (m_low, m_high), Ps = compute_quantiles(func, None, m0, m1, dm,
                                     [p_min, p_max], None)
    # if p_min == 0:
    #     m_low = 0.0
    # estimate value for dm, which is the bin size of the hypothetical
    # bin assignment, which is approximated by the random process:
    
    bin_size = (m_high - m_low) / no_spc
    
    # no of real particle cell
    # no_rpc = dV*n0
    
    # the current position of the left bin border
    m_left = m_low
    
    # generate list of masses and multiplicities
    masses = []
    xis = []
    
    # no of particles placed:
    no_pt = 0
    
    # let f(m) be the PDF!! (m can be the mass or the radius, depending,
    # which distr. is given), NOTE that f(m) is not the number concentration..
    
    pt_n = 0
    while no_pt < no_rpc:
        ### i) determine the next xi_mean by
        # integrating xi_mean = no_rpc * int_(m_left)^(m_left+dm) dm f(m)
        print(pt_n, "m_left=", m_left)
        m = m_left
        m_right = m_left + bin_size
        intl = 0.0
        cnt = 0
        while (m < m_right and cnt < 1E6):
            intl += dm * func(m)
            m += dm
            cnt += 1
            if cnt == 1E6:
                print(pt_n, "cnt=1E6")            
        xi_mean = no_rpc * intl
        print(pt_n, "xi_mean=", xi_mean)
        ### ii) draw xi from distribution: 
        # try here normal distribution if xi_mean > 10
        # else Poisson
        # with parms: mu = xi_mean, sigma = sqrt(xi_mean)
        if xi_mean > 10:
            # xi = np.random.normal(xi_mean, np.sqrt(xi_mean))
            xi = np.random.normal(xi_mean, 0.2*xi_mean)
        else:
            xi = np.random.poisson(xi_mean)
        xi = int(math.ceil(xi))
        if xi <= 0: xi = 1
        if no_pt + xi >= no_rpc:
            print("no_pt + xi=", no_pt + xi, "no_rpc =", no_rpc )
            xi = no_rpc - no_pt
            # last = True
            M = np.sum( np.array(xis)*np.array(masses) )
            # M = p_max * total_mass_in_cell - M
            M = total_mass_in_cell - M
            # mu = max(p_max*M/xi,m_left)
            mu = M/xi
            masses.append(mu)
        else:            
            ### iii) set the right right bin border
            # by no_rpc * int_(m_left)^m_right dm f(m) = xi
            m = m_left
            intl = 0.0
            cnt = 0
            while (intl < xi/no_rpc and cnt < 1E7):
                intl += dm * func(m)
                m += dm
                cnt += 1
                if cnt == 1E7:
                    print(pt_n, "cnt=", cnt)
            m_right = m
            # if m_right >= m_high:
                
            print(pt_n, "new m_right=", m_right)
            # iv) set SIP mass mu by mu=M/xi (M = total mass in the bin)
            intl = 0.0
            m = m_left
            cnt = 0
            while (m < m_right and cnt < 1E7):
                intl += dm * func(m) * m
                m += dm
                cnt += 1         
                if cnt == 1E7:
                    print(pt_n, "cnt=", cnt)                
            mu = intl * no_rpc / xi
            masses.append(mu)
        xis.append(xi)
        no_pt += xi
        print(pt_n, xi, mu, no_pt)
        pt_n += 1
        m_left = m_right
        
        if m_left >= m_high:
            print("m_left =", m_left, "m_left - m_high =", m_left-m_high)
            # print("no_pt + xi=", no_pt + xi, "no_rpc =", no_rpc )
            xi = no_rpc - no_pt
            # last = True
            M = np.sum( np.array(xis)*np.array(masses) )
            # M = p_max * total_mass_in_cell - M
            M = total_mass_in_cell - M
            # mu = max(p_max*M/xi,m_left)
            mu = M/xi
            masses.append(mu)    
            xis.append(xi)
            no_pt += xi
            print(pt_n, xi, mu, no_pt)
            pt_n += 1
    
    return np.array(masses), np.array(xis), m_low, m_high

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
