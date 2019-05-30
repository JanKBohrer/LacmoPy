#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 11:01:49 2019

@author: jdesk
"""

from collision import generate_permutation
import numpy as np


mvt = [
       [
        [111, 112],
        [121, 122]
       ],
       [
        [211, 212],
        [221, 222]
       ]
      ]

mvt = np.array(mvt)

mvt2 = np.concatenate(mvt, axis=1)

#%%

# indices = np.array( [5,7,11,14,15,16] )

# no_spct = len(indices)

# # indices = np.nonzero(xi)
# # N = no_spct ("number of super particles per cell total")
# no_spct = len(indices)
# permutation = generate_permutation(no_spct)
# print("permutation")
# print(permutation)
# ### 2. make [no_spct/2] candidate pairs [(i1,j1),(i2,j2),(),.]
# no_pairs = no_spct//2
# cand_i = permutation[0:no_pairs]
# cand_j = permutation[no_pairs:2*no_pairs]
# # map onto the original xi-list, to get the indices for that list
# print("cand_i")
# print(cand_i)
# print("cand_j")
# print(cand_j)

# # remap onto indices

# cand_i = indices[cand_i]
# cand_j = indices[cand_j]
# print("indices")
# print(indices)
# print("cand_i")
# print(cand_i)
# print("cand_j")
# print(cand_j)

# from numba import njit
# @njit()
# def simmm_np(t_end, dt, xi):
#     for cnt,t in enumerate(np.arange(0.0,t_end,dt)):
#         indices = np.nonzero(xi)[0]
#         no_spct = indices.shape[0]
#         permutation = generate_permutation(no_spct)
#         print(permutation)
        
# xi = np.ones(50, dtype=np.int64)

# t_e = 100.0
# dt = 1.0
# simmm_np(t_e, dt, xi)   



#%%
### SHIMA ALGORITHM
from collision import kernel_golovin
from numba import njit


dx = 0.01 # m
dy = 0.01 # m
# dy = 0.1 # m
dz = 0.01 # m

dV = dx * dy * dz

# we start with a monomodal exponential distribution
# droplet concentration
#n = 100 # cm^(-3)
n0 = 297.0 # cm^(-3)
# liquid water content (per volume)
LWC0 = 1.0E-6 # g/cm^3
# total number of droplets
no_spct = int(n0 * dV * 1.0E6)
print("no_spct=", no_spct)

# mean droplet mass
mu = 1.0E15*LWC0 / n0
print("mu_m=", mu)
from microphysics import compute_radius_from_mass
import constants as c
mu_R = compute_radius_from_mass(mu, c.mass_density_water_liquid_NTP)
print("mu_R=", mu_R)

# we need an initial distribution for wet droplets:
# here just for the water mass m
# Unterstrasser 2017 uses exponential distribution:
# f = 1/mu exp(m/mu)
# mean radius
# mu_R = 9.3 # mu
# mean mass
# mu = compute_mass_from_radius(mu_R, c.mass_density_water_liquid_NTP)
# print(mu)

seed = 4713
np.random.seed(seed)
masses = np.random.exponential(mu, no_spct) # in mu
print("masses.shape=", masses.shape)
m_max = np.amax(masses)
print("m_max=", m_max)
xi = np.ones(no_spct, dtype = np.int64)

dt = 1.0
t_end = 100.0
store_every = 10
from collision import collision_step
from collision import simulate_collisions_np
from collision import simulate_collisions
collision_step(xi, masses, dV, dt)
    
times, concentrations, masses_vs_time, xis_vs_time =\
    simulate_collisions(xi, masses, dV, dt, t_end, store_every)
    # simulate_collisions_np(xi, masses, dV, dt, t_end, store_every)
    # simulate_collisions(xi, masses, dV, dt, t_end, store_every)
    
    
    