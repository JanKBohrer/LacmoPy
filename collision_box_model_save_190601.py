#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 27 16:19:17 2019

@author: bohrer

Simulate the collision of a given population of warm cloud droplets in a given
grid box of size dV = dx * dy * dz
The ensemble of droplets is considered to be well mixed inside the box volume,
such that the probability of each droplet to be at a certain position is
equal for all possible positions in the box

Possible Kernels to test:
    
Golovin kernel:
    
K_ij = b * (m_i + m_j)

-> analytic solution for specific initial distributions
    
Hydrodynamic Collection kernel:
    
K_ij = E(i,j) * pi * (R_i + R_j)**2 * |v_i - v_j|

for the box model the sedimentation velocity is a direct function
of the droplet radii
we need some description or tabulated values for the collection
efficiency E(i,j), e.g. (Long, 1974; Hall, 1980;
Wang et al., 2006) (in Unterstrasser)

(we can later add a kernel for small particles "aerosols", which has
another form (Brownian motion))

Note that the kernel is defined here by

P_ij = K_ij * dt/dV = Probability, that droplets i and j in dV will coalesce
in a time interval (t, t + dt)

Dimension check:
K = m^2 * m / s = m^3/s
P = m^3/s * s/m^3 = 1
"""

#%% MODULE IMPORTS

import math
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import os

import constants as c
from microphysics import compute_mass_from_radius
from microphysics import compute_radius_from_mass

# from collision import generate_permutation
# from collision import simulate_collisions

def dst_expo(x,k):
    return np.exp(-x*k) * k

#%% INIT
simdata_path = "/home/jdesk/OneDrive/python/sim_data/"
folder = "collision_box_model/"
path = simdata_path + folder
if not os.path.exists(path):
    os.makedirs(path)

#dx = 1 # m
#dy = 1 # m
#dz = 1 # m

# dx = 0.02 # m
# dy = 0.02 # m
# dz = 0.02 # m

dx = 0.1 # m
# dy = 0.002 # m
dy = 0.1 # m
dz = 0.1 # m

dt = 1.0
# dt = 1.0
store_every = 600
t_end = 3600.0

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

fig_name = "mass_distribution_init.png"
fig, ax = plt.subplots(figsize=(8,8))
m_ = np.linspace(0.0,m_max, 10000)
ax.hist(masses, density=True, bins=50)
ax.plot(m_, dst_expo(m_, 1.0/mu))
ax.set_yscale("log")
ax.set_xlabel("particle mass (fg)")
ax.set_ylabel("normalized counts")
ax.set_title(
f"n0={n0:.3} cm^-3, LWC0={LWC0} g/cm^3, # SIP={no_spct:.2e}", pad=20)
fig.tight_layout()
fig.savefig(path + fig_name)

#%% SIMULATION

print("start sim here")
from collision import simulate_collisions_np
from collision import simulate_collisions
times, concentrations, masses_vs_time, xis_vs_time =\
    simulate_collisions(xi, masses, dV, dt, t_end, store_every)
    # simulate_collisions_np(xi, masses, dV, dt, t_end, store_every)

np.save(path + "conc.npy",concentrations)
np.save(path + "times.npy",times)
np.save(path + "masses_vs_time.npy",masses_vs_time)
np.save(path + "xis_vs_time.npy",xis_vs_time)

# print(xi)
# print(masses)

#%% runtime tests

# from analysis import compare_functions_run_time
# funcs = ["simulate_collisions_np"]
# funcs = ["simulate_collisions"]
# pars = "xi, masses, dt, t_end, store_every"
# rs = [5,5]
# ns = [10,10]
# compare_functions_run_time(funcs, pars, rs, ns, globals_ = globals())
