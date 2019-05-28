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

#%%

import numpy as np
import matplotlib.pyplot as plt

import constants as c
from microphysics import compute_mass_from_radius

def dst_expo(x,k):
    return np.exp(-x*k) * k

#%%

#dx = 1 # m
#dy = 1 # m
#dz = 1 # m

dx = 0.05 # m
dy = 0.05 # m
dz = 0.05 # m

#dx = 0.01 # m
#dy = 0.01 # m
#dz = 0.01 # m

dV = dx * dy * dz

# we start with a monomodal distribution
# droplet concentration
#n = 100 # cm^(-3)
n = 297 # cm^(-3)
# total number of droplets
N = int(n * dV * 1.0E6)
print(N)

# we need an initial distribution for wet droplets:
# here just for the water mass m
# Unterstrasser 2017 uses exponential distribution:
# f = 1/mu exp(m/mu)
# mean radius
mu_R = 9.3 # mu
# mean mass
mu = compute_mass_from_radius(mu_R, c.mass_density_water_liquid_NTP)
print(mu)

masses = np.random.exponential(mu, N) # in mu

print(masses.shape)

fig, ax = plt.subplots()

m_ = np.linspace(0.0,2.0E7, 10000)

ax.hist(masses, density=True, bins=50)
ax.plot(m_, dst_expo(m_, 1.0/mu))


# the sedimentation velocity is a direct function of the droplet radius and
# thus the combined mass...