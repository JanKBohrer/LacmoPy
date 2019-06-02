#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 13:19:10 2019

@author: jdesk
"""
import numpy as np
import math
import matplotlib.pyplot as plt
# import sys
# from numba import njit

import constants as c
# from grid import Grid
# from grid import interpolate_velocity_from_cell_bilinear
from microphysics import compute_mass_from_radius,\
                         compute_initial_mass_fraction_solute_NaCl,\
                         compute_radius_from_mass,\
                         compute_density_particle,\
                         compute_dml_and_gamma_impl_Newton_full_np,\
                         compute_R_p_w_s_rho_p
from init import generate_SIP_ensemble_expo_SingleSIP_weak_threshold
from init import generate_SIP_ensemble_expo_my_xi_rnd

dV = 1.0
# dV = 1.0E-6
dt = 1.0
# dt = 1.0
store_every = 600
t_end = 3600.0

no_spc = 80

# droplet concentration
#n = 100 # cm^(-3)
n0 = 297.0 # cm^(-3)
# liquid water content (per volume)
LWC0 = 1.0E-6 # g/cm^3
# total number of droplets
no_rpc = int(n0 * dV * 1.0E6)
print("no_rpc=", no_rpc)

# we start with a monomodal exponential distribution
# mean droplet mass
mu = 1.0E15*LWC0 / n0
print("mu_m=", mu)
mu_R = compute_radius_from_mass(mu, c.mass_density_water_liquid_NTP)
print("mu_R=", mu_R)
total_mass_in_cell = dV*LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg

r_critmin = 0.6
kappa = 10
eta = 1.0E-9
# seed = 4713
no_sims = 50
seed_list = np.arange(4711, 4711+no_sims*2, 2)

xi_list = []
masses_list = []

for i,seed in enumerate(seed_list):
    masses, xis, m_low, m_high = generate_SIP_ensemble_expo_SingleSIP_weak_threshold(
                      1.0/mu, no_rpc, kappa=kappa, seed = seed)
    xi_list.append(xis)
    masses_list.append(masses)
    print(i, (xis.sum()-no_rpc)/no_rpc,
          (np.sum(xis*masses)-total_mass_in_cell)/total_mass_in_cell)
    # print()

# print(xis.shape)
# print(masses.shape)
# print((xis.sum()-no_rpc)/no_rpc)
# print((np.sum(masses*xis)-total_mass_in_cell)/total_mass_in_cell)


# no_sims = 50
no_bins = 40
method = "log_R"

xis = np.concatenate(xi_list)
masses = np.concatenate(masses_list)
radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)

R_min = np.amin(radii)
R_max = np.amax(radii)
ind_min = np.argmin(radii)
ind_max = np.argmax(radii)
# R_min = 0.99*np.amin(radii)
# R_max = 1.01*np.amax(radii)
# R_max = 3.0*np.amax(radii)
print("R_min=", R_min, "with xi =", xis[ind_min])
print("R_max=", R_max, "with xi =", xis[ind_max])

print("m_ges =", np.sum(masses*xis)/no_sims)
print("no_rpc =", np.sum(xis)/no_sims)

if method == "log_R":
    bins = np.logspace(np.log10(R_min), np.log10(R_max), no_bins)
elif method == "lin_R":
    bins = np.linspace(R_min, R_max, no_bins)
# print(bins)

xi_binned, _ = np.histogram(radii, bins, weights=xis)
xi_binned = xi_binned / no_sims

# masses in 10^-15 gram
mass_per_ln_R, _ = np.histogram(radii, bins, weights=masses*xis)
# convert to gram
mass_per_ln_R *= 1.0E-15/no_sims
# print(mass_per_ln_R)
# print(mass_per_ln_R.shape, bins.shape)

bins_log = np.log(bins)
# # bins_mid = np.exp((bins_log[1:] + bins_log[:-1]) * 0.5)
bins_mid = (bins[1:] + bins[:-1]) * 0.5

g_ln_R = mass_per_ln_R / (bins_log[1:] - bins_log[0:-1]) / dV

# Rs = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
plt.loglog(bins_mid, xi_binned, "x-")