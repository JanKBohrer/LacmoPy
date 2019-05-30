#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 15:50:53 2019

@author: jdesk
"""

import math
import numpy as np
import matplotlib.pyplot as plt

from collision import dst_expo
from init import compute_quantiles


#%% GENERATE SIP ENSEMBLE



#%% EXECUTE TEST

dV = 1.0 # m^3
dt = 1.0
no_spc = 40

# dt = 1.0
store_every = 600
t_end = 3600.0

# droplet concentration
#n = 100 # cm^(-3)
n0 = 297.0 # cm^(-3)
# liquid water content (per volume)
LWC0 = 1.0E-6 # g/cm^3
# total number of droplets
no_rpc = int(n0 * dV * 1.0E6)
print("no_rpc=", no_rpc)
print("no_spc=", no_spc)
total_mass_in_cell = dV*LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg
# we start with a monomodal exponential distribution
# mean droplet mass
mu = 1.0E15*LWC0 / n0
print("mu_m=", mu)
from microphysics import compute_radius_from_mass
import constants as c
mu_R = compute_radius_from_mass(mu, c.mass_density_water_liquid_NTP)
print("mu_R=", mu_R)

dst = dst_expo
par = 1/mu

p_min = 0
p_max = 0.99999
dm = mu*1.0E-5
m0 = 0.0
m1 = 100*mu
seed = 4711

masses, xis, m_low, m_high = generate_SIP_ensemble(dst, par, no_spc, no_rpc,
                                    total_mass_in_cell,
                                    p_min, p_max,
                                    m0, m1, dm, seed, setseed = True)

simdata_path = "/home/jdesk/OneDrive/python/sim_data/"
folder = "collision_box_model_multi_sim/generate_SIPs/"
path = simdata_path + folder

import os
if not os.path.exists(path):
    os.makedirs(path)

np.save(path + f"masses_dV_{dV}_no_spc_aim_{no_spc}.npy", masses)
np.save(path + f"xis_dV_{dV}_no_spc_aim_{no_spc}.npy", xis)

#%%

masses = np.load(path + f"masses_dV_{dV}_no_spc_aim_{no_spc}.npy")
xis = np.load(path + f"xis_dV_{dV}_no_spc_aim_{no_spc}.npy")

fig_name = f"SIP_ensemble_dV_{dV}_no_spc_aim_{no_spc}.png"
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(masses, xis, "o")
m_max = np.amax(masses)
m_ = np.linspace(0.0,m_max, 10000)
# m_max = masses[-2]
m_min = np.amin(masses)
# no_spc = len(masses)
no_rpc = xis.sum()
# m_ges = np.sum(masses*xis)
# bin_size = m_max/no_spc
# bin_size = m_max/(no_spc-1)
bin_size = (m_high - m_low)/(no_spc)
# ax.plot(m_, no_rpc*np.exp(-m_/mu)\
#             *(np.exp(0.5*bin_size/mu)-np.exp(-0.5*bin_size/mu)))
ax.plot(m_, no_rpc*bin_size*dst_expo(m_,1.0/mu))
ax.set_yscale("log")

#%%

# masses1 = np.load(path + "masses.npy")
# xis1 = np.load(path + "xis.npy")
# masses2 = np.load(path + "masses2.npy")
# xis2 = np.load(path + "xis2.npy")
# masses3 = np.load(path + "masses3.npy")
# xis3 = np.load(path + "xis3.npy")

#%% compare with PDF

import matplotlib.pyplot as plt

masses = np.load(path + "masses0.npy")
xis = np.load(path + "xis0.npy")

m_max = np.amax(masses)
# xi = np.ones(no_spct, dtype = np.int64)
# print("sim_n", sim_n, "; masses.shape=", masses.shape, "; m_max=", m_max)

no_rpc_should = int(n0 * dV * 1.0E6)
print(xis.sum()-no_rpc_should)

total_mass_in_cell = dV * LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg
print(np.sum(masses*xis)-total_mass_in_cell)


#%%
from init import compute_quantiles

dV = 1.0E-6 # m^3
dt = 1.0
no_spc = 20

# dt = 1.0
store_every = 600
t_end = 3600.0

# droplet concentration
#n = 100 # cm^(-3)
n0 = 297.0 # cm^(-3)
# liquid water content (per volume)
LWC0 = 1.0E-6 # g/cm^3
# total number of droplets
no_rpc = int(n0 * dV * 1.0E6)
print("no_rpc=", no_rpc)
print("no_spc=", no_spc)
total_mass_in_cell = dV * LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg
# we start with a monomodal exponential distribution
# mean droplet mass
mu = 1.0E15*LWC0 / n0
print("mu_m=", mu)
from microphysics import compute_radius_from_mass
import constants as c
mu_R = compute_radius_from_mass(mu, c.mass_density_water_liquid_NTP)
print("mu_R=", mu_R)

dst = dst_expo
par = 1/mu
p_min = 0
p_max = 0.9999
dm = mu*1.0E-5
m0 = 0.0
m1 = 100*mu
seed = 4711

(m_low, m_high), Ps = compute_quantiles(dst_expo, par, m0, m1, dm,
                                 [p_min, p_max], None)

print(m_low, f"{m_high:.2e}")

#%%

fig_name = f"SIP_ensemble.png"
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(masses, xis, "o")
m_max = np.amax(masses)
m_ = np.linspace(0.0,m_max, 10000)
# m_max = masses[-2]
m_min = np.amin(masses)
no_spc = len(masses)
no_rpc = xis.sum()
m_ges = np.sum(masses*xis)
# bin_size = m_max/no_spc
# bin_size = m_max/(no_spc-1)
bin_size = (m_high - m_low)/(20)
# ax.plot(m_, no_rpc*np.exp(-m_/mu)\
#             *(np.exp(0.5*bin_size/mu)-np.exp(-0.5*bin_size/mu)))
ax.plot(m_, no_rpc*bin_size*dst_expo(m_,1.0/mu))

print(bin_size)
# no_bins = 50
# ax.hist(masses, density=True, bins=50)
# bins = ax.hist(masses, weights=xis, bins=8)[1]
# no_bins = len(bins - 1)
# ax.plot(m_, (bins[-1]-bins[0])/no_bins*masses.shape[0]*dst_expo(m_, 1.0/mu))
# ax.plot(m_, (bins[-1]-bins[0])/no_bins*no_rpc*dst_expo(m_, 1.0/mu))

