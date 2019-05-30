#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:04:09 2019

@author: jdesk
"""

import numpy as np
import matplotlib.pyplot as plt

import constants as c
from microphysics import compute_radius_from_mass


simdata_path = "/home/jdesk/OneDrive/python/sim_data/"
# folder = "collision_box_model_multi_sim/"
# folder = "collision_box_model_multi_sim/dV_4E-3_no_sim_50/"
# folder = "collision_box_model_multi_sim/dV_1E0_no_spc_40_no_sim_2/"
# folder = "collision_box_model_multi_sim/dV_1E0_no_spc_80_no_sim_50/"
folder = "collision_box_model_multi_sim/dV_1E0_no_spc_80_no_sim_50_02/"
# folder = "collision_box_model/"
path = simdata_path + folder

# path = "/mnt/D/sim_data/dV_4E-3_no_sim_50/"

t_end = 3600.0
dt = 1.0
# dV = 1.0E-3
# dV = 4.0E-3
dV = 1.0
# no_sims = 50
no_sims = 50

# droplet concentration
#n = 100 # cm^(-3)
n0 = 297.0 # cm^(-3)
# liquid water content (per volume)
LWC0 = 1.0E-6 # g/cm^3
# print(np.arange(0.0,t_end,dt))
# print(len(np.arange(0.0,t_end,dt)))

# total number of droplets
no_spct = int(n0 * dV * 1.0E6)

times = []
conc = []
masses_vs_time = []
xis_vs_time = []

tot_pt = 0
for sim_n in range(no_sims):
    # times_file = path + "times.npy"
    # conc_file = path + "conc.npy"
    # mass_file = path + "masses_vs_time.npy"
    # xi_file = path + "xis_vs_time.npy"
    times_file = path + f"times_{sim_n}.npy"
    conc_file = path + f"conc_{sim_n}.npy"
    mass_file = path + f"masses_vs_time_{sim_n}.npy"
    xi_file = path + f"xis_vs_time_{sim_n}.npy"
    
    times.append( np.load(times_file))
    conc.append( np.load(conc_file))
    mass = np.load(mass_file)
    print(mass.shape)
    tot_pt += mass.shape[1]
    masses_vs_time.append( mass )
    xis_vs_time.append( np.load(xi_file))

print(tot_pt)

times = np.array(times)[0]
conc = np.array(conc)
# masses_vs_time = np.array(masses_vs_time)
# xis_vs_time = np.array(xis_vs_time)

masses_vs_time = np.concatenate(masses_vs_time, axis=1)
xis_vs_time = np.concatenate(xis_vs_time, axis=1)

print(
    f"masses shape: {masses_vs_time.shape[0]} {masses_vs_time.shape[1]:.3e}")
print(f"xis shape: {xis_vs_time.shape[0]} {xis_vs_time.shape[1]:.3e}")

print(
np.amin(masses_vs_time.flatten())
)
#%% PLOT PARTICLE CONCENTRATION WITH TIME

fig, ax = plt.subplots()
ax.semilogy( times//60, conc[0], "o" )
ax.set_xlim([0.0,60.0])
ax.set_ylim([1.0E6,1.0E9])
ax.grid()

#%% PLOT MASS DENSITY DISTRIBUTIONS VS RADIUS

# we need g_ln_R vs R, where g_ln_R = 3 * m^2 * f_m(m)
# where f_m(m) = 1/dm * 1/dV * sum_(m_a in [m,m+dm]) {xi_a}
# i.e: for all m_i,x_i:
# sort particle in histogram with R_i(m_i) and weight with x_i * 3 * m_i^2

# dV = 0.1**3

fig_name = path + f"mass_distr_per_ln_R_vs_time_no_sims_{no_sims}_dV_{dV}.png"
fig, ax = plt.subplots(figsize=(8,8))

# method = "lin_R"
method = "log_R"
for ind_t in range(len(times)):
    masses = masses_vs_time[ind_t]
    xi = xis_vs_time[ind_t]
    radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
    
    R_min = 0.99*np.amin(radii)
    R_max = 1.01*np.amax(radii)
    # R_max = 3.0*np.amax(radii)
    print("R_min=", R_min)
    print("R_max=", R_max)
    
    no_bins = 15
    if method == "log_R":
        bins = np.logspace(np.log10(R_min), np.log10(R_max), no_bins)
    elif method == "lin_R":
        bins = np.linspace(R_min, R_max, no_bins)
    # print(bins)
    
    # masses in 10^-15 gram
    mass_per_ln_R, _ = np.histogram(radii, bins, weights=masses*xi)
    # convert to gram
    mass_per_ln_R *= 1.0E-15/no_sims
    # print(mass_per_ln_R)
    # print(mass_per_ln_R.shape, bins.shape)
    
    bins_log = np.log(bins)
    # bins_mid = np.exp((bins_log[1:] + bins_log[:-1]) * 0.5)
    bins_mid = (bins[1:] + bins[:-1]) * 0.5
    
    g_ln_R = mass_per_ln_R / (bins_log[1:] - bins_log[0:-1]) / dV
    
    # print(g_ln_R.shape)
    # print(np.log(bins_mid[1:])-np.log(bins_mid[0:-1]))
    
    ax.loglog( bins_mid, g_ln_R, "-" )
    

# ax.loglog( bins_mid, np.ones_like(bins_mid), "o" )
ax.set_xlim([1.0,1.0E3])
# ax.hist(radii,weights=masses*xi,bins=30)
ax.set_ylim([1.0E-4,1.0E1])
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.set_xlabel("radius (mu)")
ax.set_ylabel("mass distribution per ln(R) and volume (g/m^3)")
ax.set_title(
f"#sims={no_sims}, n0={n0:.3} cm^-3, LWC0={LWC0} g/cm^3, dV={dV} \
#SIP={no_spct:.2e}", pad = 10)
ax.grid()

fig.savefig(fig_name)

#%%
fig, ax = plt.subplots(figsize=(8,8))
for ind_t in range(len(times)):
    masses = masses_vs_time[ind_t]
    xi = xis_vs_time[ind_t]
    ax.plot(masses, xi, "o")
    ax.set_xscale("log")
    ax.set_yscale("log")
# masses = masses_vs_time[0]
# xi = xis_vs_time[0]
# ax.plot(masses, xi, "o", markersize=1.0)
# ax.set_xscale("log")
# ax.set_yscale("log")
