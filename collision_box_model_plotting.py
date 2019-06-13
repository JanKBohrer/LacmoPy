#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:04:09 2019

@author: jdesk
"""

#import math
import numpy as np
import matplotlib.pyplot as plt

import constants as c
from microphysics import compute_radius_from_mass

from collision import generate_folder_collision
from analysis import auto_bin_SIPs

#%%

### SET PARAMETERS
myOS = "Linux"
# myOS = "MacOS"

dV = 1.0
# dt = 1.0
dt = 20.0
t_end = 3600.0
dt_store = 600.0

# kernel = "Bott"
# kernel = "Hall"
# kernel = "Long"
kernel = "Long_Bott"
# kernel = "Golovin"

# kernel = "hall"
# kernel = "long"
# kernel = "golovin"

init = "SingleSIP"
# init == "my_xi_random"

no_sims = 500

# for sampling and plotting:
# no_bins = 25
no_bins = 30
# bin_method = "lin_R"
bin_method = "log_R"
# bin_method = "my_auto_bin"

## for SingleSIP random:
# bin exponential scaling factor
# kappa = 80
kappa = 1000

## for my xi random initialization:
# INTENDED number of SIP:
no_spc = 40
# bin linear spreading parameter
eps = 200
# area of cumulative PDF that is covered, also determines the bin width
p_min = 0
p_max = 0.9999999

# droplet concentration
#n = 100 # cm^(-3)
n0 = 297.0 # cm^(-3)
# liquid water content (per volume)
LWC0 = 1.0E-6 # g/cm^3
# total number of droplets
no_rpc = int(n0 * dV * 1.0E6)
print("no_rpc=", no_rpc)

### DERIVED
# Unterstrasser 2017 uses monomodal exponential distribution:
# f = 1/mu exp(m/mu)
# mean droplet mass
mu = 1.0E15*LWC0 / n0
print("mu_m=", mu)
# mean radius
# mu_R = 9.3 # mu
mu_R = compute_radius_from_mass(mu, c.mass_density_water_liquid_NTP)
print("mu_R=", mu_R)
total_mass_in_cell = dV*LWC0*1.0E6*1.0E15 # in fg = 1.0E-18 kg

# numerical integration parameters for my xi random init
dm = mu*1.0E-5
m0 = 0.0
m1 = 100*mu

if init == "SingleSIP":
    init_pars = [kappa]
elif init == "my_xi_random":
    init_pars = [no_spc, eps]
simdata_path, path =\
    generate_folder_collision(myOS, dV, dt, kernel, init, init_pars, no_sims)

times = []
conc_vs_time = []
lambda2_vs_time = []
masses_vs_time = []
xis_vs_time = []

tot_pt = 0

### LOAD DATA FROM FILES
# no_sims=200
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
    conc_vs_time.append( np.load(conc_file))
    mass = np.load(mass_file)
    # print(mass.shape)
    xi = np.load(xi_file)
    lam = np.sum(xi * mass * mass * 1E-36, axis=1) / dV
    lambda2_vs_time.append(lam)
    tot_pt += mass.shape[1]
    masses_vs_time.append( mass )
    xis_vs_time.append( xi )

print(tot_pt)

times = np.array(times)[0]
conc_vs_time = np.array(conc_vs_time)
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
#%% PLOT MOMENTS OF THE DISTRIBUTION f(m) WITH TIME

TTFS, LFS, TKFS = 14,14,12

if kernel == "Hall":
    ylim0 = [1.0E7,1.0E9]
    ylim2 = [1.0E-15,5.0E-8]
elif kernel == "Long" or "Long_Bott":
    ylim0 = [1.0E6,1.0E9]
    ylim2 = [1.0E-15,1.0E-7]
elif kernel == "Bott":
    ylim0 = [1.0E6,1.0E9]
    ylim2 = [1.0E-15,1.0E-7]
elif kernel == "Golovin":
    ylim0 = [1.0E6,1.0E9]
    ylim2 = [1.0E-15,1.0E-9]

fig_name = path +\
f"moments_vs_time_kernel_{kernel}_init_{init}\
_dV_{dV}_dt_{dt}_no_sims_{no_sims}.png"

fig, axes = plt.subplots(nrows=2, figsize=(8,12))

###
ax = axes[0]
ax.semilogy( times//60, np.average(conc_vs_time,axis=0), "o" )
# ax.semilogy( times//60, conc[0], "o" )

ax.set_xlim([0.0,60.0])
ax.set_ylim(ylim0)

ax.tick_params(axis='both', which='major', labelsize=TKFS)
# ax.set_xlabel("time (min)", fontsize=LFS)
ax.set_ylabel(r"$\lambda_0\; (\mathrm{m^{-3}})$ ",fontsize=LFS)
ax.grid()

###
ax = axes[1]
ax.semilogy( times//60, np.average(lambda2_vs_time,axis=0), "o" )

ax.set_xlim([0.0,60.0])
ax.set_ylim(ylim2)

ax.tick_params(axis='both', which='major', labelsize=TKFS)
ax.set_xlabel("time (min)", fontsize=LFS)
ax.set_ylabel(r"$\lambda_2\; (\mathrm{kg^2 \, m^{-3}})$ ",fontsize=LFS)
ax.grid()

###
if init == "SingleSIP":
    pars = f"kappa={kappa}"
elif init == "my_xi_random":
    pars = f"no_spc={no_spc}, eps={eps}"
title = f"kernel={kernel}, init={init}, {pars},\n#sims={no_sims}, \
dV={dV}, dt={dt}, \
n0={n0:.3}" +  " cm$^{-3}$," + f" LWC0={LWC0:.2e}" + " g/cm$^3$"
fig.suptitle(title, fontsize=TTFS)

fig.savefig(fig_name)


#%% PLOT MASS DENSITY DISTRIBUTIONS VS RADIUS

# we need g_ln_R vs R, where g_ln_R = 3 * m^2 * f_m(m)
# where f_m(m) = 1/dm * 1/dV * sum_(m_a in [m,m+dm]) {xi_a}

# we have two possibilities:
# 1. sort particles in histogram of radius with log scaled bins,
# weighting each particle with xi_ * m_i, then divide each bin by
# log(R_right) - log(R_left)
# 2. 
# sort particle in histogram with m_i and weight with x_i * 3 * m_i^2
# then divide each bin by m_right-m_left and calculate m_bin_center -> R_bin_c

# dV = 0.1**3

TTFS, LFS, TKFS = 14,14,12

fig_name = path +\
f"g_ln_R_vs_time_kernel_{kernel}_init_{init}\
_dV_{dV}_dt_{dt}_no_sims_{no_sims}.png"
fig, ax = plt.subplots(figsize=(8,8))

i1 = 153217
i2 = 141680

# for ind_t in range(1):
# NOTE that if two droplets with the same xi collide,
# two droplets are created with [xi/2] and xi - [xi/2]
# and the same masses
for ind_t in range(len(times)):
    xis = xis_vs_time[ind_t]
    ind1 = np.nonzero(xis)
    xis = xis[ind1]
    masses = masses_vs_time[ind_t][ind1]
    # masses = masses_vs_time[ind_t]
    radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
    
    R_min = np.amin(radii)
    R_max = np.amax(radii)
    ind_min = np.argmin(radii)
    ind_max = np.argmax(radii)
    # R_min = 0.99*np.amin(radii)
    # R_max = 1.01*np.amax(radii)
    # R_max = 3.0*np.amax(radii)
    print("t=", times[ind_t], "R_min=", R_min, "with xi =", xis[ind_min],
          "R_max=", R_max, "with index", ind_max, "with xi =", xis[ind_max])
    print("m_ges =", np.sum(masses*xis), "no_rpc =", np.sum(xis))
    
    R_min *= 0.99
    R_max *= 1.01
    
    if bin_method == "my_auto_bin":
        g_ln_R, bins_mid, bins, xi_bin, mass_bin =\
            auto_bin_SIPs(masses, xis, no_bins, dV, no_sims)
        print("m_ges after binning=", np.sum(xi_bin*mass_bin))
        print("no_rpc =", np.sum(xi_bin)) 
    else:        
        if bin_method == "log_R":
            bins = np.logspace(np.log10(R_min), np.log10(R_max), no_bins)
        elif bin_method == "lin_R":
            bins = np.linspace(R_min, R_max, no_bins)
        # print(bins)
        
        # masses in 10^-15 gram
        mass_per_ln_R, _ = np.histogram(radii, bins, weights=masses*xis)
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
    ax.loglog( bins_mid, g_ln_R, "-", label=f"{int(times[ind_t]/60):>3}" )
    # ax.loglog( bins_mid, g_ln_R, "o", markersize=5.0 )
    
if kernel == "golovin":
    ax.set_xlim([1.0,2.0E3])
elif kernel == "long":
    ax.set_xlim([1.0,8.0E3])
elif kernel == "hall":
    ax.set_xlim([1.0,8.0E3])
# ax.hist(radii,weights=masses*xi,bins=30)
ax.set_ylim([1.0E-4,1.0E1])
# ax.set_ylim([1.0E-6,1.0E1])
# ax.set_xscale("log")
# ax.set_yscale("log")
ax.tick_params(axis='both', which='major', labelsize=TKFS)
ax.set_xlabel("radius ($\mathrm{\mu m}$)", fontsize=LFS)
ax.set_ylabel("mass distribution per ln(R) and volume (g/m$^3$)",fontsize=LFS)
if init == "SingleSIP":
    pars = f"kappa={kappa}"
elif init == "my_xi_random":
    pars = f"no_spc={no_spc}, eps={eps}"
title = f"kernel={kernel}, init={init}, {pars},\n#sims={no_sims}, \
dV={dV}, dt={dt}, \
n0={n0:.3}" +  " cm$^{-3}$," + f" LWC0={LWC0:.2e}" + " g/cm$^3$"
ax.set_title(title
, pad = 10, fontsize=TTFS)
ax.legend()
ax.grid()

fig.savefig(fig_name)

#%% PLOT SIP TOTAL MASSES
#fig, ax = plt.subplots(figsize=(8,8))
#for ind_t in range(len(times)):
#    masses = masses_vs_time[ind_t]
#    rad = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
#    xi = xis_vs_time[ind_t]
#    # ax.plot(masses, xi, "o", markersize=1.5)
#    # ax.plot(rad, xi, "o", markersize=1.5)
#    ax.plot(rad, xi*masses, "o", markersize=1.5)
#    ax.set_xscale("log")
#    ax.set_yscale("log")
# masses = masses_vs_time[0]
# xi = xis_vs_time[0]
# ax.plot(masses, xi, "o", markersize=1.0)
# ax.set_xscale("log")
# ax.set_yscale("log")

#%% TESTS
    
# dx_mean = 1.0
# eps = 10
# x_end = 100.0


# def lin_fct(x,a,b):
#     return a*x + b
# x_ = np.linspace(0.0,x_end,10000)
# plt.plot(x_, lin_fct(x_,a,b))

# print(lin_fct(x_end,a,b)/lin_fct(0.0,a,b))