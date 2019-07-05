#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:33:04 2019

@author: jdesk
"""

import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

import constants as c
# from microphysics import compute_mass_from_radius
from microphysics import compute_radius_from_mass
from microphysics import compute_mass_from_radius

#%% CREATE SIP ENSEMBLE

# f_m(m) = number concentration density per mass
# such that int f_m(m) dm = DNC = droplet number concentration (1/m^3)
# f_m(m) = 1/LWC * exp(-m/m_avg)
# LWC = liquid water content (kg/m^3)
# m_avg = M/N = LWC/DNC
# where M = total droplet mass in dV, N = tot. # of droplets in dV
# in this function f_m(m) = conc_per_mass(m, LWC_inv, DNC_over_LWC)
# DNC_over_LWC = 1/m_avg
# m in kg
# function moments checked versus analytical values via numerical integration

def conc_per_mass_np(m, DNC, DNC_over_LWC): # = f_m(m)
    return DNC * DNC_over_LWC * np.exp(-DNC_over_LWC * m)

conc_per_mass = njit()(conc_per_mass_np)

def moments_analytical(n, DNC, LWC_over_DNC):
    if n == 0:
        return DNC
    else:
        return math.factorial(n) * DNC * LWC_over_DNC**n

# nth moment of f_m(m) -> mom_n = int(dm * m^k * f_m(m))
# function checked versus analytical values via numerical integration
def moments_f_m_num_np(n, DNC, DNC_over_LWC, steps=1E6):
    m_avg = 1.0/DNC_over_LWC
    # m_high = m_avg * steps**0.7
    m_high = m_avg * 1.0E4
    dm = m_high / steps
    m = 0.0
    intl = 0.0
    # cnt = 0
    if n == 0:
        f1 = conc_per_mass(m, DNC, DNC_over_LWC)
        while (m < m_high):
            f2 = conc_per_mass(m + 0.5*dm, DNC, DNC_over_LWC)
            f3 = conc_per_mass(m + dm, DNC, DNC_over_LWC)
            # intl_bef = intl        
            intl += 0.1666666666667 * dm * (f1 + 4.0 * f2 + f3)
            m += dm
            f1 = f3
            # cnt += 1        
            # intl += dx * x * dst_expo(x,k)
            # x += dx
            # cnt += 1
    else:
        f1 = conc_per_mass(m, DNC, DNC_over_LWC) * m**n
        while (m < m_high):
            f2 = conc_per_mass(m + 0.5*dm, DNC, DNC_over_LWC) * (m + 0.5*dm)**n
            f3 = conc_per_mass(m + dm, DNC, DNC_over_LWC) * (m + dm)**n
            # intl_bef = intl        
            intl += 0.1666666666667 * dm * (f1 + 4.0 * f2 + f3)
            m += dm
            f1 = f3
            # cnt += 1        
            # intl += dx * x * dst_expo(x,k)
            # x += dx
            # cnt += 1
    return intl
moments_f_m_num = njit()(moments_f_m_num_np)

# print( math.factorial(2-1) * LWC0**2 / DNC0 * 2 )

# SingleSIP probabilistic

# r_critmin -> m_low = m_0
# m_{l+1} = m_l * 10^(1/kappa)
# dm_l = m_{l+1} - m_l
# mu_l = m_l + rnd() * dm_l
# xi_l = f_m(mu_l) * dm_l * dV
def generate_SIP_ensemble_SingleSIP_Unt_np(DNC0, LWC0,
                                           dV, kappa, eta, r_critmin,
                                           m_high_over_m_low=1.0E6,
                                           seed=3711, setseed=True):
    if setseed: np.random.seed(seed)
    m_low = 1.0E-18 * compute_mass_from_radius(r_critmin,
                                               c.mass_density_water_liquid_NTP)
    DNC0_over_LWC0 = DNC0 / LWC0
    bin_factor = 10**(1.0/kappa)
    m_high = m_low * m_high_over_m_low
    m_left = m_low

    l_max = int(kappa * np.log10(m_high_over_m_low))
    rnd = np.random.rand( l_max )
    rnd2 = np.random.rand( l_max )

    xis = np.zeros(l_max, dtype = np.float64)
    masses = np.zeros(l_max, dtype = np.float64)
    bins = np.zeros(l_max+1, dtype = np.float64)
    bins[0] = m_left

    bin_n = 0
    while m_left < m_high:
        m_right = m_left * bin_factor
        bin_width = m_right - m_left
        mu = m_left + rnd[bin_n] * bin_width
        xi = conc_per_mass(mu, DNC0, DNC0_over_LWC0) * bin_width * dV
        # print(bin_n, xi)
        xis[bin_n] = xi
        masses[bin_n] = mu
        m_left = m_right

        bin_n += 1
        bins[bin_n] = m_left

    xi_max = xis.max()
    xi_critmin = xi_max * eta

    valid_ids = np.ones(l_max, dtype = np.int64)
    for bin_n in range(l_max):
        if xis[bin_n] < xi_critmin:
            if rnd2[bin_n] < xis[bin_n] / xi_critmin:
                xis[bin_n] = xi_critmin
            else: valid_ids[bin_n] = 0
        # else: valid_ids[bin_n] = 1
    xis = xis[np.nonzero(valid_ids)[0]]
    masses = masses[np.nonzero(valid_ids)[0]]

    return masses, xis, m_low, bins
generate_SIP_ensemble_SingleSIP_Unt =\
    njit()(generate_SIP_ensemble_SingleSIP_Unt_np)

#%% TESTING SIP ENSEMBLE
r_critmin = 0.6 # mu
m_low = 1.0E-18 * compute_mass_from_radius(r_critmin,
                                           c.mass_density_water_liquid_NTP)

dV = 1.0
kappa = 10
eta = 1.0E-9

DNC0 = 2.97E8 # 1/m^3
#DNC0 = 3.0E8 # 1/m^3
LWC0 = 1.0E-3 # kg/m^3

# LWC0_inv = 1.0 / LWC0
DNC0_over_LWC0 = DNC0 / LWC0

# print(r_critmin, m_low)
# print(conc_per_mass(0.0, DNC0, DNC0_over_LWC0))
# print(conc_per_mass(m_low, DNC0, DNC0_over_LWC0))

#no_bins = 10

no_sims = 500
start_seed = 3711

seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)

#path = "/home/jdesk/OneDrive/python/sim_data/test_SIP_ensemble_Unt/"
path = "/Users/bohrer/OneDrive - bwedu/python/sim_data/test_SIP_ensemble_Unt/"
folder = f"kappa_{kappa}/"
path = path + folder

ensemble_parameters = [dV, DNC0, LWC0, r_critmin, kappa, eta, no_sims, start_seed]

if not os.path.exists(path):
    os.makedirs(path)

for i,seed in enumerate(seed_list):
    masses, xis, m_low, bins =\
        generate_SIP_ensemble_SingleSIP_Unt(DNC0, LWC0,
                                               dV, kappa, eta, r_critmin,
                                               m_high_over_m_low=1.0E6, seed=seed)
    bins_rad = compute_radius_from_mass(1.0E18*bins,
                                        c.mass_density_water_liquid_NTP)
    radii = compute_radius_from_mass(1.0E18*masses,
                                     c.mass_density_water_liquid_NTP)
    np.save(path + f"masses_seed_{seed}", masses)
    np.save(path + f"radii_seed_{seed}", radii)
    np.save(path + f"xis_seed_{seed}", xis)
    
    if i == 0:
        np.save(path + f"bins_mass", bins)
        np.save(path + f"bins_rad", bins_rad)
        np.save(path + "ensemble_parameters", ensemble_parameters)
        

#%%

dV, DNC0, LWC0, r_critmin, kappa, eta, no_sims, start_seed = \
    tuple(np.load(path + "ensemble_parameters.npy"))
LWC0_over_DNC0 = LWC0 / DNC0

bins_mass = np.load(path + "bins_mass.npy")
bins_rad = np.load(path + "bins_rad.npy")

masses_sampled = []
radii_sampled = []
xis_sampled = []
moments_sampled = []
for seed in seed_list:
    masses=np.load(path + f"masses_seed_{seed}.npy")
    xis=np.load(path + f"xis_seed_{seed}.npy")
    masses_sampled.append(masses)
    radii_sampled.append(np.load(path + f"radii_seed_{seed}.npy"))
    xis_sampled.append(xis)
    moments = np.zeros(4,dtype=np.float64)
    moments[0] = xis.sum() / dV
    for n in range(1,4):
        moments[n] = np.sum(xis*masses**n) / dV
    moments_sampled.append(moments)

masses_sampled = np.concatenate(masses_sampled)
radii_sampled = np.concatenate(radii_sampled)
xis_sampled = np.concatenate(xis_sampled)
moments_sampled = np.transpose(moments_sampled)


H1 = np.histogram(radii_sampled, bins_rad)[0]
H2 = np.histogram(radii_sampled, bins_rad, weights=xis_sampled)[0]

H1 = H1[np.nonzero(H1)[0]]
H2 = H2[np.nonzero(H1)[0]]

H = H2 / H1


#%%
LWC0_over_DNC0 = LWC0 / DNC0
moments_an = np.zeros(4,dtype=np.float64)
for n in range(4):
    moments_an[n] = moments_analytical(n, DNC0, LWC0_over_DNC0)

for n in range(4):
    print(n, (np.average(moments_sampled[n])-moments_an[n])/moments_an[n] )

moments_sampled_avg_norm = np.average(moments_sampled, axis=1) / moments_an
moments_sampled_std_norm = np.std(moments_sampled, axis=1) / moments_an

#%% PLOTTING

#bins_mid_rad = 0.5*(bins_rad[1:] + bins_rad[:-1])



# approximate the functions f_m, f_lnR = 3*m*f_m, g_lnR=3*m^2*f_m
m_min = masses_sampled.min()
m_max = masses_sampled.max()

# m_min*=0.99
# m_max*=1.01

#m_min = m_min*0.5
#m_max = m_max*2.0

#no_bins = 20

R_min = radii_sampled.min()
R_max = radii_sampled.max()

# sample bins in logspace of mass m
#bins = np.logspace(np.log10(m_min), np.log10(m_max), no_bins)
#bins_mid = 0.5*(bins[1:] + bins[:-1])

# define centers on lin scale
bins_mass_center = 0.5 * (bins_mass[:-1] + bins_mass[1:])
bins_rad_center = 0.5 * (bins_rad[:-1] + bins_rad[1:])

# define centers on the logarithmic scale
#bins_mass_center = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
#bins_rad_center = bins_rad[:-1] * 10**(1.0/(2.0*kappa))

### new approach: calculate the center of mass for each bin and set it as the
# bin center -> s.b.


bins_mass_width = (bins_mass[1:]-bins_mass[:-1])

# estimate f_m(m) by binning:
# DNC_i = f_m(m_i) * dm_i = droplet number conc in bin i with size dm_i
f_m_num = np.histogram(masses_sampled,bins_mass,weights=xis_sampled)[0]
g_m_num = np.histogram(masses_sampled,bins_mass,weights=xis_sampled*masses_sampled)[0]

f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
f_m_ind = np.nonzero(f_m_counts)[0]

f_m_num = f_m_num[f_m_ind]
g_m_num = g_m_num[f_m_ind]
bins_mass_center = bins_mass_center[f_m_ind]
bins_mass_width = bins_mass_width[f_m_ind]

### new approach: calculate the center of mass for each bin and set it as the
# bin center
bins_mass_center = g_m_num/f_m_num

f_m_num = f_m_num / bins_mass_width / dV / no_sims
g_m_num = g_m_num / bins_mass_width / dV / no_sims

bins_rad_center = compute_radius_from_mass(bins_mass_center*1.0E18, 
                                           c.mass_density_water_liquid_NTP)

g_ln_r_num = 3 * bins_mass_center * g_m_num * 1000.0

#m_min = masses_sampled.min()*0.999
#m_max = masses_sampled.max()*1.001
#m_ = np.logspace(np.log10(m_min*0.98), np.log10(m_max*1.1), 1000)
m_ = np.logspace(np.log10(bins_mass_center[0]), np.log10(bins_mass_center[-1]), 1000)
R_ = compute_radius_from_mass(m_*1.0E18, c.mass_density_water_liquid_NTP)
f_m_ana = conc_per_mass_np(m_, DNC0, DNC0_over_LWC0)
g_m_ana = m_ * f_m_ana
g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0

# analog to Wang 2007 and Unterstr 2017
moments_num = np.zeros(4, dtype = np.float64)
for n in range(4):
    if n == 0:
        moments_num[n] = np.sum( g_m_num / bins_mass_center * bins_mass_width )
    elif n == 1:
        moments_num[n] = np.sum( g_m_num * bins_mass_width)
    else:
        moments_num[n] = np.sum( g_m_num * bins_mass_center**(n-1) * bins_mass_width )
#for n in range(4):
#    if n == 0:
#        moments_num[n] = np.log(10)/(3.0*kappa) * np.sum( g_m_num / bins_mass_center )
#    elif n == 1:
#        moments_num[n] = np.log(10)/(3.0*kappa) * np.sum( g_m_num )
#    else:
#        moments_num[n] = np.log(10)/(3.0*kappa) * np.sum( g_m_num * bins_mass_center**n )

###########################################

no_rows = 4
fig, axes = plt.subplots(nrows=no_rows, figsize=(6,4*no_rows))
# ax.loglog(radii, xis, "x")
# ax.loglog(bins_mid[:51], H, "x-")
# ax.vlines(bins_rad, xis.min(), xis.max(), linewidth=0.5, linestyle="dashed")
ax = axes[0]
ax.plot(bins_mass_center, f_m_num, "x")
ax.plot(m_, f_m_ana)
ax.set_xscale("log")
ax.set_yscale("log")
ax = axes[1]
ax.plot(bins_mass_center, g_m_num, "x")
ax.plot(m_, g_m_ana)
ax.set_xscale("log")
ax.set_yscale("log")
# ax.loglog(bins_mid, f_m_num/50, "x")
# ax.loglog(m_, f_m_ana)
# ax.set_xlim(m_min*9, m_max/9)
ax = axes[2]
ax.plot(bins_rad_center, g_ln_r_num, "x")
ax.plot(R_, g_ln_r_ana)
ax.set_xscale("log")
ax.set_yscale("log")
#ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),17))
ax.yaxis.set_ticks(np.logspace(-10,0,11))
ax.grid(which="both")


ax = axes[3]
for n in range(4):
    ax.plot( n*np.ones_like(moments_sampled[n]) , moments_sampled[n]/moments_an[n], "o")
ax.errorbar( np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
            fmt = "x" , c = "k", markersize = 20.0, linewidth =5.0, zorder=99)
ax.plot(np.arange(4), np.ones_like(np.arange(4)))

for ax in axes[:2]:
    ax.grid()

#%% PLOTTING DEVIATIONS

#m_min = masses_sampled.min()*0.999
#m_max = masses_sampled.max()*1.001
#m_ = np.logspace(np.log10(m_min*0.98), np.log10(m_max*1.1), 1000)
m_ = bins_mass_center 
#m_ = np.logspace(np.log10(bins_mass_center[0]), np.log10(bins_mass_center[-1]), 1000)
R_ = compute_radius_from_mass(m_*1.0E18, c.mass_density_water_liquid_NTP)
f_m_ana = conc_per_mass_np(m_, DNC0, DNC0_over_LWC0)
g_m_ana = m_ * f_m_ana
g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0

# analog to Wang 2007 and Unterstr 2017
moments_num = np.zeros(4, dtype = np.float64)
for n in range(4):
    if n == 0:
        moments_num[n] = np.sum( g_m_num / bins_mass_center * bins_mass_width )
    elif n == 1:
        moments_num[n] = np.sum( g_m_num * bins_mass_width)
    else:
        moments_num[n] = np.sum( g_m_num * bins_mass_center**(n-1) * bins_mass_width )
#for n in range(4):
#    if n == 0:
#        moments_num[n] = np.log(10)/(3.0*kappa) * np.sum( g_m_num / bins_mass_center )
#    elif n == 1:
#        moments_num[n] = np.log(10)/(3.0*kappa) * np.sum( g_m_num )
#    else:
#        moments_num[n] = np.log(10)/(3.0*kappa) * np.sum( g_m_num * bins_mass_center**n )

###########################################

no_rows = 4
fig, axes = plt.subplots(nrows=no_rows, figsize=(6,4*no_rows))
# ax.loglog(radii, xis, "x")
# ax.loglog(bins_mid[:51], H, "x-")
# ax.vlines(bins_rad, xis.min(), xis.max(), linewidth=0.5, linestyle="dashed")
ax = axes[0]
ax.plot(bins_mass_center, (f_m_num-f_m_ana)/f_m_ana, "x")
#ax.plot(m_, f_m_ana)
#ax.set_xscale("log")
#ax.set_yscale("log")
ax = axes[1]
ax.plot(bins_mass_center, (g_m_num-g_m_ana)/g_m_ana, "x")
#ax.plot(m_, g_m_ana)
#ax.set_xscale("log")
#ax.set_yscale("log")
# ax.loglog(bins_mid, f_m_num/50, "x")
# ax.loglog(m_, f_m_ana)
# ax.set_xlim(m_min*9, m_max/9)
ax = axes[2]
ax.plot(bins_rad_center, (g_ln_r_num-g_ln_r_ana)/g_ln_r_ana, "x")
#ax.plot(R_, g_ln_r_ana)
#ax.set_xscale("log")
#ax.set_yscale("log")
#ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),17))
#ax.yaxis.set_ticks(np.logspace(-10,0,11))
#ax.grid(which="both")


ax = axes[3]
for n in range(4):
    ax.plot( n*np.ones_like(moments_sampled[n]) , moments_sampled[n]/moments_an[n], "o")
ax.errorbar( np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
            fmt = "x" , c = "k", markersize = 20.0, linewidth =5.0, zorder=99)
ax.plot(np.arange(4), np.ones_like(np.arange(4)))

for ax in axes[:3]:
    ax.grid()

#################################

#fig, ax = plt.subplots(nrows=1, figsize=(8,8))
#ax.plot(bins_rad_center, g_ln_r_num, "x")
#ax.plot(R_, g_ln_r_ana)
#ax.set_xscale("log")
#ax.set_yscale("log")
#ax.grid()
#ax.yaxis.set_ticks(np.logspace(-8,0,9))

#################################

#%%

#fig, ax = plt.subplots(nrows=1, figsize=(8,8))
#for n in range(4):
#    ax.plot( n*np.ones_like(moments_sampled[n]) , moments_sampled[n]/moments_an[n], "o")
#ax.errorbar( np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
#            fmt = "x" , c = "k", markersize = 20.0, linewidth =5.0, zorder=99)
#ax.plot(np.arange(4), np.ones_like(np.arange(4)))
#ax.plot( np.ones_like(moments_sampled[0]) , moments_sampled[0]/moments_an[0], "o")


#%% TESTING MOMENTS OF THE CONCENTRATION DENSITY FUNCTION f_m(m)
# for n in range(4):
#     mom = moments_f_m(n,DNC0, DNC0_over_LWC0, steps = 1.0E7)
#     mom_an = moments_analytical(n, DNC0, 1.0 / DNC0_over_LWC0)
#     print(n, mom, mom_an, (mom - mom_an) / mom_an)
