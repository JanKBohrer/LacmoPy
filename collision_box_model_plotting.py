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
# folder = "collision_box_model_multi_sim/dV_1E0_no_spc_80_no_sim_50_02/"
# folder = "collision_box_model_multi_sim/dV_1E-3_no_spc_40_no_sim_2/"
# folder = "collision_box_model_multi_sim/dV_1E-3_no_spc_40_no_sim_5/"
# folder = "collision_box_model_multi_sim/dV_1E-1_no_spc_40_no_sim_50/"
# folder = "collision_box_model_multi_sim/dV_1E0_no_spc_40_no_sim_50/"
# folder = "collision_box_model_multi_sim/dV_1E0_no_spc_40_no_sim_50_eps_500/"
# folder = "collision_box_model_multi_sim/dV_1E0_no_spc_40_no_sim_50_eps_500_pmax_1E-7/"
# folder = "collision_box_model_multi_sim/dV_1E0_no_spc_80_no_sim_50_eps_500_pmax_1E-7/"
# folder = "collision_box_model_multi_sim/dV_1E0_no_spc_40_no_sim_50_eps_100_pmax_1E-7/"
folder = "collision_box_model_multi_sim/dV_1E0_no_spc_80_no_sim_50_eps_200_pmax_1E-7/"
# folder = "collision_box_model/"
path = simdata_path + folder

# path = "/mnt/D/sim_data/dV_4E-3_no_sim_50/"

t_end = 3600.0
dt = 1.0
# dV = 1.0E-1
# dV = 4.0E-3
dV = 1.0
no_sims = 50
# no_sims = 30

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

#%% BINNING OF SIPs:

def auto_bin_SIPs(masses, xis, xi_min, no_bins, dV, no_sims):

    ind = np.nonzero(xis)
    m_sort = masses[ind]
    xi_sort = xis[ind]
    # print(m_sort.shape)
    # print(xi_sort.shape)
    # print()
    
    ind = np.argsort(m_sort)
    m_sort = m_sort[ind]
    xi_sort = xi_sort[ind]
    # print(m_sort.shape)
    # print(xi_sort.shape)
    
    # plt.plot(masses, "o")
    # plt.plot(m_sort, "o")
    # print(np.nonzero(xis))
    
    # plt.loglog(m_sort, xi_sort, "+")
    # plt.plot(m_sort, xi_sort, "+")
    
    ### merge particles with xi < xi_min
    for i in range(len(xi_sort)-1):
        if xi_sort[i] < xi_min:
            xi = xi_sort[i]
            m = m_sort[i]
            xi_left = 0
            j = i
            while(j > 0 and xi_left==0):
                j -= 1
                xi_left = xi_sort[j]
            if xi_left != 0:
                m1 = m_sort[j]
                dm_left = m-m1
            else:
                dm_left = 1.0E18
            m2 = m_sort[i+1]
            if m2-m < dm_left:
                j = i+1
                # assign to m1 since distance is smaller
                # i.e. increase number of xi[i-1],
                # then reweight mass to total mass
            m_sum = m*xi + m_sort[j]*xi_sort[j]
            # print("added pt ", i, "to", j)
            # print("xi_i bef=", xi)
            # print("xi_j bef=", xi_sort[j])
            # print("m_i bef=", m)
            # print("m_j bef=", m_sort[j])
            # xi_sort[j] += xi_sort[i]
            # m_sort[j] = m_sum / xi_sort[j]
            # print("xi_j aft=", xi_sort[j])
            # print("m_j aft=", m_sort[j])
            xi_sort[i] = 0           
    
    if xi_sort[-1] < xi_min:
        i = -1
        xi = xi_sort[i]
        m = m_sort[-1]
        xi_left = 0
        j = i
        while(xi_left==0):
            j -= 1
            xi_left = xi_sort[j]
        
        m_sum = m*xi + m_sort[j]*xi_sort[j]
        xi_sort[j] += xi_sort[i]
        m_sort[j] = m_sum / xi_sort[j]
        xi_sort[i] = 0           
    
    ind = np.nonzero(xi_sort)
    xi_sort = xi_sort[ind]
    m_sort = m_sort[ind]
    
    # print()
    # print("xis.min()")
    # print("xi_sort.min()")
    # print(xis.min())
    # print(xi_sort.min())
    # print()
    
    # print("xi_sort.shape, xi_sort.dtype")
    # print("m_sort.shape, m_sort.dtype")
    # print(xi_sort.shape, xi_sort.dtype)
    # print(m_sort.shape, m_sort.dtype)
    
    # plt.loglog(m_sort, xi_sort, "x")
    # plt.plot(m_sort, xi_sort, "x")
    
    # resort, if masses have changed "places" in the selection process
    ind = np.argsort(m_sort)
    m_sort = m_sort[ind]
    xi_sort = xi_sort[ind]
    
    ### merge particles, which have masses or xis < m_lim, xi_lim
    no_bins0 = no_bins
    no_bins *= 10
    # print("no_bins")
    # print("no_bins0")
    # print(no_bins)
    # print(no_bins0)
    
    no_spc = len(xi_sort)
    n_save = int(no_spc//1000)
    if n_save < 2: n_save = 2
    
    no_rpc = np.sum(xi_sort)
    total_mass = np.sum(xi_sort*m_sort)
    xi_lim = no_rpc / no_bins
    m_lim = total_mass / no_bins
    
    bin_centers = []
    m_bin = []
    xi_bin = []
    
    n_left = no_rpc
    
    i = 0
    while(n_left > 0 and i < len(xi_sort)-n_save):
        bin_mass = 0.0
        bin_xi = 0
        # bin_center = 0.0
        
        # while(bin_xi < n_lim):
        while(bin_mass < m_lim and bin_xi < xi_lim and n_left > 0
              and i < len(xi_sort)-n_save):
            bin_xi += xi_sort[i]
            bin_mass += xi_sort[i] * m_sort[i]
            n_left -= xi_sort[i]
            i += 1
        bin_centers.append(bin_mass / bin_xi)
        m_bin.append(bin_mass)
        xi_bin.append(bin_xi)
            
    # return m_bin, xi_bin, bin_centers    
    
    xi_bin = np.array(xi_bin)
    bin_centers = np.array(bin_centers)
    m_bin = np.array(m_bin)
    
    ### merge particles, whose masses are close together in log space:
    bin_size_log =\
        (np.log10(bin_centers[-1]) - np.log10(bin_centers[0])) / no_bins0
    
    # print("np.sum(xi_bin*bin_centers), m_bin.sum() before")
    # print(np.sum(xi_bin*bin_centers), m_bin.sum())
    
    i = 0
    while(i < len(xi_bin)-1):
        m_next_bin = bin_centers[i] * 10**bin_size_log
        m = bin_centers[i]
        j = i
        while (m < m_next_bin and j < len(xi_bin)-1):
            j += 1
            m = bin_centers[j]
        if m >= m_next_bin:
            j -= 1
        if (i != j):
            m_sum = 0.0
            xi_sum = 0
            for k in range(i,j+1):
                m_sum += m_bin[k]
                xi_sum += xi_bin[k]
                if k > i:
                    xi_bin[k] = 0
            bin_centers[i] = m_sum / xi_sum
            xi_bin[i] = xi_sum
            m_bin[i] = m_sum
        i = j+1            
    
    
    ind = np.nonzero(xi_bin)
    xi_bin = xi_bin[ind]
    bin_centers = bin_centers[ind]        
    m_bin = m_bin[ind]
    
    # print("np.sum(xi_bin*bin_centers), m_bin.sum() after")
    # print(np.sum(xi_bin*bin_centers), m_bin.sum())
    
    ######
    # bin_size = 0.5 * (bin_centers[-1] - bin_centers[0]) / no_bins0
    # bin_size = (bin_centers[-1] - bin_centers[0]) / no_bins0
    
    # print("len(bin_centers) bef =", len(bin_centers))
    
    # for i, bc in enumerate(bin_centers[:-1]):
    #     if bin_centers[i+1] - bc < bin_size and xi_bin[i] != 0:
    #         m_sum = m_bin[i+1] + m_bin[i]
    #         xi_sum = xi_bin[i+1] + xi_bin[i]
    #         bin_centers[i] = m_sum / xi_sum
    #         xi_bin[i] = xi_sum
    #         xi_bin[i+1] = 0
    #         m_bin[i] = m_sum

    # ind = np.nonzero(xi_bin)
    # xi_bin = xi_bin[ind]
    # m_bin = m_bin[ind]
    # bin_centers = bin_centers[ind]
    ######
    
    # print("len(bin_centers) after =", len(bin_centers))

    # radii = compute_radius_from_mass(m_sort, c.mass_density_water_liquid_NTP)
    radii = compute_radius_from_mass(bin_centers,
                                     c.mass_density_water_liquid_NTP)
    
    ###
    # find the midpoints between the masses/radii
    # midpoints = 0.5 * ( m_sort[:-1] + m_sort[1:] )
    # m_left = 2.0 * m_sort[0] - midpoints[0]
    # m_right = 2.0 * m_sort[-1] - midpoints[-1]
    bins = 0.5 * ( radii[:-1] + radii[1:] )
    # add missing bin borders for m_min and m_max:
    R_left = 2.0 * radii[0] - bins[0]
    R_right = 2.0 * radii[-1] - bins[-1]
    
    bins = np.hstack([R_left, bins, R_right])
    bins_log = np.log(bins)
    # print(midpoints)
       
    # mass_per_ln_R = m_sort * xi_sort
    # mass_per_ln_R *= 1.0E-15/no_sims
    
    m_bin = np.array(m_bin)
    
    g_ln_R = m_bin * 1.0E-15 / no_sims / (bins_log[1:] - bins_log[0:-1]) / dV
    
    return g_ln_R, radii, bins, xi_bin, bin_centers

# masses = masses_vs_time[3]
# xis = xis_vs_time[3]
# radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
# xi_min = 100
# m_bin, xi_bin, bins = auto_bin_SIPs(masses, xis, xi_min)

# r_bin = compute_radius_from_mass(bins, c.mass_density_water_liquid_NTP)

# plt.plot(r_bin, m_bin, "o")

# masses = masses_vs_time[3]
# xis = xis_vs_time[3]
# radii = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
# print(masses.shape)
# print(xis.shape)
# print()

# xi_min = 100
# no_bins = 40
# g_ln_R, R_sort, bins, xi_bi, m_bins = auto_bin_SIPs(masses,
#                                                     xis, xi_min, no_bins,
#                                                     dV, no_sims)

# fig = plt.figure()
# ax = plt.gca()
# ax.loglog(R_sort, g_ln_R, "x")
# # ax.plot(R_sort, g_ln_R, "x")
# # ax.plot(R_sort, xi_bin)

# ###

# method = "log_R"

# R_min = 0.99*np.amin(radii)
# R_max = 1.01*np.amax(radii)
# # R_max = 3.0*np.amax(radii)
# print("R_min=", R_min)
# print("R_max=", R_max)

# no_bins = 20
# if method == "log_R":
#     bins = np.logspace(np.log10(R_min), np.log10(R_max), no_bins)
# elif method == "lin_R":
#     bins = np.linspace(R_min, R_max, no_bins)
# # print(bins)

# # masses in 10^-15 gram
# mass_per_ln_R, _ = np.histogram(radii, bins, weights=masses*xis)
# # convert to gram
# mass_per_ln_R *= 1.0E-15/no_sims
# # print(mass_per_ln_R)
# # print(mass_per_ln_R.shape, bins.shape)

# bins_log = np.log(bins)
# # bins_mid = np.exp((bins_log[1:] + bins_log[:-1]) * 0.5)
# bins_mid = (bins[1:] + bins[:-1]) * 0.5

# g_ln_R = mass_per_ln_R / (bins_log[1:] - bins_log[0:-1]) / dV

# # print(g_ln_R.shape)
# # print(np.log(bins_mid[1:])-np.log(bins_mid[0:-1]))
# ax.loglog( bins_mid, g_ln_R, "-" )
###


#%% PLOT MASS DENSITY DISTRIBUTIONS VS RADIUS

# we need g_ln_R vs R, where g_ln_R = 3 * m^2 * f_m(m)
# where f_m(m) = 1/dm * 1/dV * sum_(m_a in [m,m+dm]) {xi_a}
# i.e: for all m_i,x_i:
# sort particle in histogram with R_i(m_i) and weight with x_i * 3 * m_i^2

# dV = 0.1**3

no_bins = 30
xi_min = 1

fig_name = path + f"mass_distr_per_ln_R_vs_time_no_sims_{no_sims}_dV_{dV}.png"
fig, ax = plt.subplots(figsize=(8,8))

# method = "lin_R"
# method = "log_R"

# for ind_t in range(1):
for ind_t in range(len(times)):
    masses = masses_vs_time[ind_t]
    xis = xis_vs_time[ind_t]
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
    
    print("m_ges =", np.sum(masses*xis))
    print("no_rpc =", np.sum(xis))
    
    # no_bins = 20
    # if method == "log_R":
    #     bins = np.logspace(np.log10(R_min), np.log10(R_max), no_bins)
    # elif method == "lin_R":
    #     bins = np.linspace(R_min, R_max, no_bins)
    # # print(bins)
    
    # # masses in 10^-15 gram
    # mass_per_ln_R, _ = np.histogram(radii, bins, weights=masses*xi)
    # # convert to gram
    # mass_per_ln_R *= 1.0E-15/no_sims
    # # print(mass_per_ln_R)
    # # print(mass_per_ln_R.shape, bins.shape)
    
    # bins_log = np.log(bins)
    # # bins_mid = np.exp((bins_log[1:] + bins_log[:-1]) * 0.5)
    # bins_mid = (bins[1:] + bins[:-1]) * 0.5
    
    # g_ln_R = mass_per_ln_R / (bins_log[1:] - bins_log[0:-1]) / dV
    
    # print(g_ln_R.shape)
    # print(np.log(bins_mid[1:])-np.log(bins_mid[0:-1]))

    
    g_ln_R, bins_mid, bins, xi_bin, mass_bin =\
        auto_bin_SIPs(masses, xis, xi_min, no_bins, dV, no_sims)
    
    print("m_ges after binning=", np.sum(xi_bin*mass_bin))
    print("no_rpc =", np.sum(xi_bin))        
    
    ax.loglog( bins_mid, g_ln_R, "-" )
    # ax.loglog( bins_mid, g_ln_R, "o", markersize=5.0 )
    

# ax.loglog( bins_mid, np.ones_like(bins_mid), "o" )
ax.set_xlim([1.0,1.0E3])
# ax.hist(radii,weights=masses*xi,bins=30)
ax.set_ylim([1.0E-4,1.0E1])
# ax.set_ylim([1.0E-6,1.0E1])
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
    rad = compute_radius_from_mass(masses, c.mass_density_water_liquid_NTP)
    xi = xis_vs_time[ind_t]
    # ax.plot(masses, xi, "o", markersize=1.5)
    # ax.plot(rad, xi, "o", markersize=1.5)
    ax.plot(rad, xi*masses, "o", markersize=1.5)
    ax.set_xscale("log")
    ax.set_yscale("log")
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