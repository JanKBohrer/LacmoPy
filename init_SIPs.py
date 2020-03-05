#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:33:04 2019

@author: jdesk
"""

#%% IMPORTS

import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

import constants as c
# from microphysics import compute_mass_from_radius
from microphysics import compute_radius_from_mass_vec
from microphysics import compute_radius_from_mass_jit
from microphysics import compute_mass_from_radius_vec
from microphysics import compute_mass_from_radius_jit

#%% FUNCTION DEFS

#%% DISTRIBUTIONS


# nth moment of f_m(m) -> mom_n = int(dm * m^k * f_m(m))
# function checked versus analytical values via numerical integration
def moments_f_m_num_expo_np(n, DNC, DNC_over_LWC, steps=1E6):
    m_avg = 1.0/DNC_over_LWC
    # m_high = m_avg * steps**0.7
    m_high = m_avg * 1.0E4
    dm = m_high / steps
    m = 0.0
    intl = 0.0
    # cnt = 0
    if n == 0:
        f1 = conc_per_mass_expo(m, DNC, DNC_over_LWC)
        while (m < m_high):
            f2 = conc_per_mass_expo(m + 0.5*dm, DNC, DNC_over_LWC)
            f3 = conc_per_mass_expo(m + dm, DNC, DNC_over_LWC)
            # intl_bef = intl        
            intl += 0.1666666666667 * dm * (f1 + 4.0 * f2 + f3)
            m += dm
            f1 = f3
            # cnt += 1        
            # intl += dx * x * dst_expo(x,k)
            # x += dx
            # cnt += 1
    else:
        f1 = conc_per_mass_expo(m, DNC, DNC_over_LWC) * m**n
        while (m < m_high):
            f2 = conc_per_mass_expo(m + 0.5*dm, DNC, DNC_over_LWC) * (m + 0.5*dm)**n
            f3 = conc_per_mass_expo(m + dm, DNC, DNC_over_LWC) * (m + dm)**n
            # intl_bef = intl        
            intl += 0.1666666666667 * dm * (f1 + 4.0 * f2 + f3)
            m += dm
            f1 = f3
            # cnt += 1        
            # intl += dx * x * dst_expo(x,k)
            # x += dx
            # cnt += 1
    return intl
moments_f_m_num_expo = njit()(moments_f_m_num_expo_np)

#%% GENERATION OF SIP ENSEMBLES

### SingleSIP probabilistic

# r_critmin => m_low = m_0
# m_{l+1} = m_l * 10^(1/kappa)
# dm_l = m_{l+1} - m_l
# mu_l = m_l + rnd() * dm_l
# xi_l = f_m(mu_l) * dm_l * dV
def generate_SIP_ensemble_SingleSIP_Unt_expo_np(
        DNC0, DNC0_over_LWC0,
        mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low=1.0E6,
        seed=3711, setseed=True):
    if setseed: np.random.seed(seed)
    m_low = 1.0E-18 * compute_mass_from_radius_jit(r_critmin,
                                               mass_density)
    bin_factor = 10**(1.0/kappa)
    m_high = m_low * m_high_over_m_low
    m_left = m_low

    l_max = int(kappa * np.log10(m_high_over_m_low))
    rnd = np.random.rand( l_max )
    
    if weak_threshold:
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
        xi = conc_per_mass_expo(mu, DNC0, DNC0_over_LWC0) * bin_width * dV
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
            if weak_threshold:
                if rnd2[bin_n] < xis[bin_n] / xi_critmin:
                    xis[bin_n] = xi_critmin
                else: valid_ids[bin_n] = 0
            else: valid_ids[bin_n] = 0
    xis = xis[np.nonzero(valid_ids)[0]]
    masses = masses[np.nonzero(valid_ids)[0]]
    
    return masses, xis, m_low, bins
generate_SIP_ensemble_SingleSIP_Unt_expo =\
    njit()(generate_SIP_ensemble_SingleSIP_Unt_expo_np)

# r_critmin -> m_low = m_0
# m_{l+1} = m_l * 10^(1/kappa)
# dm_l = m_{l+1} - m_l
# mu_l = m_l + rnd() * dm_l
# xi_l = f_m(mu_l) * dm_l * dV
def generate_SIP_ensemble_SingleSIP_Unt_lognormal_np(
        DNC0, mu_m_log, sigma_m_log,
        mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low=1.0E6, seed=3711, setseed=True):
    if setseed: np.random.seed(seed)
    m_low = 1.0E-18 * compute_mass_from_radius_jit(r_critmin, mass_density)

    bin_factor = 10**(1.0/kappa)
    m_high = m_low * m_high_over_m_low
    m_left = m_low

    l_max = int(kappa * np.log10(m_high_over_m_low))
    rnd = np.random.rand( l_max )
    if weak_threshold:
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

        xi = conc_per_mass_lognormal(mu, DNC0, mu_m_log, sigma_m_log) \
             * bin_width * dV
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
            if weak_threshold:
                if rnd2[bin_n] < xis[bin_n] / xi_critmin:
                    xis[bin_n] = xi_critmin
                else: valid_ids[bin_n] = 0
            else: valid_ids[bin_n] = 0
    xis = xis[np.nonzero(valid_ids)[0]]
    masses = masses[np.nonzero(valid_ids)[0]]

    return masses, xis, m_low, bins
generate_SIP_ensemble_SingleSIP_Unt_lognormal =\
    njit()(generate_SIP_ensemble_SingleSIP_Unt_lognormal_np)

### GENERATE AND SAVE SIP ENSEMBLES SINGLE SIP UNTERSTRASSER
# r_critmin in mu
def generate_and_save_SIP_ensembles_SingleSIP_prob(
        dist, dist_par, mass_density, dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low, no_sims, start_seed, ensemble_dir):
    if dist == "expo":
        generate_SIP_ensemble_SingleSIP_Unt = \
            generate_SIP_ensemble_SingleSIP_Unt_expo
        DNC0 = dist_par[0]
        DNC0_over_LWC0 = dist_par[1]
        ensemble_parameters = [dV, DNC0, DNC0_over_LWC0, r_critmin,
                               kappa, eta, no_sims, start_seed]
    elif dist == "lognormal":
        generate_SIP_ensemble_SingleSIP_Unt = \
            generate_SIP_ensemble_SingleSIP_Unt_lognormal
        DNC0 = dist_par[0]
        mu_m_log = dist_par[1]
        sigma_m_log = dist_par[2]
#        mass_density = dist_par[3]
        ensemble_parameters = [dV, DNC0, mu_m_log, sigma_m_log, mass_density,
                               r_critmin, kappa, eta, no_sims, start_seed]
    m_low = 1.0E-18 * compute_mass_from_radius_jit(r_critmin, mass_density)

    seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)

    if not os.path.exists(ensemble_dir):
        os.makedirs(ensemble_dir)
    
    for i,seed in enumerate(seed_list):
        masses, xis, m_low, bins =\
            generate_SIP_ensemble_SingleSIP_Unt(
                *dist_par, mass_density, dV, kappa, eta, weak_threshold,
                r_critmin, m_high_over_m_low, seed)
        bins_rad = compute_radius_from_mass_vec(1.0E18*bins, mass_density)
        radii = compute_radius_from_mass_vec(1.0E18*masses, mass_density)
#        bins_rad = compute_radius_from_mass(1.0E18*bins,
#                                            mass_density)
#        radii = compute_radius_from_mass(1.0E18*masses,
#                                         mass_density)
        np.save(ensemble_dir + f"masses_seed_{seed}", masses)
        np.save(ensemble_dir + f"radii_seed_{seed}", radii)
        np.save(ensemble_dir + f"xis_seed_{seed}", xis)
        
        if i == 0:
            np.save(ensemble_dir + f"bins_mass", bins)
            np.save(ensemble_dir + f"bins_rad", bins_rad)
            np.save(ensemble_dir + "ensemble_parameters", ensemble_parameters)

#%% TESTING LOGNORMAL DIST
#DNC0 = 60.0E6 # 1/m^3
#
##DNC0 = 2.97E8 # 1/m^3
##DNC0 = 3.0E8 # 1/m^3
#
#### for expo dist
#LWC0 = 1.0E-3 # kg/m^3
#
#### for lognormal dist
#
#mu_R = 0.02 # in mu
#sigma_R = 1.4 #  no unit
#
#mu_R_log = np.log(mu_R)
#sigma_R_log = np.log(sigma_R)
#
#mass_density = c.mass_density_NaCl_dry # in kg/m^3           
#
#dist_par_R = (DNC0, mu_R_log, sigma_R_log)
#dist_par_m = (DNC0, mu_R_log, sigma_R_log, mass_density)
#
#R_ = np.logspace(-3,1,1000) 
#f_R_ana = conc_per_mass_lognormal_R(R_, *dist_par_R)
#
#m_ = 1.0E-18*compute_mass_from_radius(R_, mass_density)
#f_m_ana = conc_per_mass_lognormal(m_, *dist_par_m)
#
#intl_R = num_int_lognormal_R (R_[0], R_[-1], dist_par_R)
#intl_m = num_int_lognormal_m (m_[0], m_[-1], dist_par_m)
#
#mean_R = num_int_lognormal_mean_R(R_[0], R_[-1], dist_par_R)
#mean_m = num_int_lognormal_mean_m(m_[0], m_[-1], dist_par_m)
#
#mean_mass_R = num_int_lognormal_mean_mass_R(R_[0], R_[-1], dist_par_m)
##mean_mass_R = num_int_lognormal_mean_mass_R(R_[0], R_[-1], dist_par_m, steps = 1E8)
#
#moments_R_ana = []
#moments_m_ana = []
#
#for n in range(4):
#    moments_R_ana.append( moments_analytical_lognormal_R(n, *dist_par_R) )
#    moments_m_ana.append( moments_analytical_lognormal(n, *dist_par_m) )
#
#moments_R_ana = np.array(moments_R_ana)
#moments_m_ana = np.array(moments_m_ana)
#
#print( moments_R_ana)
#print( moments_m_ana)
#
#print(moments_m_ana/moments_R_ana)
#
#print(intl_R, (intl_R - DNC0) / DNC0)
#print(intl_m, (intl_m - DNC0) / DNC0)
#
#print(mean_R, mu_R*DNC0, moments_R_ana[1])
#print(mean_m, moments_m_ana[1], mean_m/mean_mass_R)
##print(mean_m/DNC0, mu_R)
#
#fig,axes = plt.subplots(2)
#ax = axes[0]
#ax.plot(R_, f_R_ana)
#ax.plot(R_, f_m_ana * 3.0*m_/R_)
#ax.set_xscale("log")
#ax = axes[1]
#ax.plot(R_, f_m_ana)
#ax.set_xscale("log")
#
#fig.tight_layout()

#%% CODE FRAGMENTS
### MOMENTS ANALOG to Wang 2007 and Unterstr 2017
# moments_num = np.zeros(4, dtype = np.float64)
# for n in range(4):
#     if n == 0:
#         moments_num[n] = np.sum( g_m_num / bins_mass_center * bins_mass_width )
#     elif n == 1:
#         moments_num[n] = np.sum( g_m_num * bins_mass_width)
#     else:
#         moments_num[n] = np.sum( g_m_num * bins_mass_center**(n-1) * bins_mass_width )
#for n in range(4):
#    if n == 0:
#        moments_num[n] = np.log(10)/(3.0*kappa) * np.sum( g_m_num / bins_mass_center )
#    elif n == 1:
#        moments_num[n] = np.log(10)/(3.0*kappa) * np.sum( g_m_num )
#    else:
#        moments_num[n] = np.log(10)/(3.0*kappa) * np.sum( g_m_num * bins_mass_center**n )


###### WORKING VERSION!
#def analyze_ensemble_data(dist, mass_density, kappa, no_sims, ensemble_dir,
#                          sample_mode, no_bins, bin_mode,
#                          spread_mode, shift_factor, overflow_factor,
#                          scale_factor):
#    if dist == "expo":
#        conc_per_mass_np = conc_per_mass_expo_np
#        dV, DNC0, DNC0_over_LWC0, r_critmin, kappa, eta, no_sims00, start_seed = \
#            tuple(np.load(ensemble_dir + "ensemble_parameters.npy"))
#        LWC0_over_DNC0 = 1.0 / DNC0_over_LWC0
#        dist_par = (DNC0, DNC0_over_LWC0)
#        moments_analytical = moments_analytical_expo
#    elif dist =="lognormal":
#        conc_per_mass_np = conc_per_mass_lognormal_np
#        dV, DNC0, mu_m_log, sigma_m_log, mass_density, r_critmin, \
#        kappa, eta, no_sims00, start_seed = \
#            tuple(np.load(ensemble_dir + "ensemble_parameters.npy"))
#        dist_par = (DNC0, mu_m_log, sigma_m_log)
#        moments_analytical = moments_analytical_lognormal_m
#
#    start_seed = int(start_seed)
#    no_sims00 = int(no_sims00)
#    kappa = int(kappa)
#    seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
#    
#    ### ANALYSIS START
#    masses = []
#    xis = []
#    radii = []
#    
#    moments_sampled = []
#    for i,seed in enumerate(seed_list):
#        masses.append(np.load(ensemble_dir + f"masses_seed_{seed}.npy"))
#        xis.append(np.load(ensemble_dir + f"xis_seed_{seed}.npy"))
#        radii.append(np.load(ensemble_dir + f"radii_seed_{seed}.npy"))
#    
#        moments = np.zeros(4,dtype=np.float64)
#        moments[0] = xis[i].sum() / dV
#        for n in range(1,4):
#            moments[n] = np.sum(xis[i]*masses[i]**n) / dV
#        moments_sampled.append(moments)
#    
#    masses_sampled = np.concatenate(masses)
#    radii_sampled = np.concatenate(radii)
#    xis_sampled = np.concatenate(xis)
#    
#    # moments analysis
#    moments_sampled = np.transpose(moments_sampled)
#    moments_an = np.zeros(4,dtype=np.float64)
#    for n in range(4):
#        moments_an[n] = moments_analytical(n, *dist_par)
#        
#    print(f"######## kappa {kappa} ########")    
#    print("moments_an: ", moments_an)    
#    for n in range(4):
#        print(n, (np.average(moments_sampled[n])-moments_an[n])/moments_an[n] )
#    
#    moments_sampled_avg_norm = np.average(moments_sampled, axis=1) / moments_an
#    moments_sampled_std_norm = np.std(moments_sampled, axis=1) \
#                               / np.sqrt(no_sims) / moments_an
#    
#    m_min = masses_sampled.min()
#    m_max = masses_sampled.max()
#    
#    R_min = radii_sampled.min()
#    R_max = radii_sampled.max()
#    
#    if sample_mode == "given_bins":
#        bins_mass = np.load(ensemble_dir + "bins_mass.npy")
#        bins_rad = np.load(ensemble_dir + "bins_rad.npy")
#        bin_factor = 10**(1.0/kappa)
#    
#    ### build log bins "intuitively"
#    elif sample_mode == "auto_bins":
#        if bin_mode == 1:
#            bin_factor = (m_max/m_min)**(1.0/no_bins)
#            # bin_log_dist = np.log(bin_factor)
#            # bin_log_dist_half = 0.5 * bin_log_dist
#            # add dummy bins for overflow
#            # bins_mass = np.zeros(no_bins+3,dtype=np.float64)
#            bins_mass = np.zeros(no_bins+1,dtype=np.float64)
#            bins_mass[0] = m_min
#            # bins_mass[0] = m_min / bin_factor
#            for bin_n in range(1,no_bins+1):
#                bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
#            # the factor 1.01 is for numerical stability: to be sure
#            # that m_max does not contribute to a bin larger than the
#            # last bin
#            bins_mass[-1] *= 1.0001
#            # the factor 0.99 is for numerical stability: to be sure
#            # that m_min does not contribute to a bin smaller than the
#            # 0-th bin
#            bins_mass[0] *= 0.9999
#            # m_0 = m_min / np.sqrt(bin_factor)
#            # bins_mass_log = np.log(bins_mass)
#
#        bins_rad = compute_radius_from_mass_vec(bins_mass*1.0E18, mass_density)
#
#    ### histogram generation
#    f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
#    f_m_ind = np.nonzero(f_m_counts)[0]
#    f_m_ind = np.arange(f_m_ind[0],f_m_ind[-1]+1)
#    
#    no_SIPs_avg = f_m_counts.sum()/no_sims
#
#    bins_mass_ind = np.append(f_m_ind, f_m_ind[-1]+1)
#    
#    bins_mass = bins_mass[bins_mass_ind]
#    
#    bins_rad = bins_rad[bins_mass_ind]
#    bins_rad_log = np.log(bins_rad)
#    bins_mass_width = (bins_mass[1:]-bins_mass[:-1])
#    bins_rad_width = (bins_rad[1:]-bins_rad[:-1])
#    bins_rad_width_log = (bins_rad_log[1:]-bins_rad_log[:-1])
#    
#    ### approximate the functions f_m, f_lnR = 3*m*f_m, g_lnR=3*m^2*f_m
#    # estimate f_m(m) by binning:
#    # DNC_i = f_m(m_i) * dm_i = droplet number conc in bin i with size dm_i
#    f_m_num_sampled = np.histogram(masses_sampled,bins_mass,
#                                   weights=xis_sampled)[0]
#    g_m_num_sampled = np.histogram(masses_sampled,bins_mass,
#                                   weights=xis_sampled*masses_sampled)[0]
#    
#    f_m_num_sampled = f_m_num_sampled / (bins_mass_width * dV * no_sims)
#    g_m_num_sampled = g_m_num_sampled / (bins_mass_width * dV * no_sims)
#    
#    # build g_ln_r = 3*m*g_m DIRECTLY from data
#    g_ln_r_num_sampled = np.histogram(radii_sampled,
#                                      bins_rad,
#                                      weights=xis_sampled*masses_sampled)[0]
#    g_ln_r_num_sampled = g_ln_r_num_sampled \
#                         / (bins_rad_width_log * dV * no_sims)
#    # g_ln_r_num_derived = 3 * bins_mass_center * g_m_num * 1000.0
#    
#    # define centers on lin scale
#    bins_mass_center_lin = 0.5 * (bins_mass[:-1] + bins_mass[1:])
#    bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])
#    
#    # define centers on the logarithmic scale
#    bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
#    bins_rad_center_log = bins_rad[:-1] * np.sqrt(bin_factor)
#    # bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
#    # bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
#    
#    # define the center of mass for each bin and set it as the "bin center"
#    bins_mass_center_COM = g_m_num_sampled/f_m_num_sampled
#    bins_rad_center_COM =\
#        compute_radius_from_mass_vec(bins_mass_center_COM*1.0E18, mass_density)
#    
#    # set the bin "mass centers" at the right spot such that
#    # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
#    if dist == "expo":
#        m_avg = LWC0_over_DNC0
#    elif dist == "lognormal":
#        m_avg = moments_an[1] / dist_par[0]
#        
#    bins_mass_center_exact = bins_mass[:-1] \
#                             + m_avg * np.log(bins_mass_width\
#          / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))
#    bins_rad_center_exact =\
#        compute_radius_from_mass_vec(bins_mass_center_exact*1.0E18, mass_density)
#    
#    bins_mass_centers = np.array((bins_mass_center_lin,
#                                  bins_mass_center_log,
#                                  bins_mass_center_COM,
#                                  bins_mass_center_exact))
#    bins_rad_centers = np.array((bins_rad_center_lin,
#                                  bins_rad_center_log,
#                                  bins_rad_center_COM,
#                                  bins_rad_center_exact))
#    
#    m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
#    R_ = compute_radius_from_mass_vec(m_*1.0E18, mass_density)
#    f_m_ana = conc_per_mass_np(m_, *dist_par)
#    g_m_ana = m_ * f_m_ana
#    g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0
#
#    ### STATISTICAL ANALYSIS OVER no_sim runs
#    # get f(m_i) curve for each "run" with same bins for all ensembles
#    f_m_num = []
#    g_m_num = []
#    g_ln_r_num = []
#    
#    for i,mass in enumerate(masses):
#        f_m_num.append(np.histogram(mass,bins_mass,weights=xis[i])[0] \
#                   / (bins_mass_width * dV))
#        g_m_num.append(np.histogram(mass,bins_mass,
#                                       weights=xis[i]*mass)[0] \
#                   / (bins_mass_width * dV))
#    
#        # build g_ln_r = 3*m*g_m DIRECTLY from data
#        g_ln_r_num.append(np.histogram(radii[i],
#                                          bins_rad,
#                                          weights=xis[i]*mass)[0] \
#                     / (bins_rad_width_log * dV))
#    
#    f_m_num = np.array(f_m_num)
#    g_m_num = np.array(g_m_num)
#    g_ln_r_num = np.array(g_ln_r_num)
#    
#    f_m_num_avg = np.average(f_m_num, axis=0)
#    f_m_num_std = np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
#    g_m_num_avg = np.average(g_m_num, axis=0)
#    g_m_num_std = np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
#    g_ln_r_num_avg = np.average(g_ln_r_num, axis=0)
#    g_ln_r_num_std = np.std(g_ln_r_num, axis=0, ddof=1) / np.sqrt(no_sims)
#
###############################################################################
#    
#    ### generate f_m, g_m and mass centers with my hist bin method
#    LWC0 = moments_an[1]
#    f_m_num_avg_my_ext, f_m_num_std_my_ext, g_m_num_avg_my, g_m_num_std_my, \
#    h_m_num_avg_my, h_m_num_std_my, \
#    bins_mass_my, bins_mass_width_my, \
#    bins_mass_centers_my, bins_mass_center_lin_my, lin_par, aa = \
#        generate_myHisto_SIP_ensemble_np(masses, xis, m_min, m_max,
#                                         dV, DNC0, LWC0,
#                                         no_bins, no_sims,
#                                         bin_mode, spread_mode,
#                                         shift_factor, overflow_factor,
#                                         scale_factor)
#        
#    f_m_num_avg_my = f_m_num_avg_my_ext[1:-1]
#    f_m_num_std_my = f_m_num_std_my_ext[1:-1]
#    
#    
#
###############################################################################
#    return bins_mass, bins_rad, bins_rad_log, \
#           bins_mass_width, bins_rad_width, bins_rad_width_log, \
#           bins_mass_centers, bins_rad_centers, \
#           masses, xis, radii, f_m_counts, f_m_ind,\
#           f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled,\
#           m_, R_, f_m_ana, g_m_ana, g_ln_r_ana, \
#           f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std, \
#           g_ln_r_num_avg, g_ln_r_num_std, \
#           m_min, m_max, R_min, R_max, no_SIPs_avg, \
#           moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
#           moments_an, \
#           f_m_num_avg_my_ext, \
#           f_m_num_avg_my, f_m_num_std_my, \
#           g_m_num_avg_my, g_m_num_std_my, \
#           h_m_num_avg_my, h_m_num_std_my, \
#           bins_mass_my, bins_mass_width_my, \
#           bins_mass_centers_my, bins_mass_center_lin_my, lin_par, aa
