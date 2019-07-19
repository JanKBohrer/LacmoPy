#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 18:33:04 2019

@author: jdesk
"""

#%% IMPORTS AND DEFINITIONS

import os
import math
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

import constants as c
# from microphysics import compute_mass_from_radius
from microphysics import compute_radius_from_mass
from microphysics import compute_mass_from_radius

### FUNCTION DEF: CREATE SIP ENSEMBLE AND ANALYSIS

##############################################################################
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
def conc_per_mass_expo_np(m, DNC, DNC_over_LWC): # = f_m(m)
    return DNC * DNC_over_LWC * np.exp(-DNC_over_LWC * m)
conc_per_mass_expo = njit()(conc_per_mass_expo_np)

# lognormal given for radius -> convert mass to radius
# m = mass in kg
# mu_R = geometric expect.value of the PDF
# sigma_R = geometric standard dev. of the PDF
# PDF for DNC = 1.0 tested for conc_per_mass_lognormal_R: numer. integr. = 1.0
# function PDF_m(m) = PDF_R(R) checked -> exact up to numeric errors 1E-13
two_pi_sqrt = np.sqrt(2.0*np.pi)
def conc_per_mass_lognormal_np(m, DNC, mu_R_log, sigma_R_log): # = f_m(m)
    R = compute_radius_from_mass(1.0E18*m, c.mass_density_water_liquid_NTP)
    return DNC * np.exp( -0.5*( ( np.log( R ) - mu_R_log ) / sigma_R_log )**2 ) \
           / (R * two_pi_sqrt * sigma_R_log)
conc_per_mass_lognormal = njit()(conc_per_mass_lognormal_np)

# PDF for DNC = 1.0 tested: numeric integral = 1.0
def conc_per_mass_lognormal_R_np(R, DNC, mu_R_log, sigma_R_log): # = f_m(m)
    return DNC * np.exp( -0.5*( ( np.log( R ) - mu_R_log ) / sigma_R_log )**2 ) \
           / (R * two_pi_sqrt * sigma_R_log)
conc_per_mass_lognormal_R = njit()(conc_per_mass_lognormal_R_np)

# this function needs x0, x1 in radius values
def num_int_lognormal_np(x0, x1, par, steps=1E5):
    dx = (x1 - x0) / steps
    x = x0
    intl = 0.0
    f1 = conc_per_mass_lognormal_R(x,par[0],par[1],par[2])
    # cnt = 0
    while (x < x1):
        f2 = conc_per_mass_lognormal_R(x + 0.5*dx, par[0],par[1],par[2])
        f3 = conc_per_mass_lognormal_R(x + dx, par[0],par[1],par[2])
        # intl_bef = intl        
        intl += 0.1666666666667 * dx * (f1 + 4.0 * f2 + f3)
        x += dx
        f1 = f3
        # cnt += 1        
        # intl += dx * x * dst_expo(x,k)
        # x += dx
        # cnt += 1
    return intl
num_int_lognormal = njit()(num_int_lognormal_np)

# mu_R = 0.02
# sigma_R = 1.4
# n0 = 60.0E6
# n0 = 1.0
# dist_par = np.array([n0])
# dist_par = np.append(dist_par, np.log((mu_R, sigma_R)))

# R = np.logspace(-2.7,-1.2,1000)
# m = 1.0E-18 * compute_mass_from_radius(R, c.mass_density_water_liquid_NTP)

# m0 = 1.0E-20
# m1 = 1.0E-17

# R0 = 1.0E-6
# R1 = 1.0

# # print(num_int_lognormal(m[0],m[-1], dist_par,steps = 1.0E8) )
# print(num_int_lognormal(R0, R1, dist_par, steps = 1.0E6) )
###
# mu_R = 0.02
# sigma_R = 1.4
# n0 = 60.0E6
# n0 = 1.0
# dist_par = np.array([n0])
# dist_par = np.append(dist_par, np.log((mu_R, sigma_R)))


# fig, ax = plt.subplots()
# # m = np.logspace(-18,-12,1000)
# # ax.plot( R, conc_per_mass_lognormal(m, *dist_par) )
# # ax.plot( R, conc_per_mass_lognormal_R(R, *dist_par) )

# ax.plot( R,
#         ( conc_per_mass_lognormal_R(R, *dist_par)
#           - conc_per_mass_lognormal(m, *dist_par) ) \
#         / conc_per_mass_lognormal_R(R, *dist_par) )

# # ax.plot( m, conc_per_mass_lognormal_np(m, *dist_par) )
# # ax.semilogx( m, conc_per_mass_lognormal_np(m, *dist_par) )
# # ax.loglog( m, conc_per_mass_lognormal_np(m, n0, *dist_par) )

# conc_per_mass = conc_per_mass_expo
# conc_per_mass_np = conc_per_mass_expo_np

##############################################################################

def moments_analytical(n, DNC, LWC_over_DNC):
    if n == 0:
        return DNC
    else:
        return math.factorial(n) * DNC * LWC_over_DNC**n

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

# print( math.factorial(2-1) * LWC0**2 / DNC0 * 2 )

# SingleSIP probabilistic

# r_critmin -> m_low = m_0
# m_{l+1} = m_l * 10^(1/kappa)
# dm_l = m_{l+1} - m_l
# mu_l = m_l + rnd() * dm_l
# xi_l = f_m(mu_l) * dm_l * dV
def generate_SIP_ensemble_SingleSIP_Unt_expo_np(DNC0, DNC0_over_LWC0,
                                                dV, kappa, eta, r_critmin,
                                                m_high_over_m_low=1.0E6,
                                                seed=3711, setseed=True):
    if setseed: np.random.seed(seed)
    m_low = 1.0E-18 * compute_mass_from_radius(r_critmin,
                                               c.mass_density_water_liquid_NTP)
    # DNC0_over_LWC0 = DNC0 / LWC0
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
        xi = conc_per_mass_expo(mu, DNC0, DNC0_over_LWC0) * bin_width * dV
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
generate_SIP_ensemble_SingleSIP_Unt_expo =\
    njit()(generate_SIP_ensemble_SingleSIP_Unt_expo_np)

# r_critmin -> m_low = m_0
# m_{l+1} = m_l * 10^(1/kappa)
# dm_l = m_{l+1} - m_l
# mu_l = m_l + rnd() * dm_l
# xi_l = f_m(mu_l) * dm_l * dV
def generate_SIP_ensemble_SingleSIP_Unt_lognormal_np(
        DNC0, mu_R_log, sigma_R_log,
        dV, kappa, eta, r_critmin,
        m_high_over_m_low=1.0E6, seed=3711, setseed=True):
    if setseed: np.random.seed(seed)
    m_low = 1.0E-18 * compute_mass_from_radius(r_critmin,
                                               c.mass_density_water_liquid_NTP)
    # DNC0_over_LWC0 = DNC0 / LWC0
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

        xi = conc_per_mass_lognormal(mu, DNC0, mu_R_log, sigma_R_log) \
             * bin_width * dV
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
generate_SIP_ensemble_SingleSIP_Unt_lognormal =\
    njit()(generate_SIP_ensemble_SingleSIP_Unt_lognormal_np)

### GENERATE AND SAVE SIP ENSEMBLES SINGLE SIP UNTERSTRASSER
# r_critmin in mu
def generate_and_save_SIP_ensembles_SingleSIP_prob(
        dist, dist_par, dV, kappa, eta, r_critmin,
        m_high_over_m_low, no_sims, start_seed, ensemble_dir):
    if dist == "expo":
        generate_SIP_ensemble_SingleSIP_Unt = \
            generate_SIP_ensemble_SingleSIP_Unt_expo
        DNC0 = dist_par[0]
        DNC0_over_LWC0 = dist_par[1]
        ensemble_parameters = [dV, DNC0, LWC0, r_critmin,
                               kappa, eta, no_sims, start_seed]
    elif dist == "lognormal":
        generate_SIP_ensemble_SingleSIP_Unt = \
            generate_SIP_ensemble_SingleSIP_Unt_lognormal
        DNC0 = dist_par[0]
        mu_R_log = dist_par[1]
        sigma_R_log = dist_par[2]
        ensemble_parameters = [dV, DNC0, mu_R_log, sigma_R_log, r_critmin,
                               kappa, eta, no_sims, start_seed]
    # r_critmin = 0.6 # mu
    m_low = 1.0E-18 * compute_mass_from_radius(r_critmin,
                                               c.mass_density_water_liquid_NTP)

    # dV = 1.0
    # kappa = 10
    # eta = 1.0E-9

    # DNC0 = 2.97E8 # 1/m^3
    #DNC0 = 3.0E8 # 1/m^3
    # LWC0 = 1.0E-3 # kg/m^3

    # LWC0_inv = 1.0 / LWC0
    # DNC0_over_LWC0 = DNC0 / LWC0

    # print(r_critmin, m_low)
    # print(conc_per_mass(0.0, DNC0, DNC0_over_LWC0))
    # print(conc_per_mass(m_low, DNC0, DNC0_over_LWC0))

    #no_bins = 10

    # no_sims = 1000
    # start_seed = 3711

    seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)

    if not os.path.exists(ensemble_dir):
        os.makedirs(ensemble_dir)
    
    for i,seed in enumerate(seed_list):
        masses, xis, m_low, bins =\
            generate_SIP_ensemble_SingleSIP_Unt(
                *dist_par, dV, kappa, eta, r_critmin,
                m_high_over_m_low, seed)
        bins_rad = compute_radius_from_mass(1.0E18*bins,
                                            c.mass_density_water_liquid_NTP)
        radii = compute_radius_from_mass(1.0E18*masses,
                                         c.mass_density_water_liquid_NTP)
        np.save(ensemble_dir + f"masses_seed_{seed}", masses)
        np.save(ensemble_dir + f"radii_seed_{seed}", radii)
        np.save(ensemble_dir + f"xis_seed_{seed}", xis)
        
        if i == 0:
            np.save(ensemble_dir + f"bins_mass", bins)
            np.save(ensemble_dir + f"bins_rad", bins_rad)
            np.save(ensemble_dir + "ensemble_parameters", ensemble_parameters)

### ANALYZE EXPO SIP ENSEMBLE DATA FROM DATA STORED IN FILES

# def analyze_ensemble_data_sampled(kappa, no_sims, save_directory):
#     ### DERIVED
#     folder = f"kappa_{kappa}/"
#     path = save_directory + folder
    
#     dV, DNC0, LWC0, r_critmin, kappa, eta, no_sims00, start_seed = \
#         tuple(np.load(path + "ensemble_parameters.npy"))
#     LWC0_over_DNC0 = LWC0 / DNC0
#     DNC0_over_LWC0 = DNC0 / LWC0
#     start_seed = int(start_seed)
#     no_sims00 = int(no_sims00)
#     kappa = int(kappa)
#     seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
    
#     ### ANALYSIS START
#     masses = []
#     xis = []
#     radii = []
    
#     # masses_sampled = []
#     # radii_sampled = []
#     # xis_sampled = []
#     moments_sampled = []
#     for i,seed in enumerate(seed_list):
#         masses.append(np.load(path + f"masses_seed_{seed}.npy"))
#         xis.append(np.load(path + f"xis_seed_{seed}.npy"))
#         radii.append(np.load(path + f"radii_seed_{seed}.npy"))
#         # masses_sampled.append(masses)
#         # xis_sampled.append(xis)
    
#         moments = np.zeros(4,dtype=np.float64)
#         moments[0] = xis[i].sum() / dV
#         for n in range(1,4):
#             moments[n] = np.sum(xis[i]*masses[i]**n) / dV
#         moments_sampled.append(moments)
    
#     masses_sampled = np.concatenate(masses)
#     radii_sampled = np.concatenate(radii)
#     xis_sampled = np.concatenate(xis)
    
#     # moments analysis
#     moments_sampled = np.transpose(moments_sampled)
#     moments_an = np.zeros(4,dtype=np.float64)
#     for n in range(4):
#         moments_an[n] = moments_analytical(n, DNC0, LWC0_over_DNC0)
    
#     for n in range(4):
#         print(n, (np.average(moments_sampled[n])-moments_an[n])/moments_an[n] )
    
#     moments_sampled_avg_norm = np.average(moments_sampled, axis=1) / moments_an
#     moments_sampled_std_norm = np.std(moments_sampled, axis=1) / moments_an
    
#     m_min = masses_sampled.min()
#     m_max = masses_sampled.max()
    
#     R_min = radii_sampled.min()
#     R_max = radii_sampled.max()
    
#     # H1 = np.histogram(radii_sampled, bins_rad)[0]
#     # H2 = np.histogram(radii_sampled, bins_rad, weights=xis_sampled)[0]
    
#     # H1 = H1[np.nonzero(H1)[0]]
#     # H2 = H2[np.nonzero(H1)[0]]
    
#     # H = H2 / H1
    
#     bins_mass = np.load(path + "bins_mass.npy")
#     bins_rad = np.load(path + "bins_rad.npy")
    
#     f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
#     f_m_ind = np.nonzero(f_m_counts)[0]

#     no_SIPs_avg = f_m_counts.sum()/no_sims

#     bins_mass_ind = np.append(f_m_ind, f_m_ind[-1]+1)
    
#     bins_mass = bins_mass[bins_mass_ind]
#     bins_rad = bins_rad[bins_mass_ind]
#     bins_rad_log = np.log(bins_rad)
#     bins_mass_width = (bins_mass[1:]-bins_mass[:-1])
#     bins_rad_width = (bins_rad[1:]-bins_rad[:-1])
#     bins_rad_width_log = (bins_rad_log[1:]-bins_rad_log[:-1])
    
#     ### approximate the functions f_m, f_lnR = 3*m*f_m, g_lnR=3*m^2*f_m
#     # estimate f_m(m) by binning:
#     # DNC_i = f_m(m_i) * dm_i = droplet number conc in bin i with size dm_i
#     f_m_num_sampled = np.histogram(masses_sampled,bins_mass,
#                                    weights=xis_sampled)[0]
#     g_m_num_sampled = np.histogram(masses_sampled,bins_mass,
#                                    weights=xis_sampled*masses_sampled)[0]
    
#     f_m_num_sampled = f_m_num_sampled / (bins_mass_width * dV * no_sims)
#     g_m_num_sampled = g_m_num_sampled / (bins_mass_width * dV * no_sims)
    
#     # build g_ln_r = 3*m*g_m DIRECTLY from data
#     g_ln_r_num_sampled = np.histogram(radii_sampled,
#                                       bins_rad,
#                                       weights=xis_sampled*masses_sampled)[0]
#     g_ln_r_num_sampled = g_ln_r_num_sampled \
#                          / (bins_rad_width_log * dV * no_sims)
#     # g_ln_r_num_derived = 3 * bins_mass_center * g_m_num * 1000.0
    
#     # define centers on lin scale
#     bins_mass_center_lin = 0.5 * (bins_mass[:-1] + bins_mass[1:])
#     bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])
    
#     # define centers on the logarithmic scale
#     bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
#     bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
    
#     # define the center of mass for each bin and set it as the "bin center"
#     bins_mass_center_COM = g_m_num_sampled/f_m_num_sampled
#     bins_rad_center_COM =\
#         compute_radius_from_mass(bins_mass_center_COM*1.0E18,
#                                  c.mass_density_water_liquid_NTP)
    
#     # set the bin "mass centers" at the right spot such that
#     # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
#     m_avg = LWC0_over_DNC0
#     bins_mass_center_exact = bins_mass[f_m_ind] \
#                              + m_avg * np.log(bins_mass_width\
#           / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))
#     bins_rad_center_exact =\
#         compute_radius_from_mass(bins_mass_center_exact*1.0E18,
#                                  c.mass_density_water_liquid_NTP)
    
#     bins_mass_centers = np.array((bins_mass_center_lin,
#                                   bins_mass_center_log,
#                                   bins_mass_center_COM,
#                                   bins_mass_center_exact))
#     bins_rad_centers = np.array((bins_rad_center_lin,
#                                   bins_rad_center_log,
#                                   bins_rad_center_COM,
#                                   bins_rad_center_exact))
    
#     m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
#     R_ = compute_radius_from_mass(m_*1.0E18, c.mass_density_water_liquid_NTP)
#     f_m_ana = conc_per_mass_np(m_, DNC0, DNC0_over_LWC0)
#     g_m_ana = m_ * f_m_ana
#     g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0

#     return bins_mass, bins_rad, bins_rad_log, \
#            bins_mass_width, bins_rad_width, bins_rad_width_log, \
#            bins_mass_centers, bins_rad_centers, \
#            masses, xis, radii, f_m_counts, f_m_ind,\
#            f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled,\
#            m_, R_, f_m_ana, g_m_ana, g_ln_r_ana, \
#            m_min, m_max, R_min, R_max, no_SIPs_avg, \
#            moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
#            moments_an

def analyze_ensemble_data(dist, kappa, no_sims, ensemble_dir,
                          sample_mode, no_bins=50, bin_mode=1):
    ### DERIVED
    # folder = f"kappa_{kappa}/"
    # path = save_directory + folder
    
#    start_seed = 3711
#    no_sims00 = 50
#    kappa = 40
#    print(no_sims)
#    no_sims = int(no_sims)
    
    if dist == "expo":
        conc_per_mass_np = conc_per_mass_expo_np
        dV, DNC0, LWC0, r_critmin, kappa, eta, no_sims00, start_seed = \
            tuple(np.load(ensemble_dir + "ensemble_parameters.npy"))
        LWC0_over_DNC0 = LWC0 / DNC0
        DNC0_over_LWC0 = DNC0 / LWC0
        dist_par = (DNC0, DNC0_over_LWC0)
#        start_seed = int(start_seed)
    elif dist =="lognormal":
        conc_per_mass_np = conc_per_mass_lognormal_np
        dV, DNC0, mu_R_log, sigma_R_log, r_critmin, \
        kappa, eta, no_sims00, start_seed = \
            tuple(np.load(ensemble_dir + "ensemble_parameters.npy"))
        dist_par = (DNC0, mu_R_log, sigma_R_log)
#        start_seed = int(start_seed)

#    print(start_seed)
    
    start_seed = int(start_seed)
    no_sims00 = int(no_sims00)
    kappa = int(kappa)
    seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
    
    ### ANALYSIS START
    masses = []
    xis = []
    radii = []
    
    # masses_sampled = []
    # radii_sampled = []
    # xis_sampled = []
    moments_sampled = []
    for i,seed in enumerate(seed_list):
        masses.append(np.load(ensemble_dir + f"masses_seed_{seed}.npy"))
        xis.append(np.load(ensemble_dir + f"xis_seed_{seed}.npy"))
        radii.append(np.load(ensemble_dir + f"radii_seed_{seed}.npy"))
        # masses_sampled.append(masses)
        # xis_sampled.append(xis)
    
        moments = np.zeros(4,dtype=np.float64)
        moments[0] = xis[i].sum() / dV
        for n in range(1,4):
            moments[n] = np.sum(xis[i]*masses[i]**n) / dV
        moments_sampled.append(moments)
    
    masses_sampled = np.concatenate(masses)
    radii_sampled = np.concatenate(radii)
    xis_sampled = np.concatenate(xis)
    
    # moments analysis
    moments_sampled = np.transpose(moments_sampled)
    moments_an = np.zeros(4,dtype=np.float64)
    for n in range(4):
        moments_an[n] = moments_analytical(n, DNC0, LWC0_over_DNC0)
    
    for n in range(4):
        print(n, (np.average(moments_sampled[n])-moments_an[n])/moments_an[n] )
    
    moments_sampled_avg_norm = np.average(moments_sampled, axis=1) / moments_an
    moments_sampled_std_norm = np.std(moments_sampled, axis=1) \
                               / np.sqrt(no_sims) / moments_an
    
    m_min = masses_sampled.min()
    m_max = masses_sampled.max()
    
    R_min = radii_sampled.min()
    R_max = radii_sampled.max()
    
    # H1 = np.histogram(radii_sampled, bins_rad)[0]
    # H2 = np.histogram(radii_sampled, bins_rad, weights=xis_sampled)[0]
    
    # H1 = H1[np.nonzero(H1)[0]]
    # H2 = H2[np.nonzero(H1)[0]]
    
    # H = H2 / H1

    if sample_mode == "given_bins":
        bins_mass = np.load(ensemble_dir + "bins_mass.npy")
        bins_rad = np.load(ensemble_dir + "bins_rad.npy")
        bin_factor = 10**(1.0/kappa)
    ### build log bins "intuitively"
    elif sample_mode == "auto_bins":
        if bin_mode == 1:
            bin_factor = (m_max/m_min)**(1.0/no_bins)
            # bin_log_dist = np.log(bin_factor)
            # bin_log_dist_half = 0.5 * bin_log_dist
            # add dummy bins for overflow
            # bins_mass = np.zeros(no_bins+3,dtype=np.float64)
            bins_mass = np.zeros(no_bins+1,dtype=np.float64)
            bins_mass[0] = m_min
            # bins_mass[0] = m_min / bin_factor
            for bin_n in range(1,no_bins+1):
                bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
            # the factor 1.01 is for numerical stability: to be sure
            # that m_max does not contribute to a bin larger than the
            # last bin
            bins_mass[-1] *= 1.0001
            # the factor 0.99 is for numerical stability: to be sure
            # that m_min does not contribute to a bin smaller than the
            # 0-th bin
            bins_mass[0] *= 0.9999
            # m_0 = m_min / np.sqrt(bin_factor)
            # bins_mass_log = np.log(bins_mass)

        bins_rad = compute_radius_from_mass(bins_mass*1.0E18,
                                            c.mass_density_water_liquid_NTP)

    ### histogram generation
    f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
    f_m_ind = np.nonzero(f_m_counts)[0]
    f_m_ind = np.arange(f_m_ind[0],f_m_ind[-1]+1)

    no_SIPs_avg = f_m_counts.sum()/no_sims

    bins_mass_ind = np.append(f_m_ind, f_m_ind[-1]+1)
    
    bins_mass = bins_mass[bins_mass_ind]
    bins_rad = bins_rad[bins_mass_ind]
    bins_rad_log = np.log(bins_rad)
    bins_mass_width = (bins_mass[1:]-bins_mass[:-1])
    bins_rad_width = (bins_rad[1:]-bins_rad[:-1])
    bins_rad_width_log = (bins_rad_log[1:]-bins_rad_log[:-1])
    
    ### approximate the functions f_m, f_lnR = 3*m*f_m, g_lnR=3*m^2*f_m
    # estimate f_m(m) by binning:
    # DNC_i = f_m(m_i) * dm_i = droplet number conc in bin i with size dm_i
    f_m_num_sampled = np.histogram(masses_sampled,bins_mass,
                                   weights=xis_sampled)[0]
    g_m_num_sampled = np.histogram(masses_sampled,bins_mass,
                                   weights=xis_sampled*masses_sampled)[0]
    
    f_m_num_sampled = f_m_num_sampled / (bins_mass_width * dV * no_sims)
    g_m_num_sampled = g_m_num_sampled / (bins_mass_width * dV * no_sims)
    
    # build g_ln_r = 3*m*g_m DIRECTLY from data
    g_ln_r_num_sampled = np.histogram(radii_sampled,
                                      bins_rad,
                                      weights=xis_sampled*masses_sampled)[0]
    g_ln_r_num_sampled = g_ln_r_num_sampled \
                         / (bins_rad_width_log * dV * no_sims)
    # g_ln_r_num_derived = 3 * bins_mass_center * g_m_num * 1000.0
    
    # define centers on lin scale
    bins_mass_center_lin = 0.5 * (bins_mass[:-1] + bins_mass[1:])
    bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])
    
    # define centers on the logarithmic scale
    bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
    bins_rad_center_log = bins_rad[:-1] * np.sqrt(bin_factor)
    # bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
    # bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
    
    # define the center of mass for each bin and set it as the "bin center"
    bins_mass_center_COM = g_m_num_sampled/f_m_num_sampled
    bins_rad_center_COM =\
        compute_radius_from_mass(bins_mass_center_COM*1.0E18,
                                 c.mass_density_water_liquid_NTP)
    
    # set the bin "mass centers" at the right spot such that
    # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
    m_avg = LWC0_over_DNC0
    bins_mass_center_exact = bins_mass[f_m_ind] \
                             + m_avg * np.log(bins_mass_width\
          / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))
    bins_rad_center_exact =\
        compute_radius_from_mass(bins_mass_center_exact*1.0E18,
                                 c.mass_density_water_liquid_NTP)
    
    bins_mass_centers = np.array((bins_mass_center_lin,
                                  bins_mass_center_log,
                                  bins_mass_center_COM,
                                  bins_mass_center_exact))
    bins_rad_centers = np.array((bins_rad_center_lin,
                                  bins_rad_center_log,
                                  bins_rad_center_COM,
                                  bins_rad_center_exact))
    
    m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
    R_ = compute_radius_from_mass(m_*1.0E18, c.mass_density_water_liquid_NTP)
    f_m_ana = conc_per_mass_np(m_, *dist_par)
    g_m_ana = m_ * f_m_ana
    g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0

    ### STATISTICAL ANALYSIS OVER no_sim runs
    # get f(m_i) curve for each "run" with same bins for all ensembles
    f_m_num = []
    g_m_num = []
    g_ln_r_num = []
    
    for i,mass in enumerate(masses):
        f_m_num.append(np.histogram(mass,bins_mass,weights=xis[i])[0] \
                   / (bins_mass_width * dV))
        g_m_num.append(np.histogram(mass,bins_mass,
                                       weights=xis[i]*mass)[0] \
                   / (bins_mass_width * dV))
    
        # build g_ln_r = 3*m*g_m DIRECTLY from data
        g_ln_r_num.append(np.histogram(radii[i],
                                          bins_rad,
                                          weights=xis[i]*mass)[0] \
                     / (bins_rad_width_log * dV))
    
    f_m_num = np.array(f_m_num)
    g_m_num = np.array(g_m_num)
    g_ln_r_num = np.array(g_ln_r_num)
    
    f_m_num_avg = np.average(f_m_num, axis=0)
    f_m_num_std = np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    g_m_num_avg = np.average(g_m_num, axis=0)
    g_m_num_std = np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    g_ln_r_num_avg = np.average(g_ln_r_num, axis=0)
    g_ln_r_num_std = np.std(g_ln_r_num, axis=0, ddof=1) / np.sqrt(no_sims)

    return bins_mass, bins_rad, bins_rad_log, \
           bins_mass_width, bins_rad_width, bins_rad_width_log, \
           bins_mass_centers, bins_rad_centers, \
           masses, xis, radii, f_m_counts, f_m_ind,\
           f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled,\
           m_, R_, f_m_ana, g_m_ana, g_ln_r_ana, \
           f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std, \
           g_ln_r_num_avg, g_ln_r_num_std, \
           m_min, m_max, R_min, R_max, no_SIPs_avg, \
           moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
           moments_an

# masses is a list of [masses0, masses1, ..., masses_no_sims]
# where masses_i = array of masses of a spec. SIP ensemble
def generate_myHisto_SIP_ensemble_np(masses, xis, m_min, m_max,
                                     dV, DNC0, LWC0,
                                     no_bins, no_sims,
                                     bin_mode, spread_mode,
                                     shift_factor, overflow_factor,
                                     scale_factor):
    # g_m_num = []
    # g_ln_r_num = []
    if bin_mode == 1:
            bin_factor = (m_max/m_min)**(1.0/no_bins)
            bin_log_dist = np.log(bin_factor)
            # bin_log_dist_half = 0.5 * bin_log_dist
            # add dummy bins for overflow
            # bins_mass = np.zeros(no_bins+3,dtype=np.float64)
            bins_mass = np.zeros(no_bins+1,dtype=np.float64)
            bins_mass[0] = m_min
            # bins_mass[0] = m_min / bin_factor
            for bin_n in range(1,no_bins+1):
                bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
            # the factor 1.01 is for numerical stability: to be sure
            # that m_max does not contribute to a bin larger than the
            # last bin
            bins_mass[-1] *= 1.0001
            # the factor 0.99 is for numerical stability: to be sure
            # that m_min does not contribute to a bin smaller than the
            # 0-th bin
            bins_mass[0] *= 0.9999
            # m_0 = m_min / np.sqrt(bin_factor)
            bins_mass_log = np.log(bins_mass)

    bins_mass_width = np.zeros(no_bins+2,dtype=np.float64)
    bins_mass_width[1:-1] = bins_mass[1:]-bins_mass[:-1]
    # modify for overflow bins
    bins_mass_width[0] = bins_mass_width[1]
    bins_mass_width[-1] = bins_mass_width[-2]
    dm0 = 0.5*bins_mass_width[0]
    dmN = 0.5*bins_mass_width[-1]
    # dm0 = 0.5*(bins_mass[0] - bins_mass[0] / bin_factor)
    # dmN = 0.5*(bins_mass[-1] * bin_factor - bins_mass[-1])

    ###
    # if bin_mode == 1:
    #         bin_factor = (m_max/m_min)**(1.0/no_bins)
    #         bin_log_dist = np.log(bin_factor)
    #         # bin_log_dist_half = 0.5 * bin_log_dist
    #         bins_mass = np.zeros(no_bins+1,dtype=np.float64)
    #         bins_mass[0] = m_min
    #         for bin_n in range(1,no_bins+1):
    #             bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
    #         # the factor 1.01 is for numerical stability: to be sure
    #         # that m_max does not contribute to a bin larger than the
    #         # last bin
    #         bins_mass[0] *= 0.9999
    #         bins_mass[-1] *= 1.0001
    #         bins_mass_log = np.log(bins_mass)
    #         # the factor 0.99 is for numerical stability: to be sure
    #         # that m_min does not contribute to a bin smaller than the
    #         # 0-th bin
    #         # m_0 = m_min / np.sqrt(bin_factor)


    # bins_mass_width = bins_mass[1:]-bins_mass[:-1]
    # dm0 = 0.5*(bins_mass[0] - bins_mass[0] / bin_factor)
    # dmN = 0.5*(bins_mass[-1] * bin_factor - bins_mass[-1])

##########################################################################
            ## OLD
            # bin_factor = (m_max/m_min)**(1.0/(no_bins-1.0))
            # bin_log_dist = np.log(bin_factor)
            # bin_log_dist_half = 0.5 * bin_log_dist
            # # the factor 0.99 is for numerical stability: to be sure
            # # that m_min does not contribute to a bin smaller than the
            # # 0-th bin
            # m_0 = m_min / np.sqrt(bin_factor)

            # bins_mass = np.zeros(no_bins+1,dtype=np.float64)
            # bins_mass[0] = m_0
            # for bin_n in range(1,no_bins+1):
            #     bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
            # # the factor 1.01 is for numerical stability: to be sure
            # # that m_max does not contribute to a bin larger than the
            # # last bin
            # bins_mass[0] *= 0.999
            # bins_mass[-2] *= 1.001
            # bins_mass_log = np.log(bins_mass)

            # bins_mass_width = bins_mass[1:]-bins_mass[:-1]

            # bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
##########################################################################

    f_m_num = np.zeros( (no_sims,no_bins+2), dtype=np.float64 )
    # g_m_num = np.zeros( (no_sims,no_bins+2), dtype=np.float64 )
    # h_m_num = np.zeros( (no_sims,no_bins+2), dtype=np.float64 )

    # f_m_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )
    g_m_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )
    h_m_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )

    # g_ln_r_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )
    for i,mass in enumerate(masses):
        histo = np.zeros(no_bins+2, dtype=np.float64)
        histo_g = np.zeros(no_bins+2, dtype=np.float64)
        histo_h = np.zeros(no_bins+2, dtype=np.float64)
        mass_log = np.log(mass)
        for n,m_ in enumerate(mass):
            xi = xis[i][n]
            bin_n = np.nonzero(np.histogram(m_, bins=bins_mass)[0])[0][0]
            # bin_n = np.nonzero(np.histogram(m_, bins=bins_mass)[0])[0][0]
            # print(bin_n)

            # smear functions depending on weight of data point in the bin
            # on a lin base
            if spread_mode == 0:
                # norm_dist = (mass[n] - bins_mass[bin_n])/bins_mass_width[bin_n]
                # NEW: start from right side
                norm_dist = (bins_mass[bin_n+1] - mass[n]) \
                            / bins_mass_width[bin_n]
            # on a log base
            elif spread_mode == 1:
                # norm_dist = (mass_log[n] - bins_mass_log[bin_n])/bin_log_dist
                norm_dist = (bins_mass_log[bin_n] - mass_log[n])/bin_log_dist
            if norm_dist < 0.5:
                s = 0.5 + norm_dist

                # +1 because we have overrun bins left and right in "histo"-array
                bin_n += 1
                # print(n,s,"right")
                histo[bin_n+1] += (1.0-s)*xi
                histo_g[bin_n+1] += (1.0-s)*xi*m_
                histo_h[bin_n+1] += (1.0-s)*xi*m_*m_
                # if in last bin: no outflow,
                # just extrapolation to overflow bin!
                if bin_n == no_bins:
                    histo[bin_n] += xi
                    histo_g[bin_n] += xi*m_
                    histo_h[bin_n] += xi*m_*m_
                else:
                    histo[bin_n] += s*xi
                    histo_g[bin_n] += s*xi*m_
                    histo_h[bin_n] += s*xi*m_*m_
            elif spread_mode == 0:
                # now left side of bin
                norm_dist = (mass[n] - bins_mass[bin_n]) \
                            / bins_mass_width[bin_n-1]
                # +1 because we have overrun bins left and right in "histo"-array
                bin_n += 1
                # print(n,norm_dist, "left")
                if norm_dist < 0.5:
                    s = 0.5 + norm_dist
                    # print(n,s,"left")
                    histo[bin_n-1] += (1.0-s)*xi
                    histo_g[bin_n-1] += (1.0-s)*xi*m_
                    histo_h[bin_n-1] += (1.0-s)*xi*m_*m_
                    # if in first bin: no outflow,
                    # just extrapolation to overflow bin!
                    if bin_n == 1:
                        histo[bin_n] += xi
                        histo_g[bin_n] += xi*m_
                        histo_h[bin_n] += xi*m_*m_
                    else:
                        histo[bin_n] += s*xi
                        histo_g[bin_n] += s*xi*m_
                        histo_h[bin_n] += s*xi*m_*m_
                else:
                    histo[bin_n] += xi
                    histo_g[bin_n] += xi*m_
                    histo_h[bin_n] += xi*m_*m_
            elif spread_mode == 1:
                # +1 because we have overrun bins left and right in "histo"-array
                bin_n += 1
                s = 1.5 - norm_dist
                histo[bin_n] += s*xi
                histo[bin_n-1] += (1.0-s)*xi
                histo_g[bin_n] += s*xi*m_
                histo_g[bin_n-1] += (1.0-s)*xi*m_
                histo_h[bin_n] += s*xi*m_*m_
                histo_h[bin_n-1] += (1.0-s)*xi*m_*m_

            # on a log base
            # log_dist = mass_log[n] - bins_mass_log[bin_n]
            # if log_dist < bin_log_dist_half:
            #     s = 0.5 + log_dist/bin_log_dist
            #     # print(n,s,"left")
            #     histo[bin_n] += s*xi
            #     histo[bin_n-1] += (1.0-s)*xi
            #     histo_g[bin_n] += s*xi*m_
            #     histo_g[bin_n-1] += (1.0-s)*xi*m_
            # else:
            #     s = 1.5 - log_dist/bin_log_dist
            #     # print(n,s,"right")
            #     histo[bin_n] += s*xi
            #     histo[bin_n+1] += (1.0-s)*xi
            #     histo_g[bin_n] += s*xi*m_
            #     histo_g[bin_n+1] += (1.0-s)*xi*m_

        f_m_num[i,1:-1] = histo[1:-1] / (bins_mass_width[1:-1] * dV)
        # g_m_num[i,1:-1] = histo_g[1:-1] / (bins_mass_width * dV)
        # h_m_num[i,1:-1] = histo_h[1:-1] / (bins_mass_width * dV)

        # multiply the overflow-bins by factor to get an estimation of
        # f_m at the position m_0 - dm0/2
        # f_m at the position m_no_bins + dmN/2, where
        # dm0 = 0.5*(bins_mass[0] - bins_mass[0] / bin_factor)
        # dmN = 0.5*(bins_mass[-1] * bin_factor - bins_mass[-1])
        f_m_num[i,0] = overflow_factor * histo[0] / (dm0 * dV)
        f_m_num[i,-1] = overflow_factor * histo[-1] / (dmN * dV)

        # not necessary anymore. first and last bin have no outflow
        # just interpolation to overflow bin
        # multiply first and last "regular" bin by 8/7
        # because there is no inflow, but only outflow of xi in these bins
        #      ' IN |OUT '   | ... |    ' OUT| IN '
        # add here some multiplication factor including delta_m[n-1]/delta_m[n]
        # f_m_num[i,1] *= 1.14285714
        # f_m_num[i,-2] *= 1.14285714

        ### OLD
        # f_m_num[i] = histo[1:-1] / (bins_mass_width * dV)
        g_m_num[i] = histo_g[1:-1] / (bins_mass_width[1:-1] * dV)
        h_m_num[i] = histo_h[1:-1] / (bins_mass_width[1:-1] * dV)


    # f_m_num_avg = f_m_num.mean()
    # f_m_num_std = f_m_num.std() / np.sqrt(no_sims)
    # g_m_num_avg = g_m_num.mean()
    # g_m_num_std = g_m_num.std() / np.sqrt(no_sims)
    f_m_num_avg = np.average(f_m_num, axis=0)
    f_m_num_std = np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    g_m_num_avg = np.average(g_m_num, axis=0)
    g_m_num_std = np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    h_m_num_avg = np.average(h_m_num, axis=0)
    h_m_num_std = np.std(h_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    # define centers on lin scale
    bins_mass_center_lin = np.zeros(no_bins+2, dtype=np.float64)
    bins_mass_center_lin[1:-1] = 0.5 * (bins_mass[:-1] + bins_mass[1:])
    # add dummy bin centers for quadratic approx
    bins_mass_center_lin[0] = bins_mass[0] - 0.5*dm0
    bins_mass_center_lin[-1] = bins_mass[-1] + 0.5*dmN


    # bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])
    
    # define centers on the logarithmic scale
    bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
    # bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
    # bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
    # define the center of mass for each bin and set it as the "bin center"
    bins_mass_center_COM = g_m_num_avg / f_m_num_avg[1:-1]
    # bins_rad_center_COM =\
    #     compute_radius_from_mass(bins_mass_center_COM*1.0E18,
    #                              c.mass_density_water_liquid_NTP)
    # def as 2nd moment/1st moment
    bins_mass_center_h_g = h_m_num_avg / g_m_num_avg
    # set the bin "mass centers" at the right spot such that
    # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"

    # bins_rad_center_exact =\
    #     compute_radius_from_mass(bins_mass_center_exact*1.0E18,
    #                              c.mass_density_water_liquid_NTP)


    ### LINEAR APPROX OF f_m
    # to get an idea of the shape
    # for bin n take f[n-1], f[n], f[n+1]
    # make linear approx from n-1 to n and from n to n+1
    # to get idea of shape of function
    # lin fct: f = a0 + a1*m
    # a1 = (f[n+1]-f[n])/(m[n+1] - m[n])
    # a0 = f[n] - a1*m[n]
    # bins_mass_centers_lin_fit = np.zeros(no_bins, dtype = np.float64)
    lin_par0 = np.zeros(no_bins+1, dtype = np.float64)
    lin_par1 = np.zeros(no_bins+1, dtype = np.float64)

    lin_par1 = (f_m_num_avg[1:] - f_m_num_avg[:-1]) \
               / (bins_mass_center_lin[1:] - bins_mass_center_lin[:-1])
    lin_par0 = f_m_num_avg[:-1] - lin_par1 * bins_mass_center_lin[:-1]

    f_bin_border = lin_par0 + lin_par1 * bins_mass
    # f_bin_border_delta_left = np.zeros(no_bins+1, dtype = np.float64)
    # f_bin_border_delta_left = np.abs(f_m_num_avg[1:-1]-f_bin_border[:-1])
    # f_bin_border_delta_right = np.abs(f_bin_border[1:] - f_m_num_avg[1:-1])

    ### FIRST CORRECTION:
    # by my method of spreading over several bins the bins with higher f_avg
    # "loose" counts to bins with smaller f_avg
    # by a loss/gain analysis, one can estimate the lost counts
    # using the linear approximation of f_m(m) calc. above

    # delta of counts (estimated)
    delta_N = np.zeros(no_bins, dtype=np.float64)

    delta_N[1:-1] = 0.25 * bins_mass_width[1:-3] \
                    * ( f_m_num_avg[1:-3] - f_bin_border[1:-2] ) \
                    + 0.25 * bins_mass_width[2:-2] \
                      * ( -f_m_num_avg[2:-2] + f_bin_border[2:-1] ) \
                    + 0.083333333 \
                      * ( lin_par1[1:-2] * bins_mass_width[1:-3]**2
                          - lin_par1[2:-1] * bins_mass_width[2:-2]**2)
    # first bin: only exchange with the bin to the right
    delta_N[0] = 0.25 * bins_mass_width[1] \
                 * ( -f_m_num_avg[1] + f_bin_border[1] ) \
                 - 0.083333333 \
                   * ( lin_par1[1] * bins_mass_width[1]**2 )
    # last bin: only exchange with the bin to the left
    # bin_n = no_bins-1
    delta_N[no_bins-1] = 0.25 * bins_mass_width[no_bins-1] \
                         * (f_m_num_avg[no_bins-1] - f_bin_border[no_bins-1]) \
                         + 0.083333333 \
                           * ( lin_par1[no_bins-1]
                               * bins_mass_width[no_bins-1]**2 )
    scale = delta_N / (f_m_num_avg[1:-1] * bins_mass_width[1:-1])
    scale = np.where(scale < -0.9,
                     -0.9,
                     scale)
    scale *= scale_factor
    print(scale)
    f_m_num_avg[1:-1] = f_m_num_avg[1:-1] / (1.0 + scale)
    f_m_num_avg[0] = f_m_num_avg[0] / (1.0 + scale[0])
    f_m_num_avg[-1] = f_m_num_avg[-1] / (1.0 + scale[-1])

    ## REPEAT LIN APPROX AFTER FIRST CORRECTION
    lin_par0 = np.zeros(no_bins+1, dtype = np.float64)
    lin_par1 = np.zeros(no_bins+1, dtype = np.float64)

    lin_par1 = (f_m_num_avg[1:] - f_m_num_avg[:-1]) \
                / (bins_mass_center_lin[1:] - bins_mass_center_lin[:-1])
    lin_par0 = f_m_num_avg[:-1] - lin_par1 * bins_mass_center_lin[:-1]

    f_bin_border = lin_par0 + lin_par1 * bins_mass

    # SECOND CORRECTION:
    # try to estimate the position of m in the bin where f(m) = f_avg (of bin)
    # bin avg based on the linear approximations
    # NOTE that this is just to get an idea of the function FORM
    # f_bin_border_delta_left = np.zeros(no_bins+1, dtype = np.float64)
    f_bin_border_delta_left = np.abs(f_m_num_avg[1:-1]-f_bin_border[:-1])
    f_bin_border_delta_right = np.abs(f_bin_border[1:] - f_m_num_avg[1:-1])

    bins_mass_centers_lin_fit = np.zeros(no_bins, dtype = np.float64)

    f_avg2 = 0.25 * (f_bin_border[:-1] + f_bin_border[1:]) \
             + 0.5 * f_m_num_avg[1:-1]

    for bin_n in range(no_bins):
        if f_bin_border_delta_left[bin_n] >= f_bin_border_delta_right[bin_n]:
            m_c = (f_avg2[bin_n] - lin_par0[bin_n]) / lin_par1[bin_n]
        else:
            m_c = (f_avg2[bin_n] - lin_par0[bin_n+1]) / lin_par1[bin_n+1]

        # if f_bin_border_abs[bin_n] >= f_bin_border_abs[bin_n+1]:
        #     # take left side of current bin
        #     m_c = 0.5 * ( (bins_mass[bin_n] + 0.25*bins_mass_width[bin_n]) \
        #           + lin_par1[bin_n+1]/lin_par1[bin_n] \
        #             * (bins_mass[bin_n+1] - 0.25*bins_mass_width[bin_n]) \
        #           + (lin_par0[bin_n+1] - lin_par0[bin_n]))
        # else:
        #     m_c = 0.5 * ( lin_par1[bin_n]/lin_par1[bin_n+1] \
        #                   * (bins_mass[bin_n] + 0.25*bins_mass_width[bin_n]) \
        #                   + (bins_mass[bin_n+1] - 0.25*bins_mass_width[bin_n])\
        #                   + (lin_par0[bin_n] - lin_par0[bin_n+1]) )
        # add additional shift because of two effects:
        # 1) adding xi-"mass" to bins with smaller f_avg
        # 2) wrong setting of "center" if f_avg[n] > f_avg[n+1]

        m_c = shift_factor * m_c \
              + bins_mass_center_lin[bin_n+1] * (1.0 - shift_factor)

        if m_c < bins_mass[bin_n]:
            m_c = bins_mass[bin_n]
        elif m_c > bins_mass[bin_n+1]:
            m_c = bins_mass[bin_n+1]

        bins_mass_centers_lin_fit[bin_n] = m_c
        # shift more to center: -> is covered by shift_fctor=0.5
        # bins_mass_centers_lin_fit[bin_n] = \
        #     0.5 * (m_c + bins_mass_center_lin[bin_n+1])


    # bin mass center quad approx: -->>> BIG ISSUES!
    # for every bin:
    # assume that the coordinate pairs are right with
    # (m_center_lin, f_avg)
    # approximate the function f_m(m) locally with a parabola to get
    # an estimate of the form of the function
    # assume this parabola in the bin and calculate bin_center_exact


    D_10 = bins_mass_center_lin[1:-1] - bins_mass_center_lin[0:-2]
    D_20 = bins_mass_center_lin[2:] - bins_mass_center_lin[0:-2]
    D_21 = bins_mass_center_lin[2:] - bins_mass_center_lin[1:-1]

    CD_10 = (bins_mass_center_lin[1:-1] + bins_mass_center_lin[0:-2])*D_10
    CD_20 = (bins_mass_center_lin[2:] + bins_mass_center_lin[0:-2])*D_20
    CD_21 = (bins_mass_center_lin[2:] + bins_mass_center_lin[1:-1])*D_21

    a2 = f_m_num_avg[2:]/(D_21*D_20) - f_m_num_avg[1:-1]/(D_21*D_10) \
         + f_m_num_avg[:-2]/(D_10*D_20)
    a1_a2 = (-f_m_num_avg[0:-2]*CD_21 + f_m_num_avg[1:-1]*CD_20
             - f_m_num_avg[2:]*CD_10  ) \
            / (f_m_num_avg[0:-2]*D_21 - f_m_num_avg[1:-1]*D_20
               + f_m_num_avg[2:]*D_10  )
    a1 = a2 * a1_a2
    a0 = f_m_num_avg[1:-1] - a1*bins_mass_center_lin[1:-1] \
         - a2*bins_mass_center_lin[1:-1]**2

    bins_mass_sq = bins_mass*bins_mass

    bins_mass_centers_qfit =\
        -0.5*a1_a2 \
        + np.sqrt( 0.25*(a1_a2)**2
                   + 0.5*a1_a2 * (bins_mass[:-1] + bins_mass[1:])
                   + 0.33333333 * (bins_mass_sq[:-1]
                                   + bins_mass[:-1]*bins_mass[1:]
                                   + bins_mass_sq[1:]) )

    # f_m_num_avg = f_m_num_avg[1:-1]
    # f_m_num_std = f_m_num_std[1:-1]
    bins_mass_center_lin2 = bins_mass_center_lin[1:-1]


    bins_mass_width = bins_mass_width[1:-1]
    m_avg = LWC0 / DNC0
    bins_mass_center_exact = bins_mass[:-1]\
                             + m_avg * np.log(bins_mass_width\
          / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))

    bins_mass_centers = np.array((bins_mass_center_lin2,
                                  bins_mass_center_log,
                                  bins_mass_center_COM,
                                  bins_mass_center_exact,
                                  bins_mass_centers_lin_fit,
                                  bins_mass_centers_qfit,
                                  bins_mass_center_h_g))

##########################################################################
    # for i,mass in enumerate(masses):
    #     histo = np.zeros(no_bins+2, dtype=np.float64)
    #     mass_log = np.log(mass)
    #     for n,m_ in enumerate(mass):
    #         bin_n = np.nonzero(np.histogram(m_, bins=bins_mass)[0])[0][0]
    #         # print(bin_n)
    #         log_dist = mass_log[n] - bins_mass_log[bin_n]

    #         if log_dist < bin_log_dist_half:
    #             s = 0.5 + log_dist/bin_log_dist
    #             print(n,s,"left")
    #             histo[bin_n] += s*xis[i][n]
    #             histo[bin_n-1] += (1.0-s)*xis[i][n]
    #         else:
    #             s = 1.5 - log_dist/bin_log_dist
    #             print(n,s,"right")
    #             histo[bin_n] += s*xis[i][n]
    #             histo[bin_n+1] += (1.0-s)*xis[i][n]

    #     f_m_num[i] = histo[1:-1]/(bins_mass_width * dV)



        # f_m_num.append(np.histogram(mass,bins_mass,weights=xis[i])[0] \
        #            / (bins_mass_width * dV))
        # g_m_num.append(np.histogram(mass,bins_mass,
        #                                weights=xis[i]*mass)[0] \
        #            / (bins_mass_width * dV))

        # # build g_ln_r = 3*m*g_m DIRECTLY from data
        # g_ln_r_num.append(np.histogram(radii[i],
        #                                   bins_rad,
        #                                   weights=xis[i]*mass)[0] \
        #              / (bins_rad_width_log * dV))
##########################################################################
    return f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std,\
           h_m_num_avg, h_m_num_std, \
           bins_mass, bins_mass_width, \
           bins_mass_centers, bins_mass_center_lin, \
           np.array((lin_par0,lin_par1)), np.array((a0,a1,a2))
# generate_myHisto_SIP_ensemble = njit()(generate_myHisto_SIP_ensemble_np)


#%% GENERATE SIP ENSEMBLE AND STORE TO FILES


args = [0,1,1]
# args = [0,1,1]
# args = [1,1,1]

generate = bool(args[0])
analyze = bool(args[1])
plotting = bool(args[2]) # NOTE: plotting needs analyze first (or in buffer)
# generate = True
# analyze = True

### SIP GENERATION SET PARAMETERS

OS = "LinuxDesk"
# OS = "MacOS"

no_sims = 500
start_seed = 3711

# kappa_list=[5,10]
# kappa_list=[800]
#kappa_list=[5,10,20,40]
# kappa_list=[5,10,20,40,60,100]
# kappa_list=[5,10,20,40,60,100,200]
# kappa_list=[5,6,10,20,40,60,100,200,400,600]
# kappa_list=[5,10,20,40,60,100,200,400,600,800]
kappa_list=[5,6,10,20,40,60,100,200,400,600,800]
# eta = 1.0E-8
eta = 1.0E-9
r_critmin = 0.6 # mu
m_high_over_m_low = 1.0E6

gen_method = "SinSIP"
dV = 1.0

dist = "expo"
#dist = "lognormal"

DNC0 = 2.97E8 # 1/m^3
#DNC0 = 3.0E8 # 1/m^3
### for expo dist
LWC0 = 1.0E-3 # kg/m^3

### for lognormal dist

mu_R = 0.02
sigma_R = 1.4

mu_R_log = np.log(mu_R)
sigma_R_log = np.log(sigma_R)

### DATA ANALYSIS SET PARAMETERS

# use the same bins as for generation
#sample_mode = "given_bins"
# for auto binning: set number of bins
sample_mode = "auto_bins"

# no_bins = 10
# no_bins = 25
# no_bins = 30
no_bins = 50
# no_bins = 100
# bin_mode = 1 for bins equal dist on log scale
bin_mode = 1

# for myHisto generation (additional parameters)
spread_mode = 0 # spreading based on lin scale
# spread_mode = 1 # spreading based on log scale
# shift_factor = 1.0
# the artificial bins before the first and after the last one get multiplied
# by this factor (since they only get part of the half of the first/last bin)
overflow_factor = 3.0
# scale factor for the first correction
scale_factor = 1.0
# shift factor for the second correction.
# For shift_factor = 0.5 only half of the effect:
# bin center_new = 0.5 (bin_center_lin_first_corr + shifted bin center)
shift_factor = 0.5

if OS == "MacOS":
    sim_data_path = "/Users/bohrer/sim_data/"
elif OS == "LinuxDesk":
    sim_data_path = "/mnt/D/sim_data2/"

# ensemble_directory =\
# f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/eta_{eta:.0e}/"
    # f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/"
# save_directory = "/mnt/D/sim_data/test_SIP_ensemble_Unt_new2/"

if dist == "expo":
    conc_per_mass_np = conc_per_mass_expo_np
    LWC0_over_DNC0 = LWC0 / DNC0
    DNC0_over_LWC0 = DNC0 / LWC0
    dist_par = (DNC0, DNC0_over_LWC0)
elif dist =="lognormal":
    conc_per_mass_np = conc_per_mass_lognormal_np
    dist_par = (DNC0, mu_R_log, sigma_R_log)

if generate:
    for kappa in kappa_list:
        ensemble_dir =\
            sim_data_path + \
            f"col_box_mod/ensembles/{dist}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/"
            
        generate_and_save_SIP_ensembles_SingleSIP_prob(
        dist, dist_par, dV, kappa, eta, r_critmin,
        m_high_over_m_low, no_sims, start_seed, ensemble_dir)

### DATA ANALYIS

# dV, DNC0, LWC0, r_critmin, kappa00, eta, no_sims00, start_seed = \
#     tuple(np.load(path + "ensemble_parameters.npy"))
# start_seed = int(start_seed)

for kappa in kappa_list:
    ensemble_dir =\
        sim_data_path + \
        f"col_box_mod/ensembles/{dist}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/"
        
    if analyze:
#        print(no_sims)
        bins_mass, bins_rad, bins_rad_log, \
        bins_mass_width, bins_rad_width, bins_rad_width_log, \
        bins_mass_centers, bins_rad_centers, \
        masses, xis, radii, f_m_counts, f_m_ind,\
        f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled, \
        m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, \
        f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std, \
        g_ln_r_num_avg, g_ln_r_num_std, \
        m_min, m_max, R_min, R_max, no_SIPs_avg, \
        moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
        moments_an = \
            analyze_ensemble_data(dist, kappa, no_sims, ensemble_dir,
                                  sample_mode, no_bins=50, bin_mode=1)
#            analyze_ensemble_data(kappa, no_sims, ensemble_dir,
#                                  sample_mode, no_bins, bin_mode)
        # if sample_mode == "given_bins":
        #     bins_mass, bins_rad, bins_rad_log, \
        #     bins_mass_width, bins_rad_width, bins_rad_width_log, \
        #     bins_mass_centers, bins_rad_centers, \
        #     masses, xis, radii, f_m_counts, f_m_ind,\
        #     f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled, \
        #     m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, \
        #     m_min, m_max, R_min, R_max, no_SIPs_avg, \
        #     moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
        #     moments_an = \
        #         analyze_ensemble_data_sampled(kappa, no_sims, save_directory)
        # elif sample_mode == "auto_bins":
        #     bins_mass, bins_rad, bins_rad_log, \
        #     bins_mass_width, bins_rad_width, bins_rad_width_log, \
        #     bins_mass_centers, bins_rad_centers, \
        #     masses, xis, radii, f_m_counts, f_m_ind,\
        #     f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled, \
        #     m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, \
        #     m_min, m_max, R_min, R_max, no_SIPs_avg, \
        #     moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
        #     moments_an = \
        #         analyze_ensemble_data_simple(kappa, no_bins, no_sims,
        #                                      save_directory, bin_mode)
        
        
        
        f_m_num_avg_my_ext, f_m_num_std_my_ext, g_m_num_avg_my, g_m_num_std_my, \
        h_m_num_avg_my, h_m_num_std_my, \
        bins_mass_my, bins_mass_width_my, \
        bins_mass_centers_my, bins_mass_center_lin_my, lin_par, aa =\
            generate_myHisto_SIP_ensemble_np(masses, xis, m_min, m_max, dV, DNC0, LWC0,
                                             no_bins, no_sims, bin_mode, spread_mode,
                                             shift_factor, overflow_factor,
                                             scale_factor)
        f_m_num_avg_my = f_m_num_avg_my_ext[1:-1]
        f_m_num_std_my = f_m_num_std_my_ext[1:-1]

# def fit_quad(x,a,b,c):
#     return a + b*x + c*x*x
# def fit_exp(x,a,lam,b):
#     return a*np.exp(lam*x) + b

# from scipy.optimize import curve_fit

# x = [1,2,3]
# y = [3,4,9]
# # best_vals, covar = curve_fit(fit_quad, x, y)
# p0 = ()
# best_vals, covar = curve_fit(fit_exp, x, y)

### try bin center quad fit from moments per bin... -> THAT DIDNT WORK ;P

# bins_mass_pow = np.zeros((5,len(bins_mass_my)), dtype = np.float64)
# bins_mass_pow[0] = bins_mass_my

# for n in range(1,5):
#     bins_mass_pow[n] = bins_mass_pow[n-1]*bins_mass_pow[0]

# # bins_mass_pow_width = bins_mass_pow[:,1:] - bins_mass_pow[:,:-1]
# D = bins_mass_pow[:,1:] - bins_mass_pow[:,:-1]

# N = f_m_num_avg_my * bins_mass_width_my
# M = g_m_num_avg_my * bins_mass_width_my
# Z = h_m_num_avg_my * bins_mass_width_my

# X1 = 0.66666667 * D[2] / D[1] - 0.5 * D[1]
# X2 = X1 / ( 0.75 * D[3] / D[2] - 0.5 * D[1] )
# X3 = 0.5 * D[3] / D[1] - 0.33333333*D[2]

# # a3 = c, a2 = b, a1 = a

# a3 = ( Z * 3.0 * D[0] / D[2] - N ) * X2 - M * 2.0 * D[0] / D[1] + N
# a3 = a3 / ( ( 0.6 * D[4] / D[2] - 0.33333333 * D[2] ) * X2 - X3)

# a2 = 2.0*D[0]/D[1]*M - N - X3*a3
# a2 = a2/X1

# a1 = (N - 0.5*D[1]*a2 - 0.33333333*D[2]*a3) / D[0]

# bins_mass_centers_qfit =\
#     -0.5*a2/a3 \
#     + np.sqrt( 0.25*(a2/a3)**2 + a2/a3*bins_mass_centers_my[0]
#                + 0.66666667*bins_mass_centers_my[0]**2
#                + (bins_mass_pow[1,:-1] + bins_mass_pow[1,1:]) / 6.0 )

# bins_mass_centers_h_g = h_m_num_avg_my / g_m_num_avg_my

# # bins_mass_centers_my =\
# #     np.concatenate((bins_mass_centers_my,
# #                    np.array((bins_mass_centers_qfit, bins_mass_centers_h_g)))
# #                   )
# # bins_mass_centers_my = np.append(bins_mass_centers_my, bins_mass_centers_h_g,
# #                                  axis=0)

# bins_mass_centers_my3 = np.copy(bins_mass_centers_my[3])

# fig, ax = plt.subplots(1)

# for n in range(len(bins_mass_centers_qfit)):
#     m_ = np.linspace(bins_mass_my[n], bins_mass_my[n+1], 10000)
#     f_ = a1[n] + a2[n] * m_ + a3[n] * m_*m_
#     ax.plot(m_,f_, c = "k")
# m_ = np.linspace(bins_mass_my[0], bins_mass_my[-1], 1000)
# f_ = conc_per_mass_np(m_, DNC0, DNC0/LWC0)
# ax.plot(m_,f_, c = "orange")
# ax.set_xscale("log")
# ax.set_yscale("log")

### IN WORK: conc_per_mass(dist_par) +  TEST FOR KNOWN DISTROS!!!!!!!1

### PLOTTING
### mean xi vs R
    if plotting:
        no_rows = 1
        fig, axes = plt.subplots(nrows=no_rows, figsize=(10,6*no_rows))
        ax=axes
        ax.plot(bins_rad_centers[3], f_m_num_avg*bins_mass_width)
        
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks(np.logspace(-4,8,13))
        ax.set_ylim((1.0E-4,1.0E8))
        ax.grid()
        
        ax.set_xlabel(r"radius ($\mathrm{\mu m}$)")
        ax.set_ylabel(r"mean multiplicity per SIP")
        
        fig.tight_layout()
        fig.savefig(ensemble_dir +
        f"xi_vs_R_{sample_mode}_no_sims_{no_sims}_no_bins_{no_bins}.png")

### my lin approx plot
        bins_mass_center_exact = bins_mass_centers[3]
        bins_rad_center_exact = bins_rad_centers[3]
        
        m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
        R_ = compute_radius_from_mass(m_*1.0E18, c.mass_density_water_liquid_NTP)
#        f_m_ana = conc_per_mass_np(m_, DNC0, DNC0/LWC0)
        f_m_ana = conc_per_mass_np(m_, *dist_par)
        g_m_ana = m_ * f_m_ana
        g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0
        
        no_rows = 1
        
        MS= 15.0
        
        fig, axes = plt.subplots(nrows=no_rows, figsize=(10,10*no_rows))
        # ax.loglog(radii, xis, "x")
        # ax.loglog(bins_mid[:51], H, "x-")
        # ax.vlines(bins_rad, xis.min(), xis.max(), linewidth=0.5, linestyle="dashed")
        ax = axes
        # ax.plot(bins_mass_center_exact, f_m_num_sampled, "x")
        ax.plot(m_, f_m_ana_)
        ax.plot(bins_mass_center_lin_my, f_m_num_avg_my_ext, "x", c="green",
                markersize=MS, zorder=99)
        ax.plot(bins_mass_centers_my[4], f_m_num_avg_my, "x", c = "red",
                markersize=MS,
                zorder=99)
        # for n in range(len(bins_mass_centers_my[0])):
        # lin approx
        for n in range(len(bins_mass_center_lin_my)-1):
            m_ = np.linspace(bins_mass_center_lin_my[n],
                              bins_mass_center_lin_my[n+1], 100)
            f_ = lin_par[0,n] + lin_par[1,n] * m_
            # ax.plot(m_,f_)
            ax.plot(m_,f_, "-.", c = "orange")
        # for n in range(len(bins_mass_center_lin_my)-2):
        #     m_ = np.linspace(bins_mass_center_lin_my[n],
        #                       bins_mass_center_lin_my[n+2], 1000)
        #     f_ = aa[0,n] + aa[1,n] * m_ + aa[2,n] * m_*m_
        #     ax.plot(m_,f_)
        #     # ax.plot(m_,f_, c = "k")
        ax.vlines(bins_mass_my,f_m_num_avg_my_ext[0]*2.0,f_m_num_avg_my_ext[-1]*0.5,
                  linestyle="dashed", zorder=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mass (kg)")
        ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$")
        
        fig.tight_layout()
        fig.savefig(ensemble_dir +
        f"fm_my_lin_approx_{sample_mode}_no_sims_{no_sims}_no_bins_{no_bins}.png")
        
        # f_m_num_avg_my = f_m_num_avg_my[1:-1]
        # f_m_num_std_my = f_m_num_std_my[1:-1]
        
        # ax = axes[1]
        # ax.plot(m_, f_m_ana_)
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_xlabel("mass (kg)")
        # ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$")
        
        # ax = axes[1]
        # ax.plot(bins_mass_center_exact, g_m_num_sampled, "x")
        # ax.plot(m_, g_m_ana_)
        # ax.set_xscale("log")
        # ax.set_yscale("log")
        # ax.set_xlabel("mass (kg)")
        # ax.set_ylabel(r"$g_m$ $\mathrm{(m^{-3})}$")
        # ax.loglog(bins_mid, f_m_num/50, "x")
        # ax.loglog(m_, f_m_ana)
        # ax.set_xlim(m_min*9, m_max/9)
    
### SAMPLED DATA
        ### IN WORK: add errorbars to all...
        
        bins_mass_center_exact = bins_mass_centers[3]
        bins_rad_center_exact = bins_rad_centers[3]
        
        m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
        R_ = compute_radius_from_mass(m_*1.0E18, c.mass_density_water_liquid_NTP)
        f_m_ana = conc_per_mass_np(m_, *dist_par)
        g_m_ana = m_ * f_m_ana
        g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0
        
        no_rows = 5
        fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
        # ax.loglog(radii, xis, "x")
        # ax.loglog(bins_mid[:51], H, "x-")
        # ax.vlines(bins_rad, xis.min(), xis.max(), linewidth=0.5, linestyle="dashed")
        ax = axes[0]
        ax.plot(bins_mass_center_exact, f_m_num_sampled, "x")
        # ax.plot(bins_mass_center_lin_my, f_m_num_avg_my, "x")
        ax.plot(m_, f_m_ana_)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mass (kg)")
        ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$")
        ax = axes[1]
        ax.plot(bins_mass_center_exact, g_m_num_sampled, "x")
        ax.plot(m_, g_m_ana_)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mass (kg)")
        ax.set_ylabel(r"$g_m$ $\mathrm{(m^{-3})}$")
        # ax.loglog(bins_mid, f_m_num/50, "x")
        # ax.loglog(m_, f_m_ana)
        # ax.set_xlim(m_min*9, m_max/9)
        import matplotlib
        ax = axes[2]
        ax.plot(bins_rad_center_exact, g_ln_r_num_sampled*1000.0, "x")
        ax.plot(R_, g_ln_r_ana_)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("radius $\mathrm{(\mu m)}$")
        ax.set_ylabel(r"$g_{\ln(r)}$ $\mathrm{(g \; m^{-3})}$")
        # ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),18))
        # ax.xaxis.set_ticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
        ax.set_xticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax.yaxis.set_ticks(np.logspace(-11,0,12))
        ax.grid(which="both")
        
        # my binning method
        ax = axes[3]
        ax.plot(bins_mass_centers_my[4], f_m_num_avg_my, "x")
        # ax.plot(bins_mass_centers_my[3], f_m_num_avg_my[1:-1], "x")
        # ax.plot(bins_mass_centers_my[3], f_m_num_avg_my, "x")
        ax.plot(m_, f_m_ana_)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("mass (kg)")
        ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$  [my lin fit]")
        ax.grid()
        
        ax = axes[4]
        for n in range(4):
            ax.plot(n*np.ones_like(moments_sampled[n]),
                    moments_sampled[n]/moments_an[n], "o")
        ax.errorbar(np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
                    fmt = "x" , c = "k", markersize = 20.0, linewidth =5.0,
                    capsize=10, elinewidth=5, markeredgewidth=2,
                    zorder=99)
        ax.plot(np.arange(4), np.ones_like(np.arange(4)))
        ax.xaxis.set_ticks([0,1,2,3])
        ax.set_xlabel("$k$")
        # ax.set_ylabel(r"($k$-th moment of $f_m$)/(analytic value)")
        ax.set_ylabel(r"$\lambda_k / \lambda_{k,analytic}$")
        
        for ax in axes[:2]:
            ax.grid()
        
        fig.tight_layout()
        fig.savefig(ensemble_dir + f"fm_gm_glnR_moments_{sample_mode}_no_sims_{no_sims}_no_bins_{no_bins}.png")
    
### DEVIATIONS OF SAMPLED DATA
        # kappa = 10
        # folder = f"kappa_{kappa}/"
        # ensemble_dir = save_directory + folder
        
        dV, DNC0, LWC0, r_critmin, kappa, eta, no_sims00, start_seed = \
            tuple(np.load(ensemble_dir + "ensemble_parameters.npy"))
        LWC0_over_DNC0 = LWC0 / DNC0
        DNC0_over_LWC0 = DNC0 / LWC0
        start_seed = int(start_seed)
        kappa = int(kappa)
        
        no_rows = 4
        fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows), sharex=True)
        
        ax_titles = ["lin", "log", "COM", "exact"]
        
        for n in range(no_rows):
            ax = axes[n]
            f_m_ana = conc_per_mass_np(bins_mass_centers[n], *dist_par)
            ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, "x")
            # ax.plot(bins_mass_width, (f_m_num_sampled-f_m_ana)/f_m_ana, "x")
            ax.set_xscale("log")
            ax.set_ylabel(r"$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ ")
            ax.set_title(ax_titles[n])
        # axes[3].set_xlabel("bin width $\Delta \hat{m}$ (kg)")
        axes[3].set_xlabel("mass (kg)")
        
        for ax in axes:
            ax.grid()
        
        fig.tight_layout()
        fig.savefig(ensemble_dir + f"Deviations_fm_sampled_data_{sample_mode}_no_sims_{no_sims}_no_bins_{no_bins}.png")
        
        #m_min = masses_sampled.min()*0.999
        #m_max = masses_sampled.max()*1.001
        #m_ = np.logspace(np.log10(m_min*0.98), np.log10(m_max*1.1), 1000)
        # m_ = bins_mass_center
        
        # #m_ = np.logspace(np.log10(bins_mass_center[0]), np.log10(bins_mass_center[-1]), 1000)
        # R_ = compute_radius_from_mass(m_*1.0E18, c.mass_density_water_liquid_NTP)
        # f_m_ana = conc_per_mass_np(m_, DNC0, DNC0_over_LWC0)
        # g_m_ana = m_ * f_m_ana
        # g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0
        
        # analog to Wang 2007 and Unterstr 2017
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
        
        ###########################################
    
### PLOTTING STATISTICAL ANALYSIS OVER no_sim runs

        ### fm_gm_g_ln_r with errorbars
        bins_mass_center_exact = bins_mass_centers[3]
        bins_rad_center_exact = bins_rad_centers[3]
        
        m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
        R_ = compute_radius_from_mass(m_*1.0E18, c.mass_density_water_liquid_NTP)
        f_m_ana = conc_per_mass_np(m_, *dist_par)
        g_m_ana = m_ * f_m_ana
        g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0
        
        no_rows = 5
        fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
        # ax.loglog(radii, xis, "x")
        # ax.loglog(bins_mid[:51], H, "x-")
        # ax.vlines(bins_rad, xis.min(), xis.max(), linewidth=0.5, linestyle="dashed")
        ax = axes[0]
        ### add errorbars here
        ax.errorbar(bins_mass_center_exact,
                    # bins_mass_width,
                    f_m_num_avg,
                    f_m_num_std,
                    fmt = "x" ,
                    # c = "k",
                    # c = "lightblue",
                    markersize = 5.0,
                    linewidth = 2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    zorder=99)
        # ax.plot(bins_mass_center_exact, f_m_num_sampled, "x")
        # ax.plot(bins_mass_center_lin_my, f_m_num_avg_my, "x")
        ax.plot(m_, f_m_ana_)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks(np.logspace(6,21,16))
        ax.set_ylim((f_m_ana_[-1],1.0E21))
        # ax.set_ylim((1.0E6,1.0E21))
        ax.set_xlabel("mass (kg) [exact centers]")
        ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$")
        ax = axes[1]
        ax.errorbar(bins_mass_center_exact,
                    # bins_mass_width,
                    g_m_num_avg,
                    g_m_num_std,
                    fmt = "x" ,
                    # c = "k",
                    # c = "lightblue",
                    markersize = 5.0,
                    linewidth = 2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    zorder=99)
        # ax.plot(bins_mass_center_exact, g_m_num_sampled, "x")
        ax.plot(m_, g_m_ana_)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks(np.logspace(-4,8,13))
        # ax.set_ylim((1.0E-4,3.0E8))
        ax.set_ylim((g_m_ana_[-1],3.0E8))
        ax.set_xlabel("mass (kg) [exact centers]")
        ax.set_ylabel(r"$g_m$ $\mathrm{(m^{-3})}$")
        # ax.loglog(bins_mid, f_m_num/50, "x")
        # ax.loglog(m_, f_m_ana)
        # ax.set_xlim(m_min*9, m_max/9)
        # import matplotlib
        ax = axes[2]
        ax.errorbar(bins_rad_center_exact,
                    # bins_mass_width,
                    g_ln_r_num_avg*1000,
                    g_ln_r_num_std*1000,
                    fmt = "x" ,
                    # c = "k",
                    # c = "lightblue",
                    markersize = 5.0,
                    linewidth = 2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    zorder=99)
        # ax.plot(bins_rad_center_exact, g_ln_r_num_sampled*1000.0, "x")
        ax.plot(R_, g_ln_r_ana_)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("radius $\mathrm{(\mu m)}$ [exact centers]")
        ax.set_ylabel(r"$g_{\ln(r)}$ $\mathrm{(g \; m^{-3})}$")
        # ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),18))
        # ax.xaxis.set_ticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
        ax.set_xticks([0.6,1.0,2.0,5.0,10.0,20.0,30.0])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        # ax.get_xaxis().get_major_formatter().labelOnlyBase = False
        ax.set_yticks(np.logspace(-11,0,12))
        ax.set_ylim((1.0E-11,5.0))
        ax.grid(which="both")
        
        # my binning method
        ax = axes[3]
        ax.errorbar(bins_mass_centers_my[4],
                    # bins_mass_width,
                    f_m_num_avg_my,
                    f_m_num_std_my,
                    fmt = "x" ,
                    # c = "k",
                    # c = "lightblue",
                    markersize = 5.0,
                    linewidth = 2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    zorder=99)
        # ax.plot(bins_mass_centers_my[4], f_m_num_avg_my, "x")
        # ax.plot(bins_mass_centers_my[3], f_m_num_avg_my[1:-1], "x")
        # ax.plot(bins_mass_centers_my[3], f_m_num_avg_my, "x")
        ax.plot(m_, f_m_ana_)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_yticks(np.logspace(6,21,16))
        # ax.set_ylim((1.0E6,1.0E21))
        ax.set_ylim((f_m_ana_[-1],1.0E21))
        ax.set_xlabel("mass (kg) [my lin fit centers]")
        ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$  [my lin fit]")
        ax.grid()
        
        ax = axes[4]
        for n in range(4):
            ax.plot(n*np.ones_like(moments_sampled[n]),
                    moments_sampled[n]/moments_an[n], "o")
        ax.errorbar(np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
                    fmt = "x" , c = "k", markersize = 20.0, linewidth =5.0,
                    capsize=10, elinewidth=5, markeredgewidth=2,
                    zorder=99)
        ax.plot(np.arange(4), np.ones_like(np.arange(4)))
        ax.xaxis.set_ticks([0,1,2,3])
        ax.set_xlabel("$k$")
        # ax.set_ylabel(r"($k$-th moment of $f_m$)/(analytic value)")
        ax.set_ylabel(r"$\lambda_k / \lambda_{k,\mathrm{analytic}}$")
        
        for ax in axes[:2]:
            ax.grid()
        
        fig.tight_layout()
        fig.savefig(ensemble_dir + f"fm_gm_glnR_moments_errorbars_{sample_mode}_no_sims_{no_sims}_no_bins_{no_bins}.png")
    
    
        no_rows = 4
        fig, axes = plt.subplots(nrows=no_rows, figsize=(8,4*no_rows), sharex=True)
        
        ax_titles = ["lin", "log", "COM", "exact"]
        
        ### DEVIATIONS WITH ERRORBARS
    
        # bins_mass_centers = bins_mass_centers_my
        # f_m_num_avg = f_m_num_avg_my
        # f_m_num_std = f_m_num_std_my
        
        for n in range(no_rows):
            ax = axes[n]
            f_m_ana = conc_per_mass_np(bins_mass_centers[n], *dist_par)
            # ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, "x")
            ax.errorbar(bins_mass_centers[n],
                        # bins_mass_width,
                        (f_m_num_avg-f_m_ana)/f_m_ana,
                        (f_m_num_std)/f_m_ana,
                        fmt = "x" ,
                        c = "k",
                        # c = "lightblue",
                        markersize = 5.0,
                        linewidth =2.0,
                        capsize=3, elinewidth=2, markeredgewidth=1,
                        zorder=99)
            ax.set_xscale("log")
            ax.set_ylabel(r"$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ ")
            ax.set_title(ax_titles[n])
        # axes[3].set_xlabel("bin width $\Delta \hat{m}$ (kg)")
        axes[3].set_xlabel("mass (kg)")
        
        for ax in axes:
            ax.grid()
        
        fig.tight_layout()
        fig.savefig(ensemble_dir\
            + f"Deviations_fm_errorbars_sepa_plots_{sample_mode}_no_sims_{no_sims}_no_bins_{no_bins}.png")

### PLOT ALL IN ONE
        no_rows = 2
        fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows), sharex=True)
        
        last_ind = 0
        # frac = 1.0
        frac = f_m_counts[0] / no_sims
        count_frac_limit = 0.1
        while frac > count_frac_limit and last_ind < len(f_m_counts)-2:
            last_ind += 1
            frac = f_m_counts[last_ind] / no_sims
        
        # exclude_ind_last = 3
        # last_ind = len(bins_mass_width)-exclude_ind_last
        
        # last_ind = len(bins_mass_centers[0])-16
        
        ax_titles = ["lin", "log", "COM", "exact"]
        
        # ax = axes
        ax = axes[0]
        for n in range(3):
            # ax = axes[n]
            f_m_ana = conc_per_mass_np(bins_mass_centers[n], *dist_par)
            # ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, "x")
            # ax.errorbar(bins_mass_width,
            #             100*((f_m_num_avg-f_m_ana)/f_m_ana)[0:3],
            #             100*(f_m_num_std/f_m_ana)[0:3],
            #             # 100*(f_m_num_avg[0:-3]-f_m_ana[0:-3])/f_m_ana[0:-3],
            #             # 100*f_m_num_std[0:-3]/f_m_ana[0:-3],
            #             fmt = "x" ,
            #             # c = "k",
            #             # c = "lightblue",
            #             markersize = 10.0,
            #             linewidth =2.0,
            #             capsize=3, elinewidth=2, markeredgewidth=1,
            #             zorder=99)
            ax.errorbar(bins_mass_centers[n][:last_ind],
                        # bins_mass_width[:last_ind],
                        100*(f_m_num_avg[:last_ind]-f_m_ana[:last_ind])\
                        /f_m_ana[:last_ind],
                        100*(f_m_num_std[:last_ind])/f_m_ana[:last_ind],
                        fmt = "x" ,
                        # c = "k",
                        # c = "lightblue",
                        markersize = 10.0,
                        linewidth =2.0,
                        capsize=3, elinewidth=2, markeredgewidth=1,
                        label=ax_titles[n],
                        zorder=99)
        ax.legend()
        ax.set_ylabel(r"$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ (%)")
        ax.set_xscale("log")
        # ax.set_yscale("symlog")
        # TT1 = np.array([-5,-4,-3,-2,-1,-0.5,-0.2,-0.1])
        # TT2 = np.array([-0.6,-0.2,-0.1])
        # TT1 = np.concatenate((np.append(TT2,0.0), -TT1) )
        # ax.yaxis.set_ticks(TT1)
        ax.grid()
        
        ax = axes[1]
        f_m_ana = conc_per_mass_np(bins_mass_centers[3], DNC0, DNC0/LWC0)
        # ax.plot(bins_mass_width,
        #         100*(f_m_num_avg[:-exclude_last]-f_m_ana[:-exclude_last])/f_m_ana[:-exclude_last],
        #             # 100*(f_m_num_std)/f_m_ana,
        #             "x" ,
        #             # c = "k",
        #             # c = "lightblue",
        #             markersize = 20.0,
        #             # linewidth =2.0,
        #             # capsize=3, elinewidth=2, markeredgewidth=1,
        #             zorder=99)
        ax.errorbar(bins_mass_centers[3][:last_ind],
                    # bins_mass_width[:last_ind],
                    100*(f_m_num_avg[:last_ind]-f_m_ana[:last_ind])/f_m_ana[:last_ind],
                    100*(f_m_num_std[:last_ind])/f_m_ana[:last_ind],
                    fmt = "x" ,
                    # c = "k",
                    # c = "lightblue",
                    markersize = 10.0,
                    linewidth =2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    label=ax_titles[3],
                    zorder=99)
        ax.set_ylabel(r"$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ (%)")
        # ax.set_xlabel(r"mass $\tilde{m}$ (kg)")
        ax.set_xlabel(r"mass $m$ (kg)")
        ax.legend()
        # ax.set_xscale("log")
        # ax.set_yscale("symlog")
        # TT1 = np.array([-0.08,-0.04,-0.02,-0.01])
        # TT1 = np.array([-0.08,-0.04,-0.02,-0.01,-0.005])
        # TT2 = np.array([-0.6,-0.2,-0.1])
        # TT1 = np.concatenate((np.append(TT1,0.0), -TT1) )
        # ax.yaxis.set_ticks(100*TT1)
        # ax.set_ylim([-10.0,10.0])
        ax.set_xscale("log")
        
        # # ax.set_yscale("symlog")
        # # TT1 = np.array([-0.08,-0.04,-0.02,-0.01])
        # TT1 = np.array([-0.08,-0.04,-0.02,-0.01,-0.005])
        # # TT2 = np.array([-0.6,-0.2,-0.1])
        # TT1 = np.concatenate((np.append(TT1,0.0), -TT1) )
        # ax.yaxis.set_ticks(100*TT1)
        # ax.set_ylim([-10.0,10.0])
        
        ax.grid()
        # for ax in axes:
        #     ax.grid()
        
        fig.suptitle(
            f"kappa={kappa}, eta={eta}, r_critmin={r_critmin}, no_sims={no_sims}",
            y = 0.98)
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.savefig(ensemble_dir + f"Deviations_fm_errorbars_{sample_mode}_no_sims_{no_sims}_no_bins_{no_bins}.png")
    
### PLOTTING MYHISTO BINNING STATISTICAL ANALYSIS OVER no_sim runs
    
        no_rows = 7
        fig, axes = plt.subplots(nrows=no_rows, figsize=(8,4*no_rows), sharex=True)
        
        ax_titles = ["lin", "log", "COM", "exact", "linfit", "qfit", "h_over_g"]
        
        # bins_mass_centers = bins_mass_centers_my
        # f_m_num_avg = f_m_num_avg_my
        # f_m_num_std = f_m_num_std_my
        
        # bins_mass_centers_my[3] *= 1.0/0.99
        # bins_mass_centers_my[3] *= 1.0/0.99
        # bins_mass_centers_my[3] = bins_mass_centers_my3
        # bins_mass_centers_my[3] = \
        #     bins_mass_centers_my3*( 1.0 - 0.035*bins_mass_width_my/bins_mass_width_my[-1])
        
        # f_m_num_avg_my[0] *= 1.1
        
        for n in range(no_rows):
            ax = axes[n]
            f_m_ana = conc_per_mass_np(bins_mass_centers_my[n], *dist_par)
            # ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, "x")
            ax.errorbar(bins_mass_centers_my[n],
                        # bins_mass_width,
                        (f_m_num_avg_my-f_m_ana)/f_m_ana,
                        f_m_num_std_my/f_m_ana,
                        fmt = "x" ,
                        c = "k",
                        # c = "lightblue",
                        markersize = 5.0,
                        linewidth =2.0,
                        capsize=3, elinewidth=2, markeredgewidth=1,
                        zorder=99)
            ax.set_xscale("log")
            ax.set_ylabel(r"$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ ")
            ax.set_title(ax_titles[n])
        # axes[3].set_xlabel("bin width $\Delta \hat{m}$ (kg)")
        axes[-1].set_xlabel("mass (kg)")
        
        for ax in axes:
            ax.grid()
        
        fig.tight_layout()
        fig.savefig(ensemble_dir\
            + f"Deviations_fm_errorbars_myH_sepa_plots_no_sims_{no_sims}_no_bins_{no_bins}.png")
    
### PLOT ALL IN ONE MYHISTO
        no_rows = 2
        fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows), sharex=True)
        
        # last_ind = 0
        # # frac = 1.0
        # frac = f_m_counts[0] / no_sims
        # count_frac_limit = 0.1
        # while frac > count_frac_limit:
        #     last_ind += 1
        #     frac = f_m_counts[last_ind] / no_sims
        
        # exclude_ind_last = 3
        # last_ind = len(bins_mass_width)-exclude_ind_last
        
        last_ind = len(bins_mass_centers_my[0])
        
        ax_titles = ["lin", "log", "COM", "exact"]
        
        # ax = axes
        ax = axes[0]
        for n in range(3):
            # ax = axes[n]
            f_m_ana = conc_per_mass_np(bins_mass_centers_my[n], *dist_par)
            # ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, "x")
            # ax.errorbar(bins_mass_width,
            #             100*((f_m_num_avg-f_m_ana)/f_m_ana)[0:3],
            #             100*(f_m_num_std/f_m_ana)[0:3],
            #             # 100*(f_m_num_avg[0:-3]-f_m_ana[0:-3])/f_m_ana[0:-3],
            #             # 100*f_m_num_std[0:-3]/f_m_ana[0:-3],
            #             fmt = "x" ,
            #             # c = "k",
            #             # c = "lightblue",
            #             markersize = 10.0,
            #             linewidth =2.0,
            #             capsize=3, elinewidth=2, markeredgewidth=1,
            #             zorder=99)
            ax.errorbar(bins_mass_centers_my[n][:last_ind],
                        # bins_mass_width[:last_ind],
                        100*(f_m_num_avg_my[:last_ind]-f_m_ana[:last_ind])\
                        /f_m_ana[:last_ind],
                        100*(f_m_num_std_my[:last_ind])/f_m_ana[:last_ind],
                        fmt = "x" ,
                        # c = "k",
                        # c = "lightblue",
                        markersize = 10.0,
                        linewidth =2.0,
                        capsize=3, elinewidth=2, markeredgewidth=1,
                        label=ax_titles[n],
                        zorder=99)
        ax.legend()
        ax.set_ylabel(r"$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ (%)")
        ax.set_xscale("log")
        # ax.set_yscale("symlog")
        # TT1 = np.array([-5,-4,-3,-2,-1,-0.5,-0.2,-0.1])
        # TT2 = np.array([-0.6,-0.2,-0.1])
        # TT1 = np.concatenate((np.append(TT2,0.0), -TT1) )
        # ax.yaxis.set_ticks(TT1)
        ax.grid()
        
        ax = axes[1]
        f_m_ana = conc_per_mass_np(bins_mass_centers_my[3], *dist_par)
        # ax.plot(bins_mass_width,
        #         100*(f_m_num_avg[:-exclude_last]-f_m_ana[:-exclude_last])/f_m_ana[:-exclude_last],
        #             # 100*(f_m_num_std)/f_m_ana,
        #             "x" ,
        #             # c = "k",
        #             # c = "lightblue",
        #             markersize = 20.0,
        #             # linewidth =2.0,
        #             # capsize=3, elinewidth=2, markeredgewidth=1,
        #             zorder=99)
        ax.errorbar(bins_mass_centers_my[3][:last_ind],
                    # bins_mass_width[:last_ind],
                    100*(f_m_num_avg_my[:last_ind]-f_m_ana[:last_ind])/f_m_ana[:last_ind],
                    100*(f_m_num_std_my[:last_ind])/f_m_ana[:last_ind],
                    fmt = "x" ,
                    # c = "k",
                    # c = "lightblue",
                    markersize = 10.0,
                    linewidth =2.0,
                    capsize=3, elinewidth=2, markeredgewidth=1,
                    label=ax_titles[3],
                    zorder=99)
        ax.set_ylabel(r"$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ (%)")
        # ax.set_xlabel(r"mass $\tilde{m}$ (kg)")
        ax.set_xlabel(r"mass $m$ (kg)")
        ax.legend()
        # ax.set_xscale("log")
        # ax.set_yscale("symlog")
        # TT1 = np.array([-0.08,-0.04,-0.02,-0.01])
        # TT1 = np.array([-0.08,-0.04,-0.02,-0.01,-0.005])
        # TT2 = np.array([-0.6,-0.2,-0.1])
        # TT1 = np.concatenate((np.append(TT1,0.0), -TT1) )
        # ax.yaxis.set_ticks(100*TT1)
        # ax.set_ylim([-10.0,10.0])
        ax.set_xscale("log")
        
        # # ax.set_yscale("symlog")
        # # TT1 = np.array([-0.08,-0.04,-0.02,-0.01])
        # TT1 = np.array([-0.08,-0.04,-0.02,-0.01,-0.005])
        # # TT2 = np.array([-0.6,-0.2,-0.1])
        # TT1 = np.concatenate((np.append(TT1,0.0), -TT1) )
        # ax.yaxis.set_ticks(100*TT1)
        # ax.set_ylim([-10.0,10.0])
        
        ax.grid()
        # for ax in axes:
        #     ax.grid()
        
        fig.suptitle(
            f"kappa={kappa}, eta={eta}, r_critmin={r_critmin}, no_sims={no_sims}",
            y = 0.98)
        
        fig.tight_layout()
        fig.subplots_adjust(top=0.9)
        fig.savefig(ensemble_dir + f"Deviations_fm_errorbars_myH_no_sims_{no_sims}_no_bins_{no_bins}.png")
    
        plt.close("all")

#%% CODE FRAGMENTS

### PLOTTING mu_hat = position where the average bin value of f and the value
# of f(mu_hat) are equal, i.e mu_hat = "bin mass center"
# fig, ax = plt.subplots()
# ax.plot(m_, f_m_ana, "x")
# ax.vlines(bins_mass, conc_per_mass(bins_mass[-1], DNC0, DNC0_over_LWC0),
#           conc_per_mass(bins_mass[0], DNC0, DNC0_over_LWC0), linestyle="dashed")
# ax.set_xscale("log")
# ax.set_yscale("log")

### "BY HAND" SIP GEN METHOD
# seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)

# # LINUX desk
# # path = "/home/jdesk/OneDrive/python/sim_data/test_SIP_ensemble_Unt/"
# path = "/mnt/D/sim_data/test_SIP_ensemble_Unt_new/"

# # path = "/Users/bohrer/OneDrive - bwedu/python/sim_data/test_SIP_ensemble_Unt/"

# folder = f"kappa_{kappa}/"
# path = path + folder

# ensemble_parameters = [dV, DNC0, LWC0, r_critmin, kappa, eta, no_sims, start_seed]

# if not os.path.exists(path):
#     os.makedirs(path)

# for i,seed in enumerate(seed_list):
#     masses, xis, m_low, bins =\
#         generate_SIP_ensemble_SingleSIP_Unt(DNC0, LWC0,
#                                                dV, kappa, eta, r_critmin,
#                                                m_high_over_m_low=1.0E6,
#                                                seed=seed)
#     bins_rad = compute_radius_from_mass(1.0E18*bins,
#                                         c.mass_density_water_liquid_NTP)
#     radii = compute_radius_from_mass(1.0E18*masses,
#                                      c.mass_density_water_liquid_NTP)
#     np.save(path + f"masses_seed_{seed}", masses)
#     np.save(path + f"radii_seed_{seed}", radii)
#     np.save(path + f"xis_seed_{seed}", xis)
    
#     if i == 0:
#         np.save(path + f"bins_mass", bins)
#         np.save(path + f"bins_rad", bins_rad)
#         np.save(path + "ensemble_parameters", ensemble_parameters)
        

### SAMPLED DATA ANALYSIS




# f_m_num = f_m_num[f_m_ind]
# g_m_num = g_m_num[f_m_ind]
# bins_mass_center = bins_mass_center[f_m_ind]
# bins_mass_width = bins_mass_width[f_m_ind]

#m_min = masses_sampled.min()*0.999
#m_max = masses_sampled.max()*1.001
#m_ = np.logspace(np.log10(m_min*0.98), np.log10(m_max*1.1), 1000)

# analog to Wang 2007 and Unterstr 2017
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

###########################################

### "BY HAND" ANALYZE DATA SAMPLED
### DERIVED
# folder = f"kappa_{kappa}/"
# path = save_directory + folder

# dV, DNC0, LWC0, r_critmin, kappa, eta, no_sims00, start_seed = \
#     tuple(np.load(path + "ensemble_parameters.npy"))
# LWC0_over_DNC0 = LWC0 / DNC0
# DNC0_over_LWC0 = DNC0 / LWC0
# start_seed = int(start_seed)
# no_sims00 = int(no_sims00)
# seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)

# ### ANALYSIS START
# masses = []
# xis = []
# radii = []

# # masses_sampled = []
# # radii_sampled = []
# # xis_sampled = []
# moments_sampled = []
# for i,seed in enumerate(seed_list):
#     masses.append(np.load(path + f"masses_seed_{seed}.npy"))
#     xis.append(np.load(path + f"xis_seed_{seed}.npy"))
#     radii.append(np.load(path + f"radii_seed_{seed}.npy"))
#     # masses_sampled.append(masses)
#     # xis_sampled.append(xis)

#     moments = np.zeros(4,dtype=np.float64)
#     moments[0] = xis[i].sum() / dV
#     for n in range(1,4):
#         moments[n] = np.sum(xis[i]*masses[i]**n) / dV
#     moments_sampled.append(moments)

# masses_sampled = np.concatenate(masses)
# radii_sampled = np.concatenate(radii)
# xis_sampled = np.concatenate(xis)

# # moments analysis
# moments_sampled = np.transpose(moments_sampled)
# moments_an = np.zeros(4,dtype=np.float64)
# for n in range(4):
#     moments_an[n] = moments_analytical(n, DNC0, LWC0_over_DNC0)

# for n in range(4):
#     print(n, (np.average(moments_sampled[n])-moments_an[n])/moments_an[n] )

# moments_sampled_avg_norm = np.average(moments_sampled, axis=1) / moments_an
# moments_sampled_std_norm = np.std(moments_sampled, axis=1) / moments_an

# m_min = masses_sampled.min()
# m_max = masses_sampled.max()

# R_min = radii_sampled.min()
# R_max = radii_sampled.max()

# # H1 = np.histogram(radii_sampled, bins_rad)[0]
# # H2 = np.histogram(radii_sampled, bins_rad, weights=xis_sampled)[0]

# # H1 = H1[np.nonzero(H1)[0]]
# # H2 = H2[np.nonzero(H1)[0]]

# # H = H2 / H1

# bins_mass = np.load(path + "bins_mass.npy")
# bins_rad = np.load(path + "bins_rad.npy")

# f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
# f_m_ind = np.nonzero(f_m_counts)[0]
# bins_mass_ind = np.append(f_m_ind, f_m_ind[-1]+1)

# bins_mass = bins_mass[bins_mass_ind]
# bins_rad = bins_rad[bins_mass_ind]
# bins_rad_log = np.log(bins_rad)
# bins_mass_width = (bins_mass[1:]-bins_mass[:-1])
# bins_rad_width = (bins_rad[1:]-bins_rad[:-1])
# bins_rad_width_log = (bins_rad_log[1:]-bins_rad_log[:-1])

# ### approximate the functions f_m, f_lnR = 3*m*f_m, g_lnR=3*m^2*f_m
# # estimate f_m(m) by binning:
# # DNC_i = f_m(m_i) * dm_i = droplet number conc in bin i with size dm_i
# f_m_num_sampled = np.histogram(masses_sampled,bins_mass,weights=xis_sampled)[0]
# g_m_num_sampled = np.histogram(masses_sampled,bins_mass,
#                                weights=xis_sampled*masses_sampled)[0]

# f_m_num_sampled = f_m_num_sampled / (bins_mass_width * dV * no_sims)
# g_m_num_sampled = g_m_num_sampled / (bins_mass_width * dV * no_sims)

# # build g_ln_r = 3*m*g_m DIRECTLY from data
# g_ln_r_num_sampled = np.histogram(radii_sampled,
#                                   bins_rad,
#                                   weights=xis_sampled*masses_sampled)[0]
# g_ln_r_num_sampled = g_ln_r_num_sampled \
#                      / (bins_rad_width_log * dV * no_sims)
# # g_ln_r_num_derived = 3 * bins_mass_center * g_m_num * 1000.0

# # define centers on lin scale
# bins_mass_center_lin = 0.5 * (bins_mass[:-1] + bins_mass[1:])
# bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])

# # define centers on the logarithmic scale
# bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
# bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))

# # define the center of mass for each bin and set it as the "bin center"
# bins_mass_center_COM = g_m_num_sampled/f_m_num_sampled
# bins_rad_center_COM =\
#     compute_radius_from_mass(bins_mass_center_COM*1.0E18,
#                              c.mass_density_water_liquid_NTP)

# # set the bin "mass centers" at the right spot such that
# # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
# m_avg = LWC0_over_DNC0
# bins_mass_center_exact = bins_mass[f_m_ind] + m_avg * np.log(bins_mass_width\
#       / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))
# bins_rad_center_exact =\
#     compute_radius_from_mass(bins_mass_center_exact*1.0E18,
#                              c.mass_density_water_liquid_NTP)

# bins_mass_centers = np.array((bins_mass_center_lin,
#                               bins_mass_center_log,
#                               bins_mass_center_COM,
#                               bins_mass_center_exact))

# m_ = np.logspace(np.log10(bins_mass[0]), np.log10(bins_mass[-1]), 1000)
# R_ = compute_radius_from_mass(m_*1.0E18, c.mass_density_water_liquid_NTP)
# f_m_ana = conc_per_mass_np(m_, DNC0, DNC0_over_LWC0)
# g_m_ana = m_ * f_m_ana
# g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0

# f_m_num = f_m_num[f_m_ind]
# g_m_num = g_m_num[f_m_ind]
# bins_mass_center = bins_mass_center[f_m_ind]
# bins_mass_width = bins_mass_width[f_m_ind]

#m_min = masses_sampled.min()*0.999
#m_max = masses_sampled.max()*1.001
#m_ = np.logspace(np.log10(m_min*0.98), np.log10(m_max*1.1), 1000)

# analog to Wang 2007 and Unterstr 2017
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

###########################################

# no_rows = 4
# fig, axes = plt.subplots(nrows=no_rows, figsize=(6,4*no_rows))
# # ax.loglog(radii, xis, "x")
# # ax.loglog(bins_mid[:51], H, "x-")
# # ax.vlines(bins_rad, xis.min(), xis.max(), linewidth=0.5, linestyle="dashed")
# ax = axes[0]
# ax.plot(bins_mass_center, (f_m_num-f_m_ana)/f_m_ana, "x")
# #ax.plot(m_, f_m_ana)
# #ax.set_xscale("log")
# #ax.set_yscale("log")
# ax = axes[1]
# ax.plot(bins_mass_center, (g_m_num-g_m_ana)/g_m_ana, "x")
# #ax.plot(m_, g_m_ana)
# #ax.set_xscale("log")
# #ax.set_yscale("log")
# # ax.loglog(bins_mid, f_m_num/50, "x")
# # ax.loglog(m_, f_m_ana)
# # ax.set_xlim(m_min*9, m_max/9)
# ax = axes[2]
# ax.plot(bins_rad_center, (g_ln_r_num-g_ln_r_ana)/g_ln_r_ana, "x")
# #ax.plot(R_, g_ln_r_ana)
# #ax.set_xscale("log")
# #ax.set_yscale("log")
# #ax.xaxis.set_ticks(np.logspace(np.log10(0.6), np.log10(30),17))
# #ax.yaxis.set_ticks(np.logspace(-10,0,11))
# #ax.grid(which="both")


# ax = axes[3]
# for n in range(4):
#     ax.plot( n*np.ones_like(moments_sampled[n]) , moments_sampled[n]/moments_an[n], "o")
# ax.errorbar( np.arange(4), moments_sampled_avg_norm, moments_sampled_std_norm,
#             fmt = "x" , c = "k", markersize = 20.0, linewidth =5.0, zorder=99)
# ax.plot(np.arange(4), np.ones_like(np.arange(4)))

# for ax in axes[:3]:
#     ax.grid()

#################################

#fig, ax = plt.subplots(nrows=1, figsize=(8,8))
#ax.plot(bins_rad_center, g_ln_r_num, "x")
#ax.plot(R_, g_ln_r_ana)
#ax.set_xscale("log")
#ax.set_yscale("log")
#ax.grid()
#ax.yaxis.set_ticks(np.logspace(-8,0,9))

#################################
