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

### FUNCTION DEF: CREATE SIP ENSEMBLE AND ANALYSIS

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

### GENERATE AND SAVE SIP ENSEMBLES SINGLE SIP UNTERSTRASSER
# r_critmin in mu
def generate_and_save_SIP_ensembles_SingleSIP_prob(
        DNC0, LWC0, dV, kappa, eta, r_critmin, no_sims,
        m_high_over_m_low, start_seed, save_directory):

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

    folder = f"kappa_{kappa}/"
    path = save_directory + folder
    
    ensemble_parameters = [dV, DNC0, LWC0, r_critmin,
                           kappa, eta, no_sims, start_seed]
    
    if not os.path.exists(path):
        os.makedirs(path)
    
    for i,seed in enumerate(seed_list):
        masses, xis, m_low, bins =\
            generate_SIP_ensemble_SingleSIP_Unt(
                DNC0, LWC0, dV, kappa, eta, r_critmin,
                m_high_over_m_low, seed)
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

### ANALYZE EXPO SIP ENSEMBLE DATA FROM DATA STORED IN FILES

def analyze_ensemble_data_sampled(kappa, no_sims, save_directory):
    ### DERIVED
    folder = f"kappa_{kappa}/"
    path = save_directory + folder
    
    dV, DNC0, LWC0, r_critmin, kappa, eta, no_sims00, start_seed = \
        tuple(np.load(path + "ensemble_parameters.npy"))
    LWC0_over_DNC0 = LWC0 / DNC0
    DNC0_over_LWC0 = DNC0 / LWC0
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
        masses.append(np.load(path + f"masses_seed_{seed}.npy"))
        xis.append(np.load(path + f"xis_seed_{seed}.npy"))
        radii.append(np.load(path + f"radii_seed_{seed}.npy"))
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
    moments_sampled_std_norm = np.std(moments_sampled, axis=1) / moments_an
    
    m_min = masses_sampled.min()
    m_max = masses_sampled.max()
    
    R_min = radii_sampled.min()
    R_max = radii_sampled.max()
    
    # H1 = np.histogram(radii_sampled, bins_rad)[0]
    # H2 = np.histogram(radii_sampled, bins_rad, weights=xis_sampled)[0]
    
    # H1 = H1[np.nonzero(H1)[0]]
    # H2 = H2[np.nonzero(H1)[0]]
    
    # H = H2 / H1
    
    bins_mass = np.load(path + "bins_mass.npy")
    bins_rad = np.load(path + "bins_rad.npy")
    
    f_m_counts = np.histogram(masses_sampled,bins_mass)[0]
    f_m_ind = np.nonzero(f_m_counts)[0]

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
    bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
    bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
    
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
    f_m_ana = conc_per_mass_np(m_, DNC0, DNC0_over_LWC0)
    g_m_ana = m_ * f_m_ana
    g_ln_r_ana = 3 * m_ * g_m_ana * 1000.0

    return bins_mass, bins_rad, bins_rad_log, \
           bins_mass_width, bins_rad_width, bins_rad_width_log, \
           bins_mass_centers, bins_rad_centers, \
           masses, xis, radii, f_m_counts, f_m_ind,\
           f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled,\
           m_, R_, f_m_ana, g_m_ana, g_ln_r_ana, \
           m_min, m_max, R_min, R_max, no_SIPs_avg, \
           moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
           moments_an


#%% GENERATE SIP ENSEMBLE AND STORE TO FILES
r_critmin = 0.6 # mu
# m_low = 1.0E-18 * compute_mass_from_radius(r_critmin,
#                                            c.mass_density_water_liquid_NTP)

dV = 1.0
kappa = 10
eta = 1.0E-9

DNC0 = 2.97E8 # 1/m^3
#DNC0 = 3.0E8 # 1/m^3
LWC0 = 1.0E-3 # kg/m^3

# LWC0_inv = 1.0 / LWC0
# DNC0_over_LWC0 = DNC0 / LWC0

# print(r_critmin, m_low)
# print(conc_per_mass(0.0, DNC0, DNC0_over_LWC0))
# print(conc_per_mass(m_low, DNC0, DNC0_over_LWC0))

#no_bins = 10

no_sims = 1000
start_seed = 3711

# LINUX desk
# path = "/home/jdesk/OneDrive/python/sim_data/test_SIP_ensemble_Unt/"
save_directory = "/mnt/D/sim_data/test_SIP_ensemble_Unt_new2/"
# path =\
#     "/Users/bohrer/OneDrive - bwedu/python/sim_data/test_SIP_ensemble_Unt/"

m_high_over_m_low = 1.0E6

generate_and_save_SIP_ensembles_SingleSIP_prob(
    DNC0, LWC0, dV, kappa, eta, r_critmin, no_sims,
    m_high_over_m_low, start_seed, save_directory)

#%% DATA ANALYIS
### SAMPLED DATA

### SET VALUES/PATHS
# LINUX desk
# path = "/home/jdesk/OneDrive/python/sim_data/test_SIP_ensemble_Unt/"
save_directory = "/mnt/D/sim_data/test_SIP_ensemble_Unt_new2/"
# path =\
#     "/Users/bohrer/OneDrive - bwedu/python/sim_data/test_SIP_ensemble_Unt/"

# set manually for analysis
kappa = 10
no_sims = 5000

folder = f"kappa_{kappa}/"
path = save_directory + folder

dV, DNC0, LWC0, r_critmin, kappa00, eta, no_sims00, start_seed = \
    tuple(np.load(path + "ensemble_parameters.npy"))
start_seed = int(start_seed)

bins_mass, bins_rad, bins_rad_log, \
bins_mass_width, bins_rad_width, bins_rad_width_log, \
bins_mass_centers, bins_rad_centers, \
masses, xis, radii, f_m_counts, f_m_ind,\
f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled, \
m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, \
m_min, m_max, R_min, R_max, no_SIPs_avg, \
moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
moments_an = \
    analyze_ensemble_data_sampled(kappa, no_sims, save_directory)

### STATISTICAL ANALYSIS OVER no_sim runs
# get f(m_i) curve for each "run" with same bins for all
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

# masses is a list of [masses0, masses1, ..., masses_no_sims]
# where masses_i = array of masses of a spec. SIP ensemble
def generate_myHisto_SIP_ensemble_np(masses, xis, m_min, m_max,
                                     dV, DNC0, LWC0,
                                     no_bins, no_sims,
                                     bin_mode, center_mode):
    # g_m_num = []
    # g_ln_r_num = []
    if bin_mode == 1:
        if center_mode == 1:
            bin_factor = (m_max/m_min)**(1.0/no_bins)
            bin_log_dist = np.log(bin_factor)
            bin_log_dist_half = 0.5 * bin_log_dist
            # the factor 0.99 is for numerical stability: to be sure
            # that m_min does not contribute to a bin smaller than the
            # 0-th bin
            # m_0 = m_min / np.sqrt(bin_factor)

            bins_mass = np.zeros(no_bins+1,dtype=np.float64)
            bins_mass[0] = m_min
            for bin_n in range(1,no_bins+1):
                bins_mass[bin_n] = bins_mass[bin_n-1] * bin_factor
            # the factor 1.01 is for numerical stability: to be sure
            # that m_max does not contribute to a bin larger than the
            # last bin
            bins_mass[0] *= 0.9999
            bins_mass[-1] *= 1.0001
            bins_mass_log = np.log(bins_mass)

            bins_mass_width = bins_mass[1:]-bins_mass[:-1]

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

    f_m_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )
    g_m_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )
    # g_ln_r_num = np.zeros( (no_sims,no_bins), dtype=np.float64 )
    for i,mass in enumerate(masses):
        histo = np.zeros(no_bins+2, dtype=np.float64)
        histo_g = np.zeros(no_bins+2, dtype=np.float64)
        mass_log = np.log(mass)
        for n,m_ in enumerate(mass):
            xi = xis[i][n]
            bin_n = np.nonzero(np.histogram(m_, bins=bins_mass)[0])[0][0]
            # bin_n = np.nonzero(np.histogram(m_, bins=bins_mass)[0])[0][0]
            # print(bin_n)
            log_dist = mass_log[n] - bins_mass_log[bin_n]

            # +1 because we have overrun bins left and right
            bin_n += 1

            if log_dist < bin_log_dist_half:
                s = 0.5 + log_dist/bin_log_dist
                # print(n,s,"left")
                histo[bin_n] += s*xi
                histo[bin_n-1] += (1.0-s)*xi
                histo_g[bin_n] += s*xi*m_
                histo_g[bin_n-1] += (1.0-s)*xi*m_
            else:
                s = 1.5 - log_dist/bin_log_dist
                # print(n,s,"right")
                histo[bin_n] += s*xi
                histo[bin_n+1] += (1.0-s)*xi
                histo_g[bin_n] += s*xi*m_
                histo_g[bin_n+1] += (1.0-s)*xi*m_

        f_m_num[i] = histo[1:-1] / (bins_mass_width * dV)
        g_m_num[i] = histo_g[1:-1] / (bins_mass_width * dV)
    # f_m_num_avg = f_m_num.mean()
    # f_m_num_std = f_m_num.std() / np.sqrt(no_sims)
    # g_m_num_avg = g_m_num.mean()
    # g_m_num_std = g_m_num.std() / np.sqrt(no_sims)
    f_m_num_avg = np.average(f_m_num, axis=0)
    f_m_num_std = np.std(f_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    g_m_num_avg = np.average(g_m_num, axis=0)
    g_m_num_std = np.std(g_m_num, axis=0, ddof=1) / np.sqrt(no_sims)
    # define centers on lin scale
    bins_mass_center_lin = 0.5 * (bins_mass[:-1] + bins_mass[1:])
    # bins_rad_center_lin = 0.5 * (bins_rad[:-1] + bins_rad[1:])
    
    # define centers on the logarithmic scale
    bins_mass_center_log = bins_mass[:-1] * np.sqrt(bin_factor)
    # bins_mass_center_log = bins_mass[:-1] * 10**(1.0/(2.0*kappa))
    # bins_rad_center_log = bins_rad[:-1] * 10**(1.0/(2.0*kappa))
    
    # define the center of mass for each bin and set it as the "bin center"
    bins_mass_center_COM = g_m_num_avg / f_m_num_avg
    # bins_rad_center_COM =\
    #     compute_radius_from_mass(bins_mass_center_COM*1.0E18,
    #                              c.mass_density_water_liquid_NTP)
    
    # set the bin "mass centers" at the right spot such that
    # f_avg_i in bin in = f(mm_i), where mm_i is the "mass center"
    m_avg = LWC0_over_DNC0
    bins_mass_center_exact = bins_mass[:-1]\
                             + m_avg * np.log(bins_mass_width\
          / (m_avg * (1-np.exp(-bins_mass_width/m_avg))))
    # bins_rad_center_exact =\
    #     compute_radius_from_mass(bins_mass_center_exact*1.0E18,
    #                              c.mass_density_water_liquid_NTP)
    bins_mass_centers = np.array((bins_mass_center_lin,
                                  bins_mass_center_log,
                                  bins_mass_center_COM,
                                  bins_mass_center_exact))
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
           bins_mass, bins_mass_width, bins_mass_centers
# generate_myHisto_SIP_ensemble = njit()(generate_myHisto_SIP_ensemble_np)

no_bins = 50
# no_sims =
bin_mode = 1
center_mode = 1

f_m_num_avg_my, f_m_num_std_my, g_m_num_avg_my, g_m_num_std_my, \
bins_mass_my, bins_mass_width_my, bins_mass_centers_my =\
    generate_myHisto_SIP_ensemble_np(masses, xis, m_min, m_max, dV, DNC0, LWC0,
                                     no_bins, no_sims, bin_mode, center_mode)



#%% PLOTTING
### SAMPLED DATA
### IN WORK: add errorbars to all...

bins_mass_center_exact = bins_mass_centers[3]
bins_rad_center_exact = bins_rad_centers[3]

no_rows = 5
fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows))
# ax.loglog(radii, xis, "x")
# ax.loglog(bins_mid[:51], H, "x-")
# ax.vlines(bins_rad, xis.min(), xis.max(), linewidth=0.5, linestyle="dashed")
ax = axes[0]
ax.plot(bins_mass_center_exact, f_m_num_sampled, "x")
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
ax.plot(bins_mass_centers_my[3], f_m_num_avg_my, "x")
ax.plot(m_, f_m_ana_)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("mass (kg)")
ax.set_ylabel(r"$f_m$ $\mathrm{(kg^{-1} \, m^{-3})}$")
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
fig.savefig(path + f"fm_gm_glnR_moments_no_sims_{no_sims}.png")

#%%

### DEVIATIONS OF SAMPLED DATA
# kappa = 10
# folder = f"kappa_{kappa}/"
# path = save_directory + folder

dV, DNC0, LWC0, r_critmin, kappa, eta, no_sims00, start_seed = \
    tuple(np.load(path + "ensemble_parameters.npy"))
LWC0_over_DNC0 = LWC0 / DNC0
DNC0_over_LWC0 = DNC0 / LWC0
start_seed = int(start_seed)
kappa = int(kappa)

no_rows = 4
fig, axes = plt.subplots(nrows=no_rows, figsize=(10,5*no_rows), sharex=True)

ax_titles = ["lin", "log", "COM", "exact"]

for n in range(no_rows):
    ax = axes[n]
    f_m_ana = conc_per_mass_np(bins_mass_centers[n], DNC0, DNC0_over_LWC0)
    # ax.plot(bins_mass_centers[n], (f_m_num_sampled-f_m_ana)/f_m_ana, "x")
    ax.plot(bins_mass_width, (f_m_num_sampled-f_m_ana)/f_m_ana, "x")
    ax.set_xscale("log")
    ax.set_ylabel(r"$(f_{m,num}-f_{m}(\tilde{m}))/f_{m}(\tilde{m})$ ")
    ax.set_title(ax_titles[n])
axes[3].set_xlabel("bin width $\Delta \hat{m}$ (kg)")

for ax in axes:
    ax.grid()

fig.tight_layout()
fig.savefig(path + f"Deviations_fm_sampled_data_no_sims_{no_sims}.png")

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

#%%

### PLOTTING STATISTICAL ANALYSIS OVER no_sim runs
no_rows = 4
fig, axes = plt.subplots(nrows=no_rows, figsize=(8,4*no_rows), sharex=True)

ax_titles = ["lin", "log", "COM", "exact"]

###

bins_mass_centers = bins_mass_centers_my
f_m_num_avg = f_m_num_avg_my
f_m_num_std = f_m_num_std_my

for n in range(no_rows):
    ax = axes[n]
    f_m_ana = conc_per_mass_np(bins_mass_centers[n], DNC0, DNC0_over_LWC0)
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
axes[3].set_xlabel("bin width $\Delta \hat{m}$ (kg)")

for ax in axes:
    ax.grid()

fig.tight_layout()
fig.savefig(path + f"Deviations_fm_errorbars_sepa_plots_no_sims_{no_sims}.png")

### PLOT ALL IN ONE
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

last_ind = len(bins_mass_centers[0])

ax_titles = ["lin", "log", "COM", "exact"]

# ax = axes
ax = axes[0]
for n in range(3):
    # ax = axes[n]
    f_m_ana = conc_per_mass_np(bins_mass_centers[n], DNC0, DNC0_over_LWC0)
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
f_m_ana = conc_per_mass_np(bins_mass_centers[3], DNC0, DNC0_over_LWC0)
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
fig.savefig(path + f"Deviations_fm_errorbars_no_sims_{no_sims}.png")


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
