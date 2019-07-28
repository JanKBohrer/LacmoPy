#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:31:53 2019

@author: jdesk
"""

"""
NOTES:
the difference between kernel method
analytic (which gives the same as kernel_grid_m)
AND Ecol_grid_R might be explained by the assumption that
the velocity is held constant during each timestep for Ecol_grid_R,
while it is updated on the fly in the other two methods...
# this should be reduced when reducing the timestep ;P
"""

#%% IMPORTS AND DEFS

import os
import math
import numpy as np
#from numba import njit

#import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick

incl_path = "/home/jdesk/CloudMP"
import sys
if os.path.exists(incl_path):
    sys.path.append(incl_path)
    
import constants as c

from microphysics import compute_radius_from_mass_vec
#from microphysics import compute_radius_from_mass_jit

from box_model import simulate_collisions
from box_model import analyze_sim_data
from box_model import plot_for_given_kappa
from box_model import plot_moments_kappa_var

#from kernel import compute_terminal_velocity_Beard
from kernel import compute_terminal_velocity_Beard_vec
from kernel import generate_and_save_kernel_grid_Long_Bott
from kernel import generate_and_save_E_col_grid_Long_Bott

#from init_SIPs import conc_per_mass_expo
from init_SIPs import generate_and_save_SIP_ensembles_SingleSIP_prob
from init_SIPs import analyze_ensemble_data
#from init_SIPs import generate_myHisto_SIP_ensemble_np
#from init_SIPs import plot_ensemble_data

#%% SET PARAMETERS 

OS = "LinuxDesk"
# OS = "MacOS"

############################################################################################
# SET args for SIP ensemble generation AND Kernel-grid generation
args_gen = [0,0,0,0,0]
#args_gen = [1,1,1,0,0]

act_gen_SIP = bool(args_gen[0])
act_analysis_ensembles = bool(args_gen[1])
# NOTE: plotting needs analyze first (or in buffer)
act_plot_ensembles = bool(args_gen[2])
act_gen_kernel_grid = bool(args_gen[3])
act_gen_Ecol_grid = bool(args_gen[4])

# SET args for simulation
#args_sim = [0,0,0,0]
args_sim = [1,0,0,0]
#args_sim = [0,1,1,1]
#args_sim = [0,0,0,1]
#args_sim = [1,1,1,1]

act_sim = bool(args_sim[0])
act_analysis = bool(args_sim[1])
act_plot = bool(args_sim[2])
act_plot_moments_kappa_var = bool(args_sim[3])

############################################################################################
### SET PARAMETERS FOR SIMULATION OF COLLISION BOX MODEL

kappa_list=[400]
#kappa_list=[5,10,20,40,60,100]
#kappa_list=[5,10,20,40,60,100,200]
#kappa_list=[5,10,20,40,60,100,200,400]
#kappa_list=[60,100,200]
#kappa_list=[5,10,20,40,60,100,200,400,600,800]
#kappa_list=[200,400,600,800]

no_sims = 500
#no_sims = 400
start_seed = 3711

# kernel_name = "Golovin"
kernel_name = "Long_Bott"

#kernel_method = "kernel_grid_m"
#kernel_method = "Ecol_grid_R"
kernel_method = "analytic"

#dt = 1.0
dt = 10.0
# dt = 20.0

# dt_save = 40.0
dt_save = 300.0
# t_end = 200.0
t_end = 3600.0

# NOTE that only "auto_bin" is possible for the analysis of the sim data for now
bin_method_sim_data = "auto_bin"

############################################################################################
### SET GENERAL PARAMETERS

no_bins = 50

mass_density = 1E3 # approx for water
#mass_density = c.mass_density_water_liquid_NTP
#mass_density = c.mass_density_NaCl_dry

if OS == "MacOS":
    sim_data_path = "/Users/bohrer/sim_data/"
elif OS == "LinuxDesk":
    sim_data_path = "/mnt/D/sim_data_unif/col_box_mod/"

############################################################################################
### SET PARAMETERS FOR SIP ENSEMBLES

dist = "expo"
#dist = "lognormal"
gen_method = "SinSIP"

eta = 1.0E-9

eta_threshold = "weak"
#eta_threshold = "fix"

dV = 1.0

## for EXPO dist: set r0 and LWC0 analog to Unterstr.
# -> DNC0 is calc from that
if dist == "expo":
    LWC0 = 1.0E-3 # kg/m^3
    R_mean = 9.3 # in mu
    r_critmin = 0.6 # mu
    m_high_over_m_low = 1.0E6

### for lognormal dist: set DNC0 and mu_R and sigma_R parameters of
# lognormal distr (of the radius) -> see distr. def above
if dist == "lognormal":
    r_critmin = 0.001 # mu
    m_high_over_m_low = 1.0E8
    
    # set DNC0 manually only for lognormal distr. NOT for expo
    DNC0 = 60.0E6 # 1/m^3
    #DNC0 = 2.97E8 # 1/m^3
    #DNC0 = 3.0E8 # 1/m^3
    mu_R = 0.02 # in mu
    sigma_R = 1.4 #  no unit

## PARAMS FOR DATA ANALYSIS OF INITIAL SIP ENSEMBLES

# the "bin_method_init_ensembles" is not necessary anymore
# both methods are applied and plotted by default
# use the same bins as for generation
#bin_method_init_ensembles = "given_bins"
# for auto binning: set number of bins
#bin_method_init_ensembles = "auto_bins"

# only bin_mode = 1 is available for now..
# bin_mode = 1 for bins equal dist on log scale
bin_mode = 1

## for myHisto generation (additional parameters)
spread_mode = 0 # spreading based on lin scale
# spread_mode = 1 # spreading based on log scale
# shift_factor = 1.0
# the artificial bins before the first and after the last one get multiplied
# by this factor (since they only get part of the half of the first/last bin)
overflow_factor = 2.0
# scale factor for the first correction
scale_factor = 1.0
# shift factor for the second correction.
# For shift_factor = 0.5 only half of the effect:
# bin center_new = 0.5 (bin_center_lin_first_corr + shifted bin center)
shift_factor = 0.5

############################################################################################
### SET PARAMETERS FOR KERNEL/ECOL GRID GENERATION AND LOADING

R_low_kernel, R_high_kernel, no_bins_10_kernel = 0.6, 6E3, 200

save_dir_kernel_grid = sim_data_path + f"{dist}/kernel_grid_data/"
save_dir_Ecol_grid = sim_data_path + f"{dist}/Ecol_grid_data/"

############################################################################################
### DERIVED PARAMETERS
# constant converts radius in mu to mass in kg (m = 4/3 pi rho R^3)
c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
c_mass_to_radius = 1.0 / c_radius_to_mass

if eta_threshold == "weak":
    weak_threshold = True
else: weak_threshold = False

if dist == "expo":
    m_mean = c_radius_to_mass * R_mean**3 # in kg
    DNC0 = LWC0 / m_mean # in 1/m^3
    LWC0_over_DNC0 = LWC0 / DNC0
    DNC0_over_LWC0 = DNC0 / LWC0
    print("dist = expo", f"DNC0 = {DNC0:.3e}", "LWC0 =", LWC0, "m_mean = ", m_mean)
    dist_par = (DNC0, DNC0_over_LWC0)

elif dist =="lognormal":
    mu_R_log = np.log(mu_R)
    sigma_R_log = np.log(sigma_R)
    
    # derive parameters of lognormal distribution of mass f_m(m)
    # assuming mu_R in mu and density in kg/m^3
    mu_m_log = 3.0 * mu_R_log + np.log(1.0E-18 * c.four_pi_over_three * mass_density)
    sigma_m_log = 3.0 * sigma_R_log
    print("dist = lognormal", f"DNC0 = {DNC0:.3e}",
          "mu_R =", mu_R, "sigma_R = ", sigma_R)
    dist_par = (DNC0, mu_m_log, sigma_m_log)

ensemble_path_add =\
f"{dist}/{gen_method}/eta_{eta:.0e}_{eta_threshold}/ensembles/"
result_path_add =\
f"{dist}/{gen_method}/eta_{eta:.0e}_{eta_threshold}\
/results/{kernel_name}/{kernel_method}/"

#%% KERNEL/ECOL GRID GENERATION
if act_gen_kernel_grid:
    if not os.path.exists(save_dir_kernel_grid):
        os.makedirs(save_dir_kernel_grid)        
    mg_ = generate_and_save_kernel_grid_Long_Bott(
              R_low_kernel, R_high_kernel, no_bins_10_kernel,
              mass_density, save_dir_kernel_grid )[2]
    print(f"generated kernel grid,",
          f"R_low = {R_low_kernel}, R_high = {R_high_kernel}",
          f"number of bins = {len(mg_)}")
if act_gen_Ecol_grid:
    if not os.path.exists(save_dir_Ecol_grid):
        os.makedirs(save_dir_Ecol_grid)        
    Rg_ = generate_and_save_E_col_grid_Long_Bott(
              R_low_kernel, R_high_kernel, no_bins_10_kernel,
              save_dir_Ecol_grid)[1]
    print(f"generated Ecol grid,",
          f"R_low = {R_low_kernel}, R_high = {R_high_kernel}",
          f"number of bins = {len(Rg_)}")

#%% SIP ENSEMBLE GENERATION
if act_gen_SIP:
    for kappa in kappa_list:
        ensemble_dir =\
            sim_data_path + ensemble_path_add + f"kappa_{kappa}/"
             
        generate_and_save_SIP_ensembles_SingleSIP_prob(
            dist, dist_par, mass_density, dV, kappa, eta, weak_threshold,
            r_critmin, m_high_over_m_low, no_sims, start_seed, ensemble_dir)

#%% SIP ENSEMBLE ANALYSIS AND PLOTTING
if act_analysis_ensembles:
    for kappa in kappa_list:
        ensemble_dir =\
            sim_data_path + ensemble_path_add + f"kappa_{kappa}/"
        
        analyze_ensemble_data(dist, mass_density, kappa, no_sims, ensemble_dir,
                          no_bins, bin_mode,
                          spread_mode, shift_factor, overflow_factor,
                          scale_factor, act_plot_ensembles)

#### OLD WORKING      
#        bins_mass, bins_rad, bins_rad_log, \
#        bins_mass_width, bins_rad_width, bins_rad_width_log, \
#        bins_mass_centers, bins_rad_centers, \
#        masses, xis, radii, f_m_counts, f_m_ind,\
#        f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled,\
#        m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, \
#        f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std, \
#        g_ln_r_num_avg, g_ln_r_num_std, \
#        m_min, m_max, R_min, R_max, no_SIPs_avg, \
#        moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,\
#        moments_an, \
#        f_m_num_avg_my_ext, \
#        f_m_num_avg_my, f_m_num_std_my, \
#        g_m_num_avg_my, g_m_num_std_my, \
#        h_m_num_avg_my, h_m_num_std_my, \
#        bins_mass_my, bins_mass_width_my, \
#        bins_mass_centers_my, bins_mass_center_lin_my, lin_par, aa = \
#            analyze_ensemble_data(dist, mass_density, kappa, no_sims,
#                                  ensemble_dir,
#                                  bin_method_init_ensembles, no_bins, bin_mode,
#                                  spread_mode, shift_factor, overflow_factor,
#                                  scale_factor)

### SIP ensemble plotting is now included in SIP analysis above        
#        if act_plot_ensembles:
#            plot_ensemble_data(kappa, mass_density, eta, r_critmin,
#                dist, dist_par, no_sims, no_bins, bin_method_init_ensembles,
#                bins_mass, bins_rad, bins_rad_log, 
#                bins_mass_width, bins_rad_width, bins_rad_width_log, 
#                bins_mass_centers, bins_rad_centers, 
#                masses, xis, radii, f_m_counts, f_m_ind,
#                f_m_num_sampled, g_m_num_sampled, g_ln_r_num_sampled, 
#                m_, R_, f_m_ana_, g_m_ana_, g_ln_r_ana_, 
#                f_m_num_avg, f_m_num_std, g_m_num_avg, g_m_num_std, 
#                g_ln_r_num_avg, g_ln_r_num_std, 
#                m_min, m_max, R_min, R_max, no_SIPs_avg, 
#                moments_sampled, moments_sampled_avg_norm,moments_sampled_std_norm,
#                moments_an, lin_par,
#                f_m_num_avg_my_ext,
#                f_m_num_avg_my, f_m_num_std_my, g_m_num_avg_my, g_m_num_std_my, 
#                h_m_num_avg_my, h_m_num_std_my, 
#                bins_mass_my, bins_mass_width_my, 
#                bins_mass_centers_my, bins_mass_center_lin_my,
#                ensemble_dir)

#%% SIMULATE COLLISIONS
if act_sim:
#%% SIMULATION DATA LOAD
    if kernel_method == "kernel_grid_m":
        # convert to 1E-18 kg if mass grid is given in kg...
        mass_grid = \
            1E18*np.load(save_dir_kernel_grid + "mass_grid_out.npy")
        kernel_grid = \
            np.load(save_dir_kernel_grid + "kernel_grid.npy" )
        m_kernel_low = mass_grid[0]
        bin_factor_m = mass_grid[1] / mass_grid[0]
        m_kernel_low_log = math.log(m_kernel_low)
        bin_factor_m_log = math.log(bin_factor_m)
        no_kernel_bins = len(mass_grid)
    
    if kernel_method == "Ecol_grid_R":
        radius_grid = \
            np.load(save_dir_Ecol_grid + "radius_grid_out.npy")
        E_col_grid = \
            np.load(save_dir_Ecol_grid + "E_col_grid.npy" )        
        R_kernel_low = radius_grid[0]
        bin_factor_R = radius_grid[1] / radius_grid[0]
        R_kernel_low_log = math.log(R_kernel_low)
        bin_factor_R_log = math.log(bin_factor_R)
        no_kernel_bins = len(radius_grid)
    
### SIMULATIONS
    for kappa in kappa_list:
        no_cols = np.array((0,0))
        print("simulation for kappa =", kappa)
        # SIP ensembles are already stored in directory
        # LINUX desk
        ensemble_dir =\
            sim_data_path + ensemble_path_add + f"kappa_{kappa}/"
        save_dir =\
            sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        sim_params = [dt, dV, no_sims, kappa, start_seed]
        
        seed_list = np.arange(start_seed, start_seed+no_sims*2, 2)
        for cnt,seed in enumerate(seed_list):
            # LOAD ENSEMBLE DATA
            # NOTE that enesemble masses are given in kg, but we need 1E-18 kg
            # for the collision algorithms
            masses = 1E18 * np.load(ensemble_dir + f"masses_seed_{seed}.npy")
            xis = np.load(ensemble_dir + f"xis_seed_{seed}.npy")                    
            if kernel_name == "Long_Bott":
                if kernel_method == "Ecol_grid_R":
                    mass_densities = np.ones_like(masses) * mass_density
                    radii = compute_radius_from_mass_vec(masses, mass_densities)
                    vel = compute_terminal_velocity_Beard_vec(radii)
                    SIP_quantities = (xis, masses, radii, vel, mass_densities)
                    kernel_quantities = \
                    (E_col_grid, no_kernel_bins, R_kernel_low_log,
                     bin_factor_R_log)
                        
                elif kernel_method == "kernel_grid_m":
                    SIP_quantities = (xis, masses)
                    kernel_quantities = \
                    (kernel_grid, no_kernel_bins, m_kernel_low_log,
                     bin_factor_m_log)
                        
                elif kernel_method == "analytic":
                    SIP_quantities = (xis, masses, mass_density)
                    kernel_quantities = None
            print(f"kappa {kappa}, sim {cnt}: seed {seed} simulation start")
            simulate_collisions(SIP_quantities,
                                kernel_quantities, kernel_name, kernel_method,
                                dV, dt, t_end, dt_save,
                                no_cols, seed, save_dir)
            print(f"kappa {kappa}, sim {cnt}: seed {seed} simulation finished")

#%% DATA ANALYSIS
### DATA ANALYSIS

if act_analysis:
    print("kappa, time_n, {xi_max/xi_min:.3e}, no_SIPS_avg, R_min, R_max")
    for kappa in kappa_list:
        load_dir =\
            sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        analyze_sim_data(kappa, mass_density, dV, no_sims, start_seed, no_bins, load_dir)

#%% PLOTTING
if act_plot:
    for kappa in kappa_list:
        load_dir =\
            sim_data_path + result_path_add + f"kappa_{kappa}/dt_{int(dt)}/"
        plot_for_given_kappa(kappa, eta, dt, no_sims, start_seed, no_bins,
                         kernel_name, gen_method, bin_method_sim_data, load_dir)

#%% PLOT MOMENTS VS TIME for several kappa
# act_plot_moments_kappa_var = True

TTFS, LFS, TKFS = 14,14,12
if act_plot_moments_kappa_var:
    ref_data_path = sim_data_path + f"{dist}/Wang2007_results2.txt"
    fig_dir = sim_data_path + result_path_add
    plot_moments_kappa_var(kappa_list, eta, dt, no_sims, no_bins,
                           kernel_name, gen_method,
                           dist, start_seed, ref_data_path, sim_data_path,
                           result_path_add,
                           fig_dir, TTFS, LFS, TKFS)
