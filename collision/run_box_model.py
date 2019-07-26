#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:31:53 2019

@author: jdesk
"""

#%% IMPORTS AND DEFS

import os
import math
import numpy as np
#from numba import njit

#import matplotlib.pyplot as plt
#import matplotlib.ticker as mtick

import constants as c

from microphysics import compute_radius_from_mass_vec
from microphysics import compute_radius_from_mass_jit

from box_model import simulate_collisions
from box_model import analyze_sim_data
from box_model import plot_for_given_kappa
from box_model import plot_moments_kappa_var

from kernel import compute_terminal_velocity_Beard
from kernel import compute_terminal_velocity_Beard_vec

#%% AON Method

OS = "LinuxDesk"
# OS = "MacOS"

kernel_method = "Ecol_grid_R"
kernel_method = "kernel_grid_m"
kernel_method = "analytic"

### IN WORK: include more args for SIP ensemble creation...

# args for SIP ensemble generation
args_gen = [1,1,1]

act_gen = bool(args_gen[0])
act_analysis_ensembles = bool(args_gen[1])
act_plot_ensembles = bool(args_gen[2]) # NOTE: plotting needs analyze first (or in buffer)

args_sim = [1,0,0,0]
#args_sim = [0,1,1,1]
# args_sim = [1,1,1,1]

# act_sim = True
# act_analysis = True
# act_plot = True
# act_plot_moments_kappa_var = True

act_sim = bool(args_sim[0])
act_analysis = bool(args_sim[1])
act_plot = bool(args_sim[2])
act_plot_moments_kappa_var = bool(args_sim[3])

dist = "expo"

# kappa = 40

# kappa = 800
#
kappa_list=[40]
#kappa_list=[5]
#kappa_list=[5,10,20,40]
# kappa_list=[5,10,20,40,60,100,200]
# kappa_list=[5,10,20,40,60,100,200,400]
#kappa_list=[5,10,20,40,60,100,200,400]
# kappa_list=[5,10,20,40,60,100,200,400,600,800]
# kappa_list=[600]
# kappa_list=[800]

eta = 1.0E-9

eta_threshold = "weak"
#eta_threshold = "fix"

if eta_threshold == "weak":
    weak_threshold = True
else: weak_threshold = False

# no_sims = 163
# no_sims = 450
no_sims = 500
# no_sims = 50
# no_sims = 250

# no_sims = 94

start_seed = 3711

# start_seed = 4523

# start_seed = 4127
# start_seed = 4107
# start_seed = 4385
# start_seed = 3811

no_bins = 50

gen_method = "SinSIP"
# kernel_name = "Golovin"
kernel_name = "Long_Bott"

bin_method = "auto_bin"

# dt = 1.0
dt = 10.0
# dt = 20.0
dV = 1.0

mass_density = c.mass_density_water_liquid_NTP

no_cols = np.array((0,0))

# dt_save = 40.0
dt_save = 300.0
# t_end = 200.0
t_end = 3600.0

if OS == "MacOS":
    sim_data_path = "/Users/bohrer/sim_data/"
elif OS == "LinuxDesk":
    sim_data_path = "/mnt/D/sim_data_unif/col_box_mod/"
#    sim_data_path = "/mnt/D/sim_data_my_kernel_grid_strict_thresh/"
#    sim_data_path = "/mnt/D/sim_data/"

ensemble_path = f"{dist}/{gen_method}/eta_{eta:.0e}_{eta_threshold}/ensembles/"
result_path =\
f"{dist}/{gen_method}/eta_{eta:.0e}_{eta_threshold}\
/results/{kernel_name}/{kernel_method}/"

# ensemble_dir =\
#     f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/kappa_{kappa}/"
    # f"/mnt/D/sim_data/col_box_mod/ensembles/{gen_method}/eta_{eta:.0e}/"

# seed = start_seed
# radii = compute_radius_from_mass(masses*1.0E18,
#                                  c.mass_density_water_liquid_NTP)


#%% SIP GENERATION
    


#%% SIMULATION DATA LOAD
if kernel_method == "kernel_grid_m":
    mass_grid = \
        np.load(sim_data_path + result_path + f"kernel_data/mass_grid_out.npy")
    kernel_grid = \
        np.load(sim_data_path + result_path + f"kernel_data/kernel_grid.npy" )
    m_kernel_low = mass_grid[0]
    bin_factor_m = mass_grid[1] / mass_grid[0]
    m_kernel_low_log = math.log(m_kernel_low)
    bin_factor_m_log = math.log(bin_factor_m)
    no_kernel_bins = len(mass_grid)

if kernel_method == "Ecol_grid_R":
    # IN WORK!!!!!
    radius_grid = \
        np.load(sim_data_path + result_path
                + f"kernel_data/radius_grid_out.npy")
    E_col_grid = \
        np.load(sim_data_path + result_path + f"kernel_data/E_col_grid.npy" )        
    R_kernel_low = radius_grid[0]
    bin_factor_R = radius_grid[1] / radius_grid[0]
    R_kernel_low_log = math.log(R_kernel_low)
    bin_factor_R_log = math.log(bin_factor_R)
    no_kernel_bins = len(radius_grid)

#%% SIMULATE COLLISIONS
if act_sim:
    for kappa in kappa_list:
        print("simulation for kappa =", kappa)
        # SIP ensembles are already stored in directory
        # LINUX desk
        ensemble_dir =\
            sim_data_path + ensemble_path + "kappa_{kappa}/"
        save_dir =\
            sim_data_path + result_path + "kappa_{kappa}/dt_{int(dt)}/"
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
            print(f"sim {cnt}: seed {seed} simulation start")
            simulate_collisions(SIP_quantities,
                                kernel_quantities, kernel_name, kernel_method,
                                dV, dt, t_end, dt_save,
                                no_cols, seed, save_dir)
            print(f"sim {cnt}: seed {seed} simulation finished")

#%% DATA ANALYSIS
### DATA ANALYSIS
# IN WORK: SAVE MORE DATA AFTER ANALYSIS FOR FASTER PLOTTING

if act_analysis:
    print("kappa, time_n, {xi_max/xi_min:.3e}, no_SIPS_avg, R_min, R_max")
    for kappa in kappa_list:
        load_dir =\
            sim_data_path + result_path + "kappa_{kappa}/dt_{int(dt)}/"
        analyze_sim_data(mass_density, kappa, no_sims,
                         start_seed, no_bins, load_dir)


#%% PLOTTING
if act_plot:
    for kappa in kappa_list:
        load_dir =\
            sim_data_path + result_path + "kappa_{kappa}/dt_{int(dt)}/"
        plot_for_given_kappa(kappa, eta, dt, no_sims, start_seed, no_bins,
                             kernel_name, gen_method, bin_method, load_dir)

#%% PLOT MOMENTS VS TIME for several kappa
# act_plot_moments_kappa_var = True

TTFS, LFS, TKFS = 14,14,12
if act_plot_moments_kappa_var:
    ref_data_path = sim_data_path + f"{dist}/results/{kernel_name}/Wang2007_results2.txt"
    fig_dir = sim_data_path + result_path
    plot_moments_kappa_var(kappa_list, eta, dt, no_sims, no_bins,
                           kernel_name, gen_method,
                           dist, start_seed, ref_data_path,
                           sim_data_path, fig_dir, TTFS, LFS, TKFS)    
