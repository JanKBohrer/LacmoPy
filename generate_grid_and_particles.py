#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:07:21 2019

@author: jdesk
"""

# 1. init()
# 2. spinup()
# 3. simulate()

### output:
## full saves:
# grid_parameters, grid_scalars, grid_vectors
# pos, cells, vel, masses, xi
## data:
# grid:
# initial: p, T, Theta, S, r_v, r_l, rho_dry, e_s
# continuous
# p, T, Theta, S, r_v, r_l, 
# particles:
# pos, vel, masses


# set:
# files for full save
# files for data output -> which data?

# 1. generate grid and particles + save initial to file
# grid, pos, cells, vel, masses, xi = init()

# 2. changes grid, pos, cells, vel, masses and save to file after spin up
# data output during spinup if desired
# spinup(grid, pos, cells, vel, masses, xi)
# in here:
# advection(grid) (grid, dt_adv) spread over particle time step h
# propagation(pos, vel, masses, grid) (particles) # switch gravity on/off here!
# condensation() (particle <-> grid) maybe do together with collision 
# collision() maybe do together with condensation

# 3. changes grid, pos, cells, vel, masses,
# data output to file and 
# save to file in intervals chosen
# need to set start time, end time, integr. params, interval of print,
# interval of full save, ...)
# simulate(grid, pos, vel, masses, xi)

#%% MODULE IMPORTS

import numpy as np
# import math
import matplotlib.pyplot as plt
import os
# from datetime import datetime
# import timeit

# import constants as c
from init import initialize_grid_and_particles, dst_log_normal
# from grid import Grid
# from grid import interpolate_velocity_from_cell_bilinear,\
#                  compute_cell_and_relative_position
                 # interpolate_velocity_from_position_bilinear,\
# from microphysics import compute_radius_from_mass,\
#                          compute_R_p_w_s_rho_p
#                          compute_initial_mass_fraction_solute_NaCl,\
#                          compute_density_particle,\
#                          compute_dml_and_gamma_impl_Newton_full,\
                         
# from atmosphere import compute_kappa_air_moist,\
#                        compute_pressure_vapor,\
#                        compute_pressure_ideal_gas,\
#                        epsilon_gc, compute_surface_tension_water,\
#                        kappa_air_dry,\
#                        compute_beta_without_liquid,\
#                        compute_temperature_from_potential_temperature_moist                
                       # compute_diffusion_constant,\
                       # compute_thermal_conductivity_air,\
                       # compute_specific_heat_capacity_air_moist,\
                       # compute_heat_of_vaporization,\
                       # compute_saturation_pressure_vapor_liquid,\
# from file_handling import save_grid_and_particles_full
                          # load_grid_and_particles_full
# from file_handling import dump_particle_data, save_grid_scalar_fields                          
#                         save_particles_to_files,\
# from analysis import compare_functions_run_time

### storage directories -> need to assign "simdata_path" and "fig_path"
my_OS = "Linux_desk"
# my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = home_path + "OneDrive/python/sim_data/"
    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    home_path = "/Users/bohrer/"
    simdata_path = home_path + "OneDrive - bwedu/python/sim_data/"
    fig_path = home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'



#%% I. GENERATE GRID AND PARTICLES
# from init import initialize_grid_and_particles

# 1a. generate grid and particles + save initial to file
# domain size
x_min = 0.0
x_max = 1500.0
z_min = 0.0
z_max = 1500.0

# grid steps
dx = 20.0
dy = 1.0
dz = 20.0

p_0 = 101500 # surface pressure in Pa
p_ref = 1.0E5 # ref pressure for potential temperature in Pa
r_tot_0 = 7.5E-3 # kg water / kg dry air
# r_tot_0 = 22.5E-3 # kg water / kg dry air
# r_tot_0 = 7.5E-3 # kg water / kg dry air
Theta_l = 289.0 # K
# number density of particles mode 1 and mode 2:
n_p = np.array([60.0E6, 40.0E6]) # m^3
# n_p = np.array([60.0E6, 40.0E6]) # m^3

# no_super_particles_cell = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
no_spcm = np.array([20, 20])
# no_spcm = np.array([0, 4])

### creating random particles
# parameters of log-normal distribution:
dst = dst_log_normal
# in log(mu)
par_sigma = np.log( [1.4,1.6] )
# par_sigma = np.log( [1.0,1.0] )
# in mu
par_r0 = 0.5 * np.array( [0.04, 0.15] )
dst_par = []
for i,sig in enumerate(par_sigma):
    dst_par.append([par_r0[i],sig])
dst_par = np.array(dst_par)
# parameters of the quantile-integration
P_min = 0.001
P_max = 0.999
dr = 1E-4
r0 = dr
r1 = 10 * par_sigma
reseed = False
rnd_seed = 4713

### for saturation adjustment phase
S_init_max = 1.04
dt_init = 0.1 # s
# number of iterations for the root finding
# Newton algorithm in the implicit method
Newton_iterations = 2
# maximal allowed iter counts in initial particle water take up to equilibrium
iter_cnt_limit = 800

folder = "grid_75_75_spcm_20_20/"
path = simdata_path + folder
if not os.path.exists(path):
    os.makedirs(path)

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids =\
    initialize_grid_and_particles(
        x_min, x_max, z_min, z_max, dx, dy, dz,
        p_0, p_ref, r_tot_0, Theta_l,
        n_p, no_spcm, dst, dst_par, 
        P_min, P_max, r0, r1, dr, rnd_seed, reseed,
        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, path)

#%% simple plots
# IN WORK: ADD AUTOMATIC CREATION OF DRY SIZE SPECTRA COMPARED WITH EXPECTED PDF
# FOR ALL MODES AND SAVE AS FILE 
# grid.plot_thermodynamic_scalar_fields_grid()