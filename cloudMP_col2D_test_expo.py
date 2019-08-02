#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:13:49 2019

@author: jdesk
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import os
from datetime import datetime
# import timeit

#import constants as c
#from init import initialize_grid_and_particles_SinSIP, dst_log_normal
# from grid import Grid
#from grid import interpolate_velocity_from_cell_bilinear,\
#                 compute_cell_and_relative_position
                 # interpolate_velocity_from_position_bilinear,\
#from microphysics import compute_radius_from_mass ,\
#                         compute_R_p_w_s_rho_p
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
from microphysics import compute_radius_from_mass_vec
from microphysics import compute_mass_from_radius_vec
#from file_handling import save_grid_and_particles_full,\
#                          load_grid_and_particles_full
#from file_handling import dump_particle_data, save_grid_scalar_fields                          
#                         save_particles_to_files,\
#from analysis import compare_functions_run_time

from collision.AON import collision_step_Long_Bott_Ecol_grid_R_2D
from collision.AON import collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_np

from collision.AON import \
    collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np

from collision.kernel import generate_and_save_E_col_grid_Long_Bott

#from numba import njit

#from generate_SIP_ensemble_dst import \
#    gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl

# keep velocities and mass densities fixed for a collision timestep
# the np-method is 2 times faster than the jitted version
# 1000 collision steps for 75 x 75 cells take 3240 s = 54 min
# ADD active ids and id list!!!
#def collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_np(
#        xi, m_w, vel, mass_densities, cells, no_cells,
#        dt_over_dV, E_col_grid, no_kernel_bins,
#        R_kernel_low_log, bin_factor_R_log, no_cols):
#    
#    for i in range(no_cells[0]):
#        mask_i = (cells[0] == i)
#        for j in range(no_cells[1]):
##            mask_j = (cells[1] == j)
#            mask_ij = np.logical_and(mask_i , (cells[1] == j))
#            
#            # for given cell:
#            xis = xi[mask_ij]
#            masses = m_w[mask_ij]
##            mass_densities = np.ones_like(masses) * mass_density
#            radii = compute_radius_from_mass_vec(masses,
#                                                 mass_densities[mask_ij])
#            vels = vel[:,mask_ij]
#            
#            ### IN WORK: SAFETY: IS THERE A PARTICLE IN THE CELL AT ALL?
#            
#            collision_step_Long_Bott_Ecol_grid_R_2D(
#                    xis, masses, radii, vels, mass_densities,
#                    dt_over_dV, E_col_grid, no_kernel_bins,
#                    R_kernel_low_log, bin_factor_R_log, no_cols)            
#            
#            xi[mask_ij] = xis
#            m_w[mask_ij] = masses    

#the np-method is 2 times faster
#collision_step_Long_Bott_Ecol_grid_R_all_cells_2D = \
#    njit()(collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_np)


#%%
### storage directories -> need to assign "sim_data_path" and "fig_path"
my_OS = "Linux_desk"
# my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
#    sim_data_path = "/mnt/D/sim_data_cloudMP_col/test_col2D_expo/"
#    sim_data_path = "/mnt/D/sim_data_cloudMP_col/test_col2D_expo2/"
#    sim_data_path = "/mnt/D/sim_data_cloudMP_col/test_col2D_expo3/"
#    sim_data_path = "/mnt/D/sim_data_cloudMP_col/test_col2D_expo4/"
#    sim_data_path = "/mnt/D/sim_data_cloudMP_col/test_col2D_expo5/"
#    sim_data_path = "/mnt/D/sim_data_cloudMP_col/test_col2D_expo6/"
#    sim_data_path = "/mnt/D/sim_data_cloudMP_col/test_col2D_expo7/"
#    sim_data_path = "/mnt/D/sim_data_cloudMP_col/test_col2D_expo8/"
    sim_data_path = "/mnt/D/sim_data_cloudMP_col/test_col2D_expo9/"
    
#    sim_data_path = home_path + "OneDrive/python/sim_data/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    home_path = "/Users/bohrer/"
    sim_data_path = home_path + "OneDrive - bwedu/python/sim_data/"
    fig_path = home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'


#%%
args = [0,1,1]

#act_gen_grid = bool(args[0])
act_gen_Ecol_grid = bool(args[0])
act_sim = bool(args[1])
act_analysis = bool(args[2])

#folder = "190514/grid_75_75_spcm_0_4/"
#folder = "grid_75_75_no_spcm_20_20/"
#folder = "grid_3_3_no_spcm_2_2/"
#path = sim_data_path + folder


#%%
        
   
#load_dir = sim_data_path + folder

#mass_density = 1E3

t_start = 0.0

#dt = 1.0
dt = 10.0
#dt = 20.0
#dt = 100.0
dV = 1.

dt_over_dV = dt / dV

# dt_save = 40.0
dt_save = 300.0
# t_end = 200.0
#t_end = 100.0
t_end = 3600.0

no_times = int(math.ceil(t_end/dt))
#no_times = 10

dn_save = int(math.ceil(dt_save/dt))

save_dir_Ecol_grid = sim_data_path + f"Ecol_grid_data/"
kernel_method = "Ecol_grid_R"

eta = 1E-9
#eta_threshold = "weak"
eta_threshold = "fix"
if eta_threshold == "weak":
    weak_threshold = True
else: weak_threshold = False

m_high_over_m_low = 1.0E8

seed = 3711
#no_cells = (2,2)
#no_cells = (5,5)
#no_cells = (10,10)
no_cells = (15,15)
#no_cells = (10,10)

#%%

R_low_kernel, R_high_kernel, no_bins_10_kernel = 0.6, 6E3, 100

no_cols = np.array((0,0))

import time

#ndim = 2

if act_gen_Ecol_grid:
    if not os.path.exists(save_dir_Ecol_grid):
        os.makedirs(save_dir_Ecol_grid)        
    Rg_ = generate_and_save_E_col_grid_Long_Bott(
              R_low_kernel, R_high_kernel, no_bins_10_kernel,
              save_dir_Ecol_grid)[1]
    print(f"generated Ecol grid,",
          f"R_low = {R_low_kernel}, R_high = {R_high_kernel}",
          f"number of bins = {len(Rg_)}")

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

#%%

mass_density = 1E3
R_mean = 9.3 # mu
m_mean = compute_mass_from_radius_vec(R_mean, mass_density)

#kappa = 40
#kappa = 20
#kappa = 10
kappa = 5

r_critmin = 0.6

LWC0 = 1E-3 # kg /m^3

DNC0 = 1E18 * LWC0 / m_mean

no_rpcm = DNC0 * dV

from generate_SIP_ensemble_dst import \
    gen_mass_ensemble_weights_SinSIP_expo_grid_np
    
m_w, xi, cells = \
    gen_mass_ensemble_weights_SinSIP_expo_grid_np(
        m_mean, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low, seed, no_cells, no_rpcm)

m_s = np.zeros_like(m_w)

#%%

def plot_xi_vs_R_grid(m_w, mass_density, cells, fig_path):
    R = compute_radius_from_mass_vec(m_w, mass_density)
    
    no_rows = no_cells[0]
    no_cols = no_cells[1]
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                           figsize=(no_cols*5,no_rows*4) )
    
    for i in range(no_rows):
        mask_i = (cells[0] == i)
        for j in range(no_cols):
            mask_ij = np.logical_and(mask_i , (cells[1] == j))
            ax = axes[i,j]
            ax.loglog(R[mask_ij], xi[mask_ij], "x")
            ax.grid()   
            ax.set_yticks(np.logspace(-2,8,11))
    fig.tight_layout()
    fig.savefig(fig_path)
    
def plot_xi_vs_R_all_in_one(m_w, mass_density, cells, fig_path):
    R = compute_radius_from_mass_vec(m_w, mass_density)
    
    no_rows = 1
    no_cols = 1
#    no_rows = no_cells[0]
#    no_cols = no_cells[1]
    
    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
                           figsize=(no_cols*5,no_rows*4) )
    
#    for i in range(no_rows):
#        mask_i = (cells[0] == i)
#        for j in range(no_cols):
#            mask_ij = np.logical_and(mask_i , (cells[1] == j))
#            ax = axes[i,j]
#            ax.loglog(R[mask_ij], xi[mask_ij], "x")
#            ax.grid()   
#            ax.set_yticks(np.logspace(-2,8,11))
    
    ax = axes
    ax.loglog(R, xi, "x")
    ax.grid()   
    ax.set_yticks(np.logspace(-2,8,11))
    
    fig.tight_layout()
    fig.savefig(fig_path)

def plot_moments_vs_time(moments_vs_time, save_times, ref_data_path, fig_path):
    t_Wang = np.linspace(0,60,7)
    moments_vs_time_Wang = np.loadtxt(ref_data_path)
    moments_vs_time_Wang = np.reshape(moments_vs_time_Wang,(4,7)).T
    moments_vs_time_Wang[:,0] *= 1.0E6
    moments_vs_time_Wang[:,1] *= 1.0E-3
    moments_vs_time_Wang[:,2] *= 1.0E-6
    moments_vs_time_Wang[:,3] *= 1.0E-12
    
    no_rows = moments_vs_time.shape[0]
    
    save_times = save_times /  60.
    
    fig, axes = plt.subplots(nrows=no_rows, 
                           figsize=(10,no_rows*6) )
    
    for i, ax in enumerate(axes):
        ax.plot(save_times, moments_vs_time[i], "x-")
        ax.plot(t_Wang, moments_vs_time_Wang[:,i],
                "o", c = "k",fillstyle='none',
                markersize = 10, mew=3.5, label="Wang")        
        if i != 1:
            ax.set_yscale("log")
        ax.grid(which="both")
        ax.set_xlim([0,60])
    
    
    fig.tight_layout()
    fig.savefig(fig_path)
    
grid_temperature = 283. * np.ones( no_cells )

#from collision.kernel import compute_terminal_velocity_Beard
#from collision.kernel import compute_terminal_velocity_Beard_my_mat_const
from collision.kernel import compute_terminal_velocity_Beard_vec

vel_x = np.zeros_like(xi)
rad = compute_radius_from_mass_vec(m_w, mass_density)
vel_z = compute_terminal_velocity_Beard_vec(rad)

vel = np.array((vel_x, vel_z))

save_times = []

save_path = sim_data_path\
    + f"no_cells_{no_cells[0]}_{no_cells[1]}/kappa_{kappa}/"
if not os.path.exists(save_path):
    os.makedirs(save_path)

plt.ioff()

if act_sim:
    start_time = time.time()
    
    fig_path = save_path + "xi_vs_R_t_0.png"
    plot_xi_vs_R_all_in_one(m_w, mass_density, cells, fig_path)

    mass_densities = mass_density * np.ones_like(xi)
#    plot_xi_vs_R_grid(m_w, mass_density, cells, fig_path)
    
#    R = compute_radius_from_mass_vec(m_w, mass_density)
#    
#    no_rows = no_cells[0]
#    no_cols = no_cells[1]
#    
#    fig, axes = plt.subplots(nrows=no_rows, ncols=no_cols,
#                           figsize=(no_cols*4,no_rows*4) )
#    
#    for i in range(no_rows):
#        for j in range(no_cols):
#            ax = axes[i,j]
#            ax.loglog(R, xi, "x")
    
    np.save(save_path + f"cells", cells)
    for time_n in range(no_times):
#        if time_n % 10 == 0: print("time step = ", time_n)
        
        rad = compute_radius_from_mass_vec(m_w, mass_density)
        vel[1] = compute_terminal_velocity_Beard_vec(rad)
        
        if time_n % dn_save == 0:
            t = int(time_n * dt)
            np.save(save_path + f"xi_{t}", xi)
            np.save(save_path + f"m_w_{t}", m_w)
            np.save(save_path + f"m_s_{t}", m_s)
            save_times.append(t)
        
    # folder = expo
#        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_np(
#        xi, m_w, vel, mass_densities, cells, no_cells,
#        dt_over_dV, E_col_grid, no_kernel_bins,
#        R_kernel_low_log, bin_factor_R_log, no_cols)
        
    # folder = expo3/expo4
#        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
#            xi, m_w, m_s, vel, mass_densities, cells, no_cells,
#            dt_over_dV, E_col_grid, no_kernel_bins,
#            R_kernel_low_log, bin_factor_R_log, no_cols)
    # folder = expo2
        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_multicomp_np(
            xi, m_w, m_s, vel, grid_temperature, cells, no_cells,
            dt_over_dV, E_col_grid, no_kernel_bins,
            R_kernel_low_log, bin_factor_R_log, no_cols)
        
    t = int(t_end)
    np.save(save_path + f"xi_{t}", xi)
    np.save(save_path + f"m_w_{t}", m_w)
    np.save(save_path + f"m_s_{t}", m_s)
    save_times.append(t)
    np.save(save_path + f"save_times", save_times)
    
    
    
    end_time = time.time()
    
    print("no_cols")
    print(no_cols)

    print("end_time - start_time")
    print(end_time - start_time)


    fig_path = save_path + f"xi_vs_R_t_{int(t_end)}.png"
    plot_xi_vs_R_all_in_one(m_w, mass_density, cells, fig_path)
#    plot_xi_vs_R_grid(m_w, mass_density, cells, fig_path)
        
    
def compute_moments_num(xi, m_w, no_cells_tot, no_moments):
    moments_num = np.zeros(no_moments)
    
    m_w *= 1E-18
    
    moments_num[0] = np.sum(xi) / no_cells_tot
    
    if no_moments > 1:
        for n in range(1, no_moments):
            moments_num[n] = np.sum(xi * m_w**n) / no_cells_tot
    return moments_num
    
if act_analysis:
    save_times = np.load(save_path + f"save_times.npy")
    cells = np.load(save_path + f"cells.npy")
    no_cells_tot = no_cells[0] * no_cells[1]
    
    no_times = len(save_times)
    
    no_moments = 4
    
    moments_vs_time = np.zeros( (no_moments, no_times) )
    
    for time_n,t in enumerate(save_times):
        xi = np.load(save_path + f"xi_{t}.npy")
        m_w = np.load(save_path + f"m_w_{t}.npy")
        
        moments_vs_time[:,time_n] = \
            compute_moments_num(xi, m_w, no_cells_tot, no_moments)
            
    fig_path = save_path + "moments_vs_time.png"
    
    ref_data_path = sim_data_path + "Wang2007_results2.txt"
    plot_moments_vs_time(moments_vs_time, save_times, ref_data_path,
                         fig_path)        
        
    
    

