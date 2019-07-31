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

import constants as c
from init import initialize_grid_and_particles, dst_log_normal
# from grid import Grid
from grid import interpolate_velocity_from_cell_bilinear,\
                 compute_cell_and_relative_position
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
from file_handling import save_grid_and_particles_full,\
                          load_grid_and_particles_full
from file_handling import dump_particle_data, save_grid_scalar_fields                          
#                         save_particles_to_files,\
from analysis import compare_functions_run_time

from collision.AON import collision_step_Long_Bott_Ecol_grid_R_2D
from collision.kernel import generate_and_save_E_col_grid_Long_Bott

from numba import njit

from generate_SIP_ensemble_dst import \
    gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl

# keep velocities and mass densities fixed for a collision timestep
# the np-method is 2 times faster than the jitted version
# 1000 collision steps for 75 x 75 cells take 3240 s = 54 min
# ADD active ids and id list!!!
def collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_np(
        xi, m_w, vel, mass_densities, cells, no_cells,
        dt_over_dV, E_col_grid, no_kernel_bins,
        R_kernel_low_log, bin_factor_R_log, no_cols):
    
    for i in range(no_cells[0]):
        mask_i = (cells[0] == i)
        for j in range(no_cells[1]):
#            mask_j = (cells[1] == j)
            mask_ij = np.logical_and(mask_i , (cells[1] == j))
            
            # for given cell:
            xis = xi[mask_ij]
            masses = m_w[mask_ij]
#            mass_densities = np.ones_like(masses) * mass_density
            radii = compute_radius_from_mass_vec(masses,
                                                 mass_densities[mask_ij])
            vels = vel[:,mask_ij]
            
            ### IN WORK: SAFETY: IS THERE A PARTICLE IN THE CELL AT ALL?
            
            collision_step_Long_Bott_Ecol_grid_R_2D(
                    xis, masses, radii, vels, mass_densities,
                    dt_over_dV, E_col_grid, no_kernel_bins,
                    R_kernel_low_log, bin_factor_R_log, no_cols)            
            
            xi[mask_ij] = xis
            m_w[mask_ij] = masses    

#the np-method is 2 times faster
#collision_step_Long_Bott_Ecol_grid_R_all_cells_2D = \
#    njit()(collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_np)


#%%
### storage directories -> need to assign "sim_data_path" and "fig_path"
my_OS = "Linux_desk"
# my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    sim_data_path = "/mnt/D/sim_data_cloudMP/test_col2D/"
#    sim_data_path = home_path + "OneDrive/python/sim_data/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    home_path = "/Users/bohrer/"
    sim_data_path = home_path + "OneDrive - bwedu/python/sim_data/"
    fig_path = home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'


#%% I. GENERATE GRID AND PARTICLES

args = [0,0,0]

act_gen_grid = bool(args[0])
act_gen_Ecol_grid = bool(args[1])
act_sim = bool(args[2])
#act_gen_Ecol_grid = True

# from init import initialize_grid_and_particles

# 1a. generate grid and particles + save initial to file
# domain size
x_min = 0.0
x_max = 1500.0
z_min = 0.0
z_max = 1500.0

# grid steps
#dx = 750.0
#dy = 1.0
#dz = 750.0
dx = 500.0
dy = 1.0
dz = 500.0
#dx = 150.0
#dy = 1.0
#dz = 150.0
#dx = 20.0
#dy = 1.0
#dz = 20.0

p_0 = 101500 # surface pressure in Pa
p_ref = 1.0E5 # ref pressure for potential temperature in Pa
r_tot_0 = 7.5E-3 # kg water / kg dry air
# r_tot_0 = 22.5E-3 # kg water / kg dry air
# r_tot_0 = 7.5E-3 # kg water / kg dry air
Theta_l = 289.0 # K
# number density of particles mode 1 and mode 2:
#n_p = np.array([60.0E6, 100.0E6]) # m^3
n_p = np.array([60.0E6, 40.0E6]) # m^3

# no_super_particles_cell = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
#no_spcm = np.array([20, 20])
no_spcm = np.array([2, 2])

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
S_init_max = 1.05
dt_init = 0.1 # s
# number of iterations for the root finding
# Newton algorithm in the implicit method
Newton_iterations = 2
# maximal allowed iter counts in initial particle water take up to equilibrium
iter_cnt_limit = 800

#folder = "190514/grid_75_75_spcm_0_4/"
#folder = "grid_75_75_no_spcm_20_20/"
folder = "grid_3_3_no_spcm_2_2/"
path = sim_data_path + folder

#%%



#%%

if act_gen_grid:
    if not os.path.exists(path):
        os.makedirs(path)
    
    grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids =\
        initialize_grid_and_particles(
            x_min, x_max, z_min, z_max, dx, dy, dz,
            p_0, p_ref, r_tot_0, Theta_l,
            n_p, no_spcm, dst, dst_par, 
            P_min, P_max, r0, r1, dr, rnd_seed, reseed,
            S_init_max, dt_init, Newton_iterations, iter_cnt_limit, path)

#%%
        
t_start = 0.0
   
load_dir = sim_data_path + folder
grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(t_start, load_dir)    

mass_density = 1E3

#dt = 1.0
dt = 10.0
#dt = 20.0
#dt = 100.0
dV = grid.volume_cell

dt_over_dV = dt / dV

# dt_save = 40.0
dt_save = 300.0
# t_end = 200.0
t_end = 3600.0

save_dir_Ecol_grid = sim_data_path + f"Ecol_grid_data/"
kernel_method = "Ecol_grid_R"


#%%
n_p = np.array([60.0E6, 40.0E6]) # 1/m^3

no_rpcm = grid.volume_cell * n_p

no_spcm = np.array([2, 2])

kappa = np.rint( no_spcm / 20 * 35) // 10

kappa = np.maximum(kappa, 0.5)

print("kappa =", kappa)


# in log(mu)
par_sigma = np.log( [1.4, 1.6] )
# par_sigma = np.log( [1.0,1.0] )
# in mu
par_r0 = 0.5 * np.array( [0.04, 0.15] )
###
idx_mode_nonzero = np.nonzero(no_spcm)[0]

no_modes = len(idx_mode_nonzero)

print(idx_mode_nonzero)
print(no_modes)

#%%

r_critmin = np.array([0.001, 0.00375]) # mu

if no_modes == 1:
    no_spcm = no_spcm[idx_mode_nonzero][0]
    par_sigma = par_sigma[idx_mode_nonzero][0]
    par_r0 = par_r0[idx_mode_nonzero][0]
    no_rpcm = no_rpcm[idx_mode_nonzero][0]
    r_critmin = r_critmin[idx_mode_nonzero][0] # mu
else:    
    no_spcm = no_spcm[idx_mode_nonzero]
    par_sigma = par_sigma[idx_mode_nonzero]
    par_r0 = par_r0[idx_mode_nonzero]
    no_rpcm = no_rpcm[idx_mode_nonzero]
    r_critmin = r_critmin[idx_mode_nonzero] # mu

# derive parameters of lognormal distribution of mass f_m(m)
# assuming mu_R in mu and density in kg/m^3
# mu_m in 1E-18 kg
mu_m_log = np.log(compute_mass_from_radius_vec(par_r0,c.mass_density_NaCl_dry))
sigma_m_log = 3.0 * par_sigma


eta = 1E-9
#eta_threshold = "weak"
eta_threshold = "fix"
if eta_threshold == "weak":
    weak_threshold = True
else: weak_threshold = False


m_high_over_m_low = 1.0E8

# set DNC0 manually only for lognormal distr. NOT for expo
#DNC0 = 60.0E6 # 1/m^3
##DNC0 = 2.97E8 # 1/m^3
##DNC0 = 3.0E8 # 1/m^3
#mu_R = 0.02 # in mu
#sigma_R = 1.4 #  no unit

seed = 3711
no_cells = (3,3)

seed_list = np.arange(seed, seed + 2*no_cells[1], 2)

masses = []
xis = []
cells_x = []
cells_z = []
for j in range(no_cells[1]):
    masses_lvl, xis_lvl, cells_x_lvl, modes_lvl = \
        gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl(no_modes,
                mu_m_log, sigma_m_log, c.mass_density_NaCl_dry,
                dV, kappa, eta, weak_threshold, r_critmin,
                m_high_over_m_low, seed_list[j], no_cells[0], no_rpcm)
    # now, masses_lvl , xis_lvl are arrays and can be indexed by cells_x_lvl
    # modes_lvl is just for checking
    # do magic stuff... init saturation adjustment..
    # -> think about, what we need right now: R_s, m_w, m_p, R_p, ...
    # other things can be done later by full array operation
    masses.append(masses_lvl)
    xis.append(xis_lvl)
    cells_x.append(cells_x_lvl)
    cells_z.append(np.ones_like(cells_x_lvl) * j)
    

#%%

R_low_kernel, R_high_kernel, no_bins_10_kernel = 0.01, 51.0, 100

no_cols = np.array((0,0))

import time

no_times = 1000

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

if act_sim:

    start_time = time.time()
    
    rad0 = compute_radius_from_mass_vec(m_w+m_s, mass_density)
    
    vel0 = np.copy(vel)
    m_w0 = np.copy(m_w)
    xi0 = np.copy(xi)

    mass_densities = np.ones_like(m_w) * mass_density
    
#    no_cells = grid.no_cells
    
    
    for time_n in range(no_times):
        if time_n % 10 == 0: print("time step = ", time_n)
#        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D(
        collision_step_Long_Bott_Ecol_grid_R_all_cells_2D_np(
            xi, m_w, vel, mass_densities, cells, grid.no_cells,
            dt_over_dV, E_col_grid, no_kernel_bins,
            R_kernel_low_log, bin_factor_R_log, no_cols)
    
    end_time = time.time()
    
    
#    ##### COLLISION STEP FOR ALL CELLS
#    for i in range(grid.no_cells[0]):
#        mask_i = (cells[0] == i)
#        for j in range(grid.no_cells[1]):
##            mask_j = (cells[1] == j)
#            mask_ij = np.logical_and(mask_i , (cells[1] == j))
##            print(i,j)
##            print(cells[:,mask_ij])
##            print(xi[mask_ij])
##    #        print(f"{xi[mask_ij]:.2e}")
##            print()
#            
#            # for given cell:
#            xis = xi[mask_ij]
#            masses = m_w[mask_ij]
#            mass_densities = np.ones_like(masses) * mass_density
#            radii = compute_radius_from_mass_vec(masses, mass_densities)
#            vels = vel[:,mask_ij]
#            
#            collision_step_Long_Bott_Ecol_grid_R_2D(
#                    xis, masses, radii, vels, mass_densities,
#                    dt_over_dV, E_col_grid, no_kernel_bins,
#                    R_kernel_low_log, bin_factor_R_log, no_cols)            
#            
#            xi[mask_ij] = xis
#            m_w[mask_ij] = masses
##            vel[:,mask_ij] = vels
            
#print(xi - xi0)
#print(m_w - m_w0)

    print(no_cols)

#print( compute_radius_from_mass_vec(m_w, mass_density) )

    print(end_time - start_time)

