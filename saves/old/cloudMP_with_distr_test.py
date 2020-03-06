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

# 3. changes grid, pos, cells, vel, masses, data outout to file and 
# save to file in intervals chosen
# need to set start time, end time, integr. params, interval of print,
# interval of full save, ...)
# simulate(grid, pos, vel, masses, xi)

#%%

import numpy as np
import math
import matplotlib.pyplot as plt
import os

import constants as c
from init import initialize_grid_and_particles, dst_log_normal
from grid import Grid
from grid import interpolate_velocity_from_cell_bilinear_jit,\
                 interpolate_velocity_from_position_bilinear_jit,\
                 compute_cell_and_relative_position_jit
from microphysics import compute_mass_from_radius,\
                         compute_initial_mass_fraction_solute_NaCl,\
                         compute_radius_from_mass,\
                         compute_density_particle,\
                 compute_delta_water_liquid_and_mass_rate_implicit_Newton_full,\
                         compute_R_p_w_s_rho_p
                         
from atmosphere import compute_kappa_air_moist,\
                       compute_diffusion_constant,\
                       compute_thermal_conductivity_air,\
                       compute_specific_heat_capacity_air_moist,\
                       compute_heat_of_vaporization,\
                       compute_saturation_pressure_vapor_liquid,\
                       compute_pressure_vapor,\
                       compute_pressure_ideal_gas,\
                       epsilon_gc, compute_surface_tension_water,\
                       kappa_air_dry,\
                       compute_beta_without_liquid,\
                       compute_temperature_from_potential_temperature_moist                
from file_handling import save_particles_to_files,\
                          save_grid_and_particles_full,\
                          load_grid_and_particles_full
from evaluation import sample_masses, sample_radii

def plot_pos_vel_pt(pos, vel, grid,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARRSCALE=2):
    u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
    v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
    ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
    ax.quiver(*pos, *vel, scale=ARRSCALE, pivot="mid")
    # ax.quiver(*grid.centers, u_g[:,0:-1], v_g[0:-1],
              # scale=ARRSCALE, pivot="mid", color="red")
    # ax.quiver(grid.corners[0], grid.corners[1] + 0.5*grid.steps[1],
    #           grid.velocity[0], np.zeros_like(grid.velocity[0]),
    #           scale=0.5, pivot="mid", color="red")
    # ax.quiver(grid.corners[0] + 0.5*grid.steps[0], grid.corners[1],
    #           np.zeros_like(grid.velocity[1]), grid.velocity[1],
    #           scale=0.5, pivot="mid", color="blue")
    x_min = grid.ranges[0,0]
    x_max = grid.ranges[0,1]
    y_min = grid.ranges[1,0]
    y_max = grid.ranges[1,1]
    ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
    ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
    # ax.set_xticks(grid.corners[0][:,0])
    # ax.set_yticks(grid.corners[1][0,:])
    ax.set_xticks(grid.corners[0][:,0], minor = True)
    ax.set_yticks(grid.corners[1][0,:], minor = True)
    # plt.minorticks_off()
    # plt.minorticks_on()
    ax.grid()
    # ax.grid(which="minor")
    plt.show()


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

#%%

# 1a. generate grid and particles + save initial to file
# domain size
x_min = 0.0
x_max = 1500.0
z_min = 0.0
z_max = 1500.0

# grid steps
dx = 150.0
dy = 1.0
dz = 150.0

p_0 = 101500 # surface pressure in Pa
p_ref = 1.0E5 # ref pressure for potential temperature in Pa
r_tot_0 = 7.5E-3 # kg water / kg dry air
# r_tot_0 = 22.5E-3 # kg water / kg dry air
# r_tot_0 = 7.5E-3 # kg water / kg dry air
Theta_l = 289.0 # K
# number density of particles mode 1 and mode 2:
# n_p = np.array([60.0E6, 100.0E6]) # m^3
n_p = np.array([60.0E6, 40.0E6]) # m^3

# no_super_particles_cell = [N1,N2] is a list with N1 = no super part. per cell in mode 1 etc.
no_spcm = np.array([2, 2])
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
rnd_seed = 4711

### for saturation adjustment phase
S_init_max = 1.05
dt_init = 0.1 # s
# number of iterations for the root finding (full) Newton algorithm in the implicit method
Newton_iterations = 2
# maximal allowed iter counts in initial particle water take up to equilibrium
iter_cnt_limit = 800

### save initial profile/particles to files

folder = "190508/grid_10_10_spct_4/"
path = simdata_path + folder
if not os.path.exists(path):
    os.makedirs(path)
# path = folder1 + folder2
#####

# grid_para_file = path + "grid_paras.txt"
# paras = [x_min, x_max, z_min, z_max, dx, dy, dz, p_0, p_ref, r_tot_0,
#          Theta_l, n_p,
#          no_spcm,
#          par_sigma, par_r0, rnd_seed, S_init_max, dt_init, iter_cnt_limit]
# para_names = 'x_min, x_max, z_min, z_max, dx, dy, dz, p_0, p_ref, r_tot_0, \
# Theta_l, n_p, no_super_particles_cell, \
# par_sigma, par_r0, rnd_seed, S_init_max, dt_init, iter_cnt_limit'

# with open(grid_para_file, "w") as f:
#     f.write( para_names + '\n' )
#     for item in paras:
#         if type(item) is list or type(item) is np.ndarray:
#             for el in item:
#                 f.write( f'{el} ' )
#         else: f.write( f'{item} ' )

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids =\
    initialize_grid_and_particles(
        x_min, x_max, z_min, z_max, dx, dy, dz,
        p_0, p_ref, r_tot_0, Theta_l,
        n_p, no_spcm, dst, dst_par, 
        P_min, P_max, r0, r1, dr, rnd_seed, reseed,
        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, path)

#%%
# 1b. load grid from file
# folder = "190507/test4/"
# path = simdata_path + folder

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)

R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)] )
R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)


# print(cells)
# print(cells.shape)
# print(type(xi[0]))

#%%
# first testings

plot_pos_vel_pt(pos, vel, grid, MS = 2.0, ARRSCALE=10)

grid.plot_thermodynamic_scalar_fields_grid()


#%%
### distribution tests

from testing import test_initial_size_distribution

test_initial_size_distribution(R_s, xi, no_spcm, dst_par,
                               fig_path=path + "init_size_dist.png")

target_cell = [5,5]
no_cells_x = 5
no_cells_z = 5

from plotting import plot_particle_size_spectra
plot_particle_size_spectra(m_w, m_s, xi, cells, grid,
                          target_cell, no_cells_x, no_cells_z)


#%%



                