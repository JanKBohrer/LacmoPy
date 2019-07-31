#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 13 12:28:30 2019

@author: jdesk
"""

#%% MODULE IMPORTS & LOAD GRID

import numpy as np
import matplotlib.pyplot as plt
# import os
# from datetime import datetime
# import timeit

import constants as c
from microphysics import compute_R_p_w_s_rho_p, compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\
                          
from analysis import sample_masses, sample_radii, compare_functions_run_time,\
                        plot_scalar_field_frames, plot_pos_vel_pt, \
                        plot_particle_size_spectra
# from integration import compute_dt_max_from_CFL

### storage directories -> need to assign "simdata_path" and "fig_path"
my_OS = "Linux_desk"
# my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = "/mnt/D/sim_data_cloudMP/test_gen_grid_and_pt/"    
#    simdata_path = home_path + "OneDrive/python/sim_data/"
    # fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    home_path = "/Users/bohrer/"
    simdata_path = home_path + "OneDrive - bwedu/python/sim_data/"
    # fig_path =\
    #   home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'

#%% GRID PARAMETERS
# domain size
x_min = 0.0
x_max = 1500.0
z_min = 0.0
z_max = 1500.0

# grid steps
dx = 150.0
dy = 1.0
dz = 150.0
#dx = 20.0
#dy = 1.0
#dz = 20.0
#dx = 500.0
#dy = 1.0
#dz = 500.0

dV = dx*dy*dz

p_0 = 101500 # surface pressure in Pa
p_ref = 1.0E5 # ref pressure for potential temperature in Pa
r_tot_0 = 7.5E-3 # kg water / kg dry air (constant over whole domain in setup)
# r_tot_0 = 22.5E-3 # kg water / kg dry air
# r_tot_0 = 7.5E-3 # kg water / kg dry air
Theta_l = 289.0 # K

#%% PARTICLE PARAMETERS

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
no_spcm = np.array([10, 10])
#no_spcm = np.array([20, 20])
#no_spcm = np.array([0, 4])

from grid import compute_no_grid_cells_from_step_sizes

no_cells = compute_no_grid_cells_from_step_sizes(
               ((x_min, x_max),(z_min, z_max)), (dx, dz) ) 

grid_folder =\
    f"grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/"

# load_folder = "190511/grid_75_75_spcm_4_4/"
# folder_load_base = "190512/grid_75_75_spcm_0_4/"
# folder_load = "190512/grid_75_75_spcm_0_4/sim8/"
# folder_load_base = "190508/grid_10_10_spct_4/sim4/"
# folder_load = "190508/grid_10_10_spct_4/sim4/"
# folder_load_base = "grid_75_75_spcm_20_20/"
# folder_load = "grid_75_75_spcm_20_20/"
# folder_load_base = "grid_75_75_spcm_20_20/spinup/"
# folder_load = "grid_75_75_spcm_20_20/spinup/"
#folder_load = "grid_75_75_spcm_20_20/after_spinup_2/"
# folder_load = "grid_75_75_spcm_20_20/after_spinup/"
# folder_load_base = "grid_75_75_spcm_20_20/spinup/"
# folder_load = "grid_75_75_spcm_0_4/sim5/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
# folder_save = "190508/grid_10_10_spct_4/"
# folder_save = folder_load

#folder_add = "sim01/"
folder_add = "sim02_spin_up_2h/"

load_path = simdata_path + grid_folder + folder_add
# path = simdata_path + folder_load_base
#t = 0
#t = 7200
t = 14400
# t = 10800
t_start = 0
#t_start = 7200
#t_end = 7200
# t_end = 10800
t_end = 14400
reload = True

if reload:
    grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
        load_grid_and_particles_full(t, load_path)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                            grid.temperature[tuple(cells)] )
    R_s = compute_radius_from_mass_vec(m_s, c.mass_density_NaCl_dry)

#load_folder = grid_folder
#path = simdata_path + load_folder

#%% GRID THERMODYNAMIC FIELDS

grid.plot_thermodynamic_scalar_fields_grid()

#%% SPECTRA INIT

fig_path = load_path

target_cell = (0,0)
no_cells_x = 1
no_cells_z = 1

#for i in range(0,75,20):
#    for j in range(0,75,20):
for i in range(0,10,4):
    for j in range(0,10,4):
        target_cell = (i,j)
        plot_particle_size_spectra(m_w, m_s, xi, cells, grid,
                                   target_cell, no_cells_x, no_cells_z,
                                   no_rows=1, no_cols=1,
                                   TTFS=12, LFS=10, TKFS=10,
                                   fig_path = fig_path)

#%% PARTICLE TRAJECTORIES

frame_every, no_grid_frames, dump_every = np.load(load_path+"data_saving_paras.npy")
pt_dumps_per_grid_frame = frame_every // dump_every
grid_save_times = np.load(load_path+"grid_save_times.npy")

# grid_save_times =\
#     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
print("grid_save_times")
print(grid_save_times)

from file_handling import load_particle_data_from_blocks
vecs, scals, xis = load_particle_data_from_blocks(load_path, grid_save_times,
                                                  pt_dumps_per_grid_frame)

from analysis import plot_particle_trajectories
fig_name = load_path + "particle_trajectories.png"
plot_particle_trajectories(vecs[:,0], grid, MS=2.0, arrow_every=5,
                           fig_name=fig_name, figsize=(10,10),
                           TTFS=14, LFS=12, TKFS=12,
                           t_start=t_start, t_end=t_end)

#%% PARTICLE "DENSITIES"
from file_handling import load_particle_data_all
from analysis import plot_pos_vel_pt_with_time

# plot only every x IDs
ID_every = 8
# plot a series of "frames" defined by grid_save_times OR just one frame at
# the time of the current loaded grid and particles
plot_time_series = True
plot_frame_every = 1

frame_every, no_grid_frames, dump_every = np.load(load_path+"data_saving_paras.npy")
pt_dumps_per_grid_frame = frame_every // dump_every
grid_save_times = np.load(load_path+"grid_save_times.npy")

# grid_save_times =\
#     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
print("grid_save_times")
print(grid_save_times)

if plot_time_series:
    vec_data, scal_data, xi_data = load_particle_data_all(load_path, grid_save_times)
    pos_data = vec_data[:,0]
    vel_data = vec_data[:,1]
    pos2 = pos_data[::plot_frame_every,:,::ID_every]
    vel2 = vel_data[::plot_frame_every,:,::ID_every]
    fig_name = load_path+"particle_densities.png"
    plot_pos_vel_pt_with_time(pos2, vel2, grid, grid_save_times,
                        figsize=(8,8*len(pos2)), no_ticks = [6,6],
                        MS = 1.0, ARRSCALE=20, fig_name=fig_name)
else:
    pos2 = pos[:,::ID_every]
    vel2 = vel[:,::ID_every]
    
    fig_name = load_path + "particle_density.png"
    plot_pos_vel_pt(pos2, vel2, grid,
                        figsize=(8,8), no_ticks = [6,6],
                        MS = 1.0, ARRSCALE=30, fig_name=fig_name)

    
#%% PLOT GRID SCALAR FIELD FRAMES OVER TIME

#path = simdata_path + load_folder
plot_frame_every = 1
# field_ind = np.arange(6)
field_ind = np.array((0,1,2,5))

frame_every, no_grid_frames, dump_every = np.load(load_path+"data_saving_paras.npy")
pt_dumps_per_grid_frame = frame_every // dump_every
grid_save_times = np.load(load_path+"grid_save_times.npy")

fields = load_grid_scalar_fields(load_path, grid_save_times)
print("fields.shape")
print(fields.shape)

time_ind = np.arange(0, len(grid_save_times), plot_frame_every)

# grid_save_times =\
#     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
print("grid_save_times")
print(grid_save_times)

print("grid_save_times indices and times chosen:")
for idx_t in time_ind:
    print(idx_t, grid_save_times[idx_t])

no_ticks=[6,6]

fig_name = load_path + "scalar_fields.png"
# fig_name = None

plot_scalar_field_frames(grid, fields, grid_save_times,
                         field_ind, time_ind, no_ticks, fig_name)
