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
from microphysics import compute_R_p_w_s_rho_p, compute_radius_from_mass
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\
                          
from analysis import sample_masses, sample_radii, compare_functions_run_time,\
                        plot_scalar_field_frames, plot_pos_vel_pt
# from integration import compute_dt_max_from_CFL

### storage directories -> need to assign "simdata_path" and "fig_path"
my_OS = "Linux_desk"
# my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = home_path + "OneDrive/python/sim_data/"
    # fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    home_path = "/Users/bohrer/"
    simdata_path = home_path + "OneDrive - bwedu/python/sim_data/"
    # fig_path =\
    #   home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'

# folder_load = "190511/grid_75_75_spcm_4_4/"
# folder_load_base = "190512/grid_75_75_spcm_0_4/"
# folder_load = "190512/grid_75_75_spcm_0_4/sim8/"
# folder_load_base = "190508/grid_10_10_spct_4/sim4/"
# folder_load = "190508/grid_10_10_spct_4/sim4/"
# folder_load_base = "grid_75_75_spcm_20_20/"
# folder_load = "grid_75_75_spcm_20_20/"
# folder_load_base = "grid_75_75_spcm_20_20/spinup/"
folder_load = "grid_75_75_spcm_20_20/spinup/"
# folder_load = "grid_75_75_spcm_20_20/after_spinup/"
# folder_load_base = "grid_75_75_spcm_20_20/spinup/"
# folder_load = "grid_75_75_spcm_0_4/sim5/"
# folder_load = "190507/test1/"
# folder_load = "190508/grid_75_75_spct_20/"
# folder_save = "190508/grid_10_10_spct_4/"
# folder_save = folder_load

path = simdata_path + folder_load
# path = simdata_path + folder_load_base
t = 7200
# t = 10800
t_start = 0
# t_start = 7200
t_end = 7200
# t_end = 10800
reload = True

if reload:
    grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
        load_grid_and_particles_full(t, path)
    R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                            grid.temperature[tuple(cells)] )
    R_s = compute_radius_from_mass(m_s, c.mass_density_NaCl_dry)

path = simdata_path + folder_load

#%% GRID THERMODYNAMIC FIELDS

grid.plot_thermodynamic_scalar_fields_grid()

#%% PARTICLE TRAJECTORIES
frame_every, no_grid_frames, dump_every = np.load(path+"data_saving_paras.npy")
pt_dumps_per_grid_frame = frame_every // dump_every
grid_save_times = np.load(path+"grid_save_times.npy")

# grid_save_times =\
#     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
print("grid_save_times")
print(grid_save_times)

from file_handling import load_particle_data_from_blocks
vecs, scals, xis = load_particle_data_from_blocks(path, grid_save_times,
                                                  pt_dumps_per_grid_frame)

from analysis import plot_particle_trajectories
fig_name = path + "particle_trajectories.png"
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

frame_every, no_grid_frames, dump_every = np.load(path+"data_saving_paras.npy")
pt_dumps_per_grid_frame = frame_every // dump_every
grid_save_times = np.load(path+"grid_save_times.npy")

# grid_save_times =\
#     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
print("grid_save_times")
print(grid_save_times)

if plot_time_series:
    vec_data, scal_data, xi_data = load_particle_data_all(path, grid_save_times)
    pos_data = vec_data[:,0]
    vel_data = vec_data[:,1]
    pos2 = pos_data[::plot_frame_every,:,::ID_every]
    vel2 = vel_data[::plot_frame_every,:,::ID_every]
    fig_name = path+"particle_densities.png"
    plot_pos_vel_pt_with_time(pos2, vel2, grid, grid_save_times,
                        figsize=(8,8*len(pos2)), no_ticks = [6,6],
                        MS = 1.0, ARRSCALE=20, fig_name=fig_name)
else:
    pos2 = pos[:,::ID_every]
    vel2 = vel[:,::ID_every]
    
    fig_name = path + "particle_density.png"
    plot_pos_vel_pt(pos2, vel2, grid,
                        figsize=(8,8), no_ticks = [6,6],
                        MS = 1.0, ARRSCALE=30, fig_name=fig_name)

    
#%% PLOT GRID SCALAR FIELD FRAMES OVER TIME

path = simdata_path + folder_load
plot_frame_every = 1
# field_ind = np.arange(6)
field_ind = np.array((0,1,2,5))

frame_every, no_grid_frames, dump_every = np.load(path+"data_saving_paras.npy")
pt_dumps_per_grid_frame = frame_every // dump_every
grid_save_times = np.load(path+"grid_save_times.npy")

fields = load_grid_scalar_fields(path, grid_save_times)
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

fig_name = path + "scalar_fields.png"
# fig_name = None

plot_scalar_field_frames(grid, fields, grid_save_times,
                         field_ind, time_ind, no_ticks, fig_name)
