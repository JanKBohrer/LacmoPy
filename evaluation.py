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
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\
                          

#                     plot_particle_size_spectra
# from integration import compute_dt_max_from_CFL
#from grid import compute_no_grid_cells_from_step_sizes

### STORAGE DIRECTORIES
my_OS = "Linux_desk"
#my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = "/mnt/D/sim_data_cloudMP/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
#    fig_path = home_path \
#               + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'

#%% GRID PARAMETERS

no_cells = (75, 75)
#no_cells = (3, 3)

dx = 20.
dy = 1.
dz = 20.
dV = dx*dy*dz

#%% PARTICLE PARAMETERS

# solute material: NaCl OR ammonium sulfate
#solute_type = "NaCl"
solute_type = "AS"

# no_super_particles_cell_mode = [N1,N2] is a list with
# N1 = no super part. per cell in mode 1 etc.
# with init method = SingleSIP, this is only the target value.
# the true number of particles per cell and mode will fluctuate around this
#no_spcm = np.array([10, 10])
#no_spcm = np.array([12, 12])
no_spcm = np.array([16, 24])

# seed of the SIP generation -> needed for the right grid folder
# 3711, 3713, 3715, 3717
# 3719, 3721, 3723, 3725
seed_SIP_gen = 3711

# for collisons
seed_sim = 4711

simulation_mode = "spin_up"
#simulation_mode = "wo_collision"
#simulation_mode = "with_collision"

spin_up_finished = True
#spin_up_finished = False

# path = simdata_path + folder_load_base
t_grid = 0
#t = 60
#t = 3600
#t = 7200
#t = 14400
# t = 10800

t_start = 0
#t_start = 7200

#t_end = 60
#t_end = 3600
t_end = 7200
# t_end = 10800
#t_end = 14400

args_plot = [1,1,1,1,1]
#args_plot = [0,1,0,0,0]
#args_plot = [0,0,1,0,0]
#args_plot = [0,0,0,1,0]
#args_plot = [0,0,0,0,1]

act_plot_scalar_fields_once = args_plot[0]
act_plot_spectra = args_plot[1]
act_plot_particle_trajectories = args_plot[2]
act_plot_particle_positions = args_plot[3]
act_plot_scalar_field_frames = args_plot[4]

#%% LOAD GRID AND PARTICLES AT TIME t

grid_folder =\
    f"{solute_type}" \
    + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
    + f"{seed_SIP_gen}/"

if simulation_mode == "spin_up":
    save_folder = "spin_up_wo_col_wo_grav/"
elif simulation_mode == "wo_collision":
    if spin_up_finished:
        save_folder = "w_spin_up_wo_col/"
    else:
        save_folder = "wo_spin_up_wo_col/"
elif simulation_mode == "with_collision":
    if spin_up_finished:
        save_folder = "w_spin_up_w_col/"
    else:
        save_folder = "wo_spin_up_w_col/"

# load grid and particles full from grid_path at time t
if int(t_grid) == 0:        
    grid_path = simdata_path + grid_folder
else:    
    grid_path = simdata_path + grid_folder + save_folder

load_path = simdata_path + grid_folder + save_folder

#reload = True
#
#if reload:
grid, pos, cells, vel, m_w, m_s, xi, active_ids  = \
    load_grid_and_particles_full(t_grid, grid_path)

if solute_type == "AS":
    compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_AS
    mass_density_dry = c.mass_density_AS_dry
elif solute_type == "NaCl":
    compute_R_p_w_s_rho_p = compute_R_p_w_s_rho_p_NaCl
    mass_density_dry = c.mass_density_NaCl_dry

R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s,
                                        grid.temperature[tuple(cells)] )
R_s = compute_radius_from_mass_vec(m_s, mass_density_dry)

#%% GRID THERMODYNAMIC FIELDS
if act_plot_scalar_fields_once:
    fig_path = load_path
    grid.plot_thermodynamic_scalar_fields(t=t_grid, fig_path = fig_path)

#%% PARTICLE SPECTRA

from analysis import plot_particle_size_spectra_tg_list

plt.ioff()

if act_plot_spectra:
    t = t_grid
    fig_path = load_path
    
    target_cell = (0,0)
    no_cells_x = 9
    no_cells_z = 1
    
    i_tg = [20,40,60]
    j_tg = [20,40,45,50,55,60,65,70,74]
#    i_tg = [20]
#    j_tg = [20,40]
    i_list, j_list = np.meshgrid(i_tg,j_tg, indexing = "xy")
    target_cell_list = [i_list.flatten(), j_list.flatten()]
    
    no_rows = len(j_tg)
    no_cols = len(i_tg)
#    no_rows = 3
#    no_cols = 2
    filename_ = load_path + "grid_scalar_fields_t_" + str(int(t)) + ".npy"
    fields_ = np.load(filename_)
    grid_temperature = fields_[3]
    
    [m_w, m_s] = np.load(load_path + f"particle_scalar_data_all_{int(t)}.npy")
    pos = np.load(load_path + f"particle_vector_data_all_{int(t)}.npy")[0]
    xi = np.load(load_path + f"particle_xi_data_all_{int(t)}.npy")
    
    # ACTIVATE LATER
    cells = np.load(load_path + f"particle_cells_data_all_{int(t)}.npy")
    # FOR NOW..., DEACTIVATE LATER
#    cells = np.array( [np.floor(pos[0]/dx) , np.floor(pos[1]/dz)] ).astype(int)
#    np.save(load_path + f"particle_cells_data_all_{int(t)}.npy", cells)
    
    plot_particle_size_spectra_tg_list(
        t, m_w, m_s, xi, cells,
        dV,
        dz,
        grid_temperature,
        solute_type,
        target_cell_list, no_cells_x, no_cells_z,
        no_rows, no_cols,
        TTFS=12, LFS=10, TKFS=10,
        fig_path = fig_path)
#    plot_particle_size_spectra_tg_list(t, m_w, m_s, xi, cells, grid, solute_type,
#                          target_cell_list, no_cells_x, no_cells_z,
#                          no_rows, no_cols,
#                          TTFS=12, LFS=10, TKFS=10,
#                          fig_path = fig_path)
    
#    for i in range(0,75,20):
##    for i in range(0,75,20):
#        for j in range(49,75,5):
##        for j in range(60,75,5):
#    #for i in range(0,10,4):
#    #    for j in range(0,10,4):
#    #for i in range(0,3):
#    #    for j in range(0,3):
#            target_cell = (i,j)
#            plot_particle_size_spectra(m_w, m_s, xi, cells, grid, solute_type,
#                                       target_cell, no_cells_x, no_cells_z,
#                                       no_rows=1, no_cols=1,
#                                       TTFS=12, LFS=10, TKFS=10,
#                                       fig_path = fig_path)

#%% PARTICLE TRAJECTORIES

if act_plot_particle_trajectories:
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    pt_dumps_per_grid_frame = frame_every // dump_every
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    # grid_save_times =\
    #     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
    print("grid_save_times")
    print(grid_save_times)
    
    from file_handling import load_particle_data_from_blocks
    vecs, scals, xis = load_particle_data_from_blocks(load_path,
                                                      grid_save_times,
                                                      pt_dumps_per_grid_frame)
    
    from analysis import plot_particle_trajectories
    fig_name = load_path + "particle_trajectories_" \
               + f"t_{int(t_start)}_" \
               + f"{int(t_end)}.png"
    plot_particle_trajectories(vecs[:,0], grid, MS=2.0, arrow_every=5,
                               fig_name=fig_name, figsize=(10,10),
                               TTFS=14, LFS=12, TKFS=12,
                               t_start=t_start, t_end=t_end)

#%% PARTICLE POSITIONS AND VELOCITIES

if act_plot_particle_positions:
    from file_handling import load_particle_data_all
    from file_handling import load_particle_data_all_old
    from analysis import plot_pos_vel_pt_with_time, plot_pos_vel_pt
    # plot only every x IDs
    ID_every = 40
    # plot a series of "frames" defined by grid_save_times OR just one frame at
    # the time of the current loaded grid and particles
    plot_time_series = True
    plot_frame_every = 4
    
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    pt_dumps_per_grid_frame = frame_every // dump_every
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    # grid_save_times =\
    #     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
    print("grid_save_times")
    print(grid_save_times)
    fig_name = load_path + "particle_positions_" \
               + f"t_{grid_save_times[0]}_" \
               + f"{grid_save_times[-1]}.png"
    if plot_time_series:
        vec_data, scal_data, xi_data  = \
            load_particle_data_all_old(load_path, grid_save_times)
#        vec_data, cells_data, scal_data, xi_data, active_ids_data = \
#            load_particle_data_all(load_path, grid_save_times)
        pos_data = vec_data[:,0]
        vel_data = vec_data[:,1]
        pos2 = pos_data[::plot_frame_every,:,::ID_every]
        vel2 = vel_data[::plot_frame_every,:,::ID_every]
        plot_pos_vel_pt_with_time(pos2, vel2, grid,
                                  grid_save_times[::plot_frame_every],
                            figsize=(8,8*len(pos2)), no_ticks = [6,6],
                            MS = 1.0, ARRSCALE=20, fig_name=fig_name)
    else:
        pos2 = pos[:,::ID_every]
        vel2 = vel[:,::ID_every]
        
        plot_pos_vel_pt(pos2, vel2, grid,
                            figsize=(8,8), no_ticks = [6,6],
                            MS = 1.0, ARRSCALE=30, fig_name=fig_name)
    
#%% PLOT GRID SCALAR FIELD FRAMES OVER TIME

if act_plot_scalar_field_frames:
    from analysis import plot_scalar_field_frames
    #path = simdata_path + load_folder
    plot_frame_every = 2
    # field_ind = np.arange(6)
    field_ind = np.array((0,1,2,5))
    
    frame_every, no_grid_frames, dump_every = \
        np.load(load_path+"data_saving_paras.npy")
    pt_dumps_per_grid_frame = frame_every // dump_every
    grid_save_times = np.load(load_path+"grid_save_times.npy")
    
    fields = load_grid_scalar_fields(load_path, grid_save_times)
    print("fields.shape")
    print(fields.shape)
    
    time_ind = np.arange(0, len(grid_save_times), plot_frame_every)
    
    # grid_save_times =\
    #     np.arange(t_start, t_end + 0.5 * dt_save, dt_save).astype(int)
    print()
    print("plot scalar field frames with times:")
    print("grid_save_times")
    print(grid_save_times)
    
    print("grid_save_times indices and times chosen:")
    for idx_t in time_ind:
        print(idx_t, grid_save_times[idx_t])
    
    no_ticks=[6,6]
    
    fig_name = load_path \
               + f"scalar_fields_" \
               + f"t_{grid_save_times[0]}_" \
               + f"{grid_save_times[-1]}.png"
    # fig_name = None
    
    plot_scalar_field_frames(grid, fields, grid_save_times,
                             field_ind, time_ind, no_ticks, fig_name)
