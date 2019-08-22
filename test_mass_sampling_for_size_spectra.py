#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 12:45:30 2019

@author: jdesk
"""

import numpy as np

ind_i = [0,0,1,1,2,2,3,3,4,4]
ind_j = [0,1,2,0,1,2,4,5,6,4]

ind_i = np.array(ind_i)
ind_j = np.array(ind_j)

id_list = np.arange(len(ind_i))

target_cell = [2,2]

no_cells_x = 3
no_cells_z = 3

dx = no_cells_x // 2
dz = no_cells_z // 2

i_low = target_cell[0] - dx
i_high = target_cell[0] + dx
j_low = target_cell[0] - dz
j_high = target_cell[0] + dz

i_an = range(target_cell[0] - dx, target_cell[0] + dx + 1)
j_an = range(target_cell[1] - dz, target_cell[1] + dz + 1)

MM = np.meshgrid(i_an, j_an, indexing = "ij")

mask =   ((ind_i >= i_low) & (ind_i <= i_high)) \
       & ((ind_j >= j_low) & (ind_j <= j_high))
       
#%% STORAGE DIRECTORIES
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
seed_SIP_gen_list = [3711, 3713, 3715, 3717]

# for collisons
seed_sim = 4711
seed_sim_list = [4711, 4711, 4711, 4711]

simulation_mode = "spin_up"
#simulation_mode = "wo_collision"
#simulation_mode = "with_collision"

#spin_up_finished = True
spin_up_finished = False

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

#args_plot = [1,1,1,1,1,1]
#args_plot = [0,1,0,0,0]
#args_plot = [0,0,1,0,0]
#args_plot = [0,0,0,1,0]
#args_plot = [0,0,0,0,1]
args_plot = [0,0,0,0,0,0,1]

act_plot_scalar_fields_once = args_plot[0]
act_plot_spectra = args_plot[1]
act_plot_particle_trajectories = args_plot[2]
act_plot_particle_positions = args_plot[3]
act_plot_scalar_field_frames = args_plot[4]
act_plot_scalar_field_frames_ext = args_plot[5]
act_plot_grid_frames_avg = args_plot[6]
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
        save_folder = f"w_spin_up_w_col/{seed_sim}/"
    else:
        save_folder = f"wo_spin_up_w_col/{seed_sim}/"

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





       