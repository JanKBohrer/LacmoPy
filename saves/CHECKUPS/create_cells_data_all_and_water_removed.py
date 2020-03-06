#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 13:09:11 2019

@author: jdesk
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
# import os
# from datetime import datetime
# import timeit

import constants as c
from microphysics import compute_R_p_w_s_rho_p_AS
from microphysics import compute_R_p_w_s_rho_p_NaCl
from microphysics import compute_radius_from_mass_vec
from file_handling import load_grid_and_particles_full,\
                          load_grid_scalar_fields\

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

#%% LOAD GRID AND PARTICLES AT TIME t_grid

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
    
#%%

save_times = np.load(load_path + "grid_save_times.npy")

#%%

t=0

filename_ = load_path + "grid_scalar_fields_t_" + str(int(t)) + ".npy"
fields_ = np.load(filename_)
grid_temperature = fields_[3]

[m_w, m_s] = np.load(load_path + f"particle_scalar_data_all_{int(t)}.npy")
pos = np.load(load_path + f"particle_vector_data_all_{int(t)}.npy")[0]
xi = np.load(load_path + f"particle_xi_data_all_{int(t)}.npy")

# ACTIVATE LATER
cells1 = np.load(load_path + f"particle_cells_data_all_{int(t)}.npy")    
cells2 = np.array( [np.floor(pos[0]/dx) , np.floor(pos[1]/dz)] ).astype(int)
#    np.save(load_path + f"particle_cells_data_all_{int(t)}.npy", cells)  
    
    