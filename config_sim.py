#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:40:09 2019

@author: bohrer
"""

import os
#import math
import numpy as np

import constants as c

from file_handling import load_kernel_data

#%% SET THESE VARIABLES

# this is the initial value,
# if another value is given as script execution argument
# then the following value is overwritten by the execution argument
#seed_SIP_gen = 3711
seed_SIP_gen = 1011

# this is the initial value,
# if another value is given as script execution argument
# then the following value is overwritten by the execution argument
seed_sim = 2011

simdata_path = "/Users/bohrer/sim_data_cloudMP/"
#simdata_path = "/Users/bohrer/sim_data_cloudMP_TEST191206/"
#no_cells = [75,75]
no_cells = [5,5]

solute_type = "AS"

#no_spcm = [26,38]
no_spcm = [2,2]

#simulation_mode = "with_collision"
simulation_mode = "spin_up"
#simulation_mode = "wo_collision"

# set True when starting from a spin-up state,
# this will be set to False below, if the simulation mode is "spin_up"
spin_up_before = True
#spin_up_before = False

t_start = 0
#t_end = 7200
t_end = 300
dt_adv = 1
no_cond_per_adv = 10
no_col_per_adv = 2
# RENAME to no_iter_impl_mass or similar
no_iter_impl_mass = 3
#frame_every = 300
frame_every = 100
#trace_ids = 80
trace_ids = 20
dump_every = 10

### collision kernel parameters
kernel_type = "Long_Bott"
# kernel_type = "Hall_Bott"
# kernel_type = "Hydro_E_const"
kernel_method = "Ecol_grid_R"
# value must be set, but is only used for kernel_method = "Ecol_const",
E_col_const = 0.5
save_dir_Ecol_grid =  f"Ecol_grid_data/{kernel_type}/"

#%% DERIVED

# IN WORK: put the derived part in a separate function! then only paras
# in config file, for better use of different config files and übersichtlich

# timescale "scale_dt" = of subloop with timestep dt_sub = dt/(2 * scale_dt)
# => scale_dt = dt/(2 dt_sub)
# with implicit Newton:
# from experience: dt_sub <= 0.1 s, then depends on dt, e.g. 1.0, 5.0 or 10.0:
# => scale_dt = 1.0/(0.2) = 5 OR scale_dt = 5.0/(0.2) = 25 OR 10.0/0.2 = 50
scale_dt_cond = no_cond_per_adv // 2

### load collision kernel data
E_col_grid, radius_grid, \
R_kernel_low, bin_factor_R, \
R_kernel_low_log, bin_factor_R_log, \
no_kernel_bins =\
    load_kernel_data(kernel_method, save_dir_Ecol_grid, E_col_const)

### set the load and save paths
if simulation_mode == "spin_up" or int(t_start) == 0:
    spin_up_before = False

# g must be positive (9.8...) or 0.0 (for spin up)
if simulation_mode == "spin_up":
    g_set = 0.0
else:
    g_set = c.earth_gravity

if simulation_mode == "with_collision":
    act_collisions = True
else:    
    act_collisions = False

if simulation_mode == "spin_up":
    save_folder = "spin_up_wo_col_wo_grav/"
elif simulation_mode == "wo_collision":
    if spin_up_before:
        save_folder = "w_spin_up_wo_col/"
    else:
        save_folder = "wo_spin_up_wo_col/"
elif simulation_mode == "with_collision":
    if spin_up_before:
        save_folder = "w_spin_up_w_col/"
    else:
        save_folder = "wo_spin_up_w_col/"

grid_folder =\
    f"{solute_type}" \
    + f"/grid_{no_cells[0]}_{no_cells[1]}_spcm_{no_spcm[0]}_{no_spcm[1]}/" \
    + f"{seed_SIP_gen}/"

save_path = simdata_path + grid_folder + save_folder

if act_collisions:
    save_path += f"{seed_sim}/"
#path = simdata_path + folder_save
if not os.path.exists(save_path):
    os.makedirs(save_path)

if spin_up_before:
#    if t_start <= 7200.:
    if t_start <= 7201.:
        grid_folder += "spin_up_wo_col_wo_grav/"
    elif simulation_mode == "with_collision":
        grid_folder += f"w_spin_up_w_col/{seed_sim}/"

# set "counters" for number of collisions and total removed water
no_cols = np.array((0,0))
if t_start > 0.:
    water_removed = np.load(simdata_path + grid_folder + f"water_removed_{int(t_start)}.npy")
else:        
    water_removed = np.array([0.0])
        