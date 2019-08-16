#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:07:21 2019

@author: jdesk
"""

### IN WORK: make main file, which can be executed with "python cloudMP.py"
# give textfile to program, with type of simulation:
# generate grid and particles = yes, no; spinup = yes, no; 
# simulation: include gravity, condensation, collision, which type of collision
#

# 1. init()
# 2. spinup()
# 3. simulate()

## output:
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
### BUILT IN
import math
import numpy as np
#import matplotlib.pyplot as plt
import os
# from datetime import datetime
# import timeit

### MY MODULES
import constants as c
# from init import initialize_grid_and_particles, dst_log_normal

#from grid import compute_no_grid_cells_from_step_sizes

from file_handling import load_grid_and_particles_full
# from file_handling import dump_particle_data, save_grid_scalar_fields                          
#                         save_particles_to_files,\
# from analysis import compare_functions_run_time

from integration import simulate
#from integration import simulate_wout_col, simulate_col 

### STORAGE DIRECTORIES
my_OS = "Linux_desk"
#my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = "/mnt/D/sim_data_cloudMP/"
#    simdata_path = "/mnt/D/sim_data_cloudMP/test_gen_grid_and_pt/"
#    sim_data_path = home_path + "OneDrive/python/sim_data/"
#    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
#    home_path = "/Users/bohrer/sim_data_cloudMP/test_gen_grid_and_pt/"
    simdata_path = "/Users/bohrer/sim_data_cloudMP/"
#    simdata_path = home_path + "OneDrive - bwedu/python/sim_data/"
#    fig_path = home_path \
#               + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'

#%% SET PARAMETERS

#%% GRID PARAMETERS
#no_cells = (15, 15)
no_cells = (75, 75)

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
seed_SIP_gen = 3717

#%% SIMULATION PARAMETERS

####################################

# "spin up": g_set = 0.0, without collisions
# "wo_collisions": g_set = 9.805.., without collisions
# "with_collisions": g_set = 9.805.., with collisions
simulation_mode = "spin_up"
#simulation_mode = "wo_collision"
#simulation_mode = "with_collision"
 
# set True when starting from a spin-up state
#spin_up_before = True
spin_up_before = False

t_start = 0.0
#t_start = 600.0
#t_start = 7200.0 # s

t_end = 7200.0 # s
#t_end = 7200.0+2.0 # s
#t_end = 7200.0*2 # s
#t_end = 60.0 # s
#t_end = 1200.0 # s
#t_end = 1800.0 # s
#t_end = 2.0 # s
#t_end = 20.0 # s

dt = 1.0 # s # timestep of advection

# timescale "scale_dt" = of subloop with timestep dt_sub = dt/(2 * scale_dt)
# => scale_dt = dt/(2 dt_sub)
# with implicit Newton:
# from experience: dt_sub <= 0.1 s, then depends on dt, e.g. 1.0, 5.0 or 10.0:
# => scale_dt = 1.0/(0.2) = 5 OR scale_dt = 5.0/(0.2) = 25 OR 10.0/0.2 = 50
scale_dt = 5
Newton_iter = 3 # number of root finding iterations for impl. mass integration

# save grid properties T, p, Theta, r_v, r_l, S every "frame_every" steps dt
# MUST be >= than dump_every and an integer multiple of dump every
# grid frames are taken at
# t = t_start, t_start + n * frame_every * dt AND additionally at t = t_end
#frame_every = 1200
#frame_every = 600
frame_every = 300
#frame_every = 30
#frame_every = 1

# number of particles to be traced, evenly distributed over "active_ids"
# can also be an explicit array( [ID0, ID1, ...] )
trace_ids = 40

# positions and velocities of traced particles are saved at
# t = t_start, t_start + n * dump_every * dt AND additionally at t = t_end
# dump_every must be <= frame_every and frame_every/dump_every must be integer
dump_every = 10
#dump_every = 5
#dump_every = 1

#%% COLLISIONS PARAMS

# for collisions
seed_sim = 3711

kernel_type = "Long_Bott"
kernel_method = "Ecol_grid_R"

save_dir_Ecol_grid = simdata_path + f"Ecol_grid_data/{kernel_type}/"

### SET END
####################################

#%% DERIVED

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

no_cols = np.array((0,0))

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

if spin_up_before:
    grid_folder += "spin_up_wo_col_wo_grav/"

# the seed is added later automatically for collision simulations
#save_folder = "no_spin_up_with_col/"
#save_folder = "no_spin_up_with_col_speedtest/"
#save_folder = "spin_up_wo_col_wo_grav/"
#save_folder = simdata_path + grid_folder + "no_spin_up_col_speed_test/"

#%% INIT GRID AND PARTICLES
#path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
    load_grid_and_particles_full(t_start, simdata_path + grid_folder)

if act_collisions:
    save_path += f"{seed_sim}/"
#path = simdata_path + folder_save
if not os.path.exists(save_path):
    os.makedirs(save_path)
### INIT END
#################################### 

water_removed = np.array([0.0])

#%% SIMULATION

#if act_collisions:
simulate(grid, pos, vel, cells, m_w, m_s, xi, solute_type,
         water_removed,
         active_ids,
         dt, scale_dt, t_start, t_end, Newton_iter, g_set,
         act_collisions,
         frame_every, dump_every, trace_ids, 
         E_col_grid, no_kernel_bins,
         R_kernel_low_log, bin_factor_R_log,
         kernel_type, kernel_method,
         no_cols, seed_sim,
         save_path, simulation_mode)             
#else:
#    simulate_wout_col(grid,
#        pos, vel, cells, m_w, m_s, xi, solute_type, water_removed,
#        active_ids,
#        dt, scale_dt, t_start, t_end,
#        Newton_iter, g_set,
#        frame_every, dump_every, trace_ids,
#        save_path, simulation_mode)
