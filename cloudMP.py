#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:07:21 2019

@author: jdesk
"""

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
# import math
import numpy as np
import matplotlib.pyplot as plt
import os
# from datetime import datetime
# import timeit

### MY MODULES
import constants as c
# from init import initialize_grid_and_particles, dst_log_normal

from file_handling import load_grid_and_particles_full
# from file_handling import dump_particle_data, save_grid_scalar_fields                          
#                         save_particles_to_files,\
# from analysis import compare_functions_run_time

from integration import simulate 

### STORAGE DIRECTORIES
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
    #     home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'

#%% III. SIMULATION

####################################
### SET
# load grid and particle list from directory
# folder_load = "190508/grid_10_10_spct_4/"
# folder_load = "190510/grid_15_15_spcm_4_4/"
# folder_load = "190514/grid_75_75_spcm_0_4/"
# folder_load = "grid_75_75_spcm_20_20/spinup/"
folder_load = "grid_75_75_spcm_4_4/"
folder_save = "grid_75_75_spcm_4_4/grav_from_start/"
# folder_save = "190508/grid_10_10_spct_4/sim5/"
# folder_save = "190510/grid_15_15_spcm_4_4/sim2/"
# folder_save = "190514/grid_75_75_spcm_0_4/sim5/"
# folder_save = "grid_75_75_spcm_20_20/after_spinup_2/"
# folder_save = "190511/grid_75_75_spcm_4_4"

t_start = 0.0
t_end = 14400.0 # s
# t_end = 3600.0 # s

dt = 1.0 # s # timestep of advection

# timescale "scale_dt" = of subloop with timestep dt_sub = dt/(2 * scale_dt)
# => scale_dt = dt/(2 dt_sub)
# with implicit Newton:
# from experience: dt_sub <= 0.1 s, then depending on dt, e.g. 1.0, 5.0 or 10.0:
# => scale_dt = 1.0/(0.2) = 5 OR scale_dt = 5.0/(0.2) = 25 OR N = 10.0/0.2 = 50
scale_dt = 5

Newton_iter = 2 # number of root finding iterations for impl. mass integration

# save grid properties T, p, Theta, r_v, r_l, S every "frame_every" steps dt
# MUST be >= than dump_every and an integer multiple of dump every
# grid frames are taken at
# t = t_start, t_start + n * frame_every * dt AND additionally at t = t_end
frame_every = 1200

# number of particles to be traced, evenly distributed over "active_ids"
# can also be an explicit array( [ID0, ID1, ...] )
trace_ids = 40

# positions and velocities of traced particles are saved at
# t = t_start, t_start + n * dump_every * dt AND additionally at t = t_end
# dump_every must be <= frame_every and frame_every/dump_every must be integer
dump_every = 10

# g must be positive (9.8...) or 0.0 (for spin up)
# g_set = 0.0
g_set = c.earth_gravity
### SET END
####################################

#################################### 
### INIT
path = simdata_path + folder_load

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(t_start, path)

path = simdata_path + folder_save
if not os.path.exists(path):
    os.makedirs(path)
### INIT END
#################################### 

simulate(grid,
         pos, vel, cells, m_w, m_s, xi,
         dt, scale_dt, t_start, t_end,
         Newton_iter, g_set,
         frame_every, dump_every, trace_ids,
         path)
