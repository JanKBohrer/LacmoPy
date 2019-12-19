#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:35:52 2019

@author: bohrer
"""

#%% MODULE IMPORTS
### BUILT IN
#import os
import sys
#import math
#import numpy as np

### CUSTOM MODULES
#import constants as c
from file_handling import load_grid_and_particles_full
from integration import simulate
from init import set_initial_sim_config

#%% IMPORT PARAMETERS FROM COMFIG FILE
from config_sim import simpar

if len(sys.argv) > 1:
    simpar['seed_SIP_gen'] = int(sys.argv[1])
if len(sys.argv) > 2:
    simpar['seed_sim'] = int(sys.argv[2])

#%% CONFIG FILE PROCESSING

if simpar['simulation_mode'] == "with_collision_spin_up_included":
    simpar['simulation_mode'] = 'spin_up'
    data_paths, E_col_grid, no_cols, water_removed =\
        set_initial_sim_config(simpar)
    # set stdout file, if requested
    if simpar["set_std_out_file"]:
    #    sys.stdout = open(confpar.save_path + "std_out.log", 'w')
        sys.stdout = open( data_paths['output'] + "std_out.log", 'w')
    grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
        load_grid_and_particles_full(simpar["t_start"], data_paths["grid"])
    simulate(grid, pos, vel, cells, m_w, m_s, xi, active_ids,
             water_removed, no_cols, simpar, E_col_grid, data_paths['output'])
    
        
        

#%% INIT GRID AND PARTICLES


#%% SIMULATION    


if simpar['simulation_mode'] == "with_collision_spin_up_included":
    simpar['simulation_mode'] = 'with_collisions'
    data_paths, E_col_grid, no_cols, water_removed =\
        set_initial_sim_config(simpar)
