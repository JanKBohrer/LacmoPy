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

#%% CONFIG FILE PROCESSING

data_paths, E_col_grid, no_cols, water_removed = set_initial_sim_config(simpar)

# set stdout file, if requested
if simpar["set_std_out_file"]:
#    sys.stdout = open(confpar.save_path + "std_out.log", 'w')
    sys.stdout = open( data_paths['output'] + "std_out.log", 'w')

#%% INIT GRID AND PARTICLES

grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
    load_grid_and_particles_full(simpar["t_start"], data_paths["grid"])

#%% SIMULATION    

#simulate(grid, pos, vel, cells, m_w, m_s, xi, active_ids,
#         water_removed, no_cols, inpar, E_col_grid, data_paths['output'])
