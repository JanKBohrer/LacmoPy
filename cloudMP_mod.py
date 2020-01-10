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

#%% SPIN UP, IF REQUESTED

if simpar['execute_spin_up']:

    sim_mode = simpar['simulation_mode']
    simpar['simulation_mode'] = "spin_up"
#    simpar['t_start'] = simpar['t_start_spin_up']
#    simpar['t_end'] = simpar['t_end_spin_up']
    data_paths, E_col_grid, no_cols, water_removed =\
        set_initial_sim_config(simpar)
    # set stdout file, if requested
    if simpar['set_std_out_file']:
        sys.stdout = open( data_paths['output'] + "std_out.log", 'w')
    if len(sys.argv) > 1:
        print("seed_SIP_gen set via argument =", simpar['seed_SIP_gen'])        
    grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
        load_grid_and_particles_full(simpar['t_start'], data_paths['grid'])
    simulate(grid, pos, vel, cells, m_w, m_s, xi, active_ids,
             water_removed, no_cols, simpar, E_col_grid, data_paths['output'])
    simpar['simulation_mode'] = sim_mode
    simpar['spin_up_complete'] = True
    
#if simpar['simulation_mode'] == "with_collision_spin_up_included":
#    simpar['simulation_mode'] = "spin_up"
#    data_paths, E_col_grid, no_cols, water_removed =\
#        set_initial_sim_config(simpar)
#    # set stdout file, if requested
#    if simpar['set_std_out_file']:
#    #    sys.stdout = open(confpar.save_path + "std_out.log", 'w')
#        sys.stdout = open( data_paths['output'] + "std_out.log", 'w')
#    grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
#        load_grid_and_particles_full(simpar['t_start'], data_paths['grid'])
#    simulate(grid, pos, vel, cells, m_w, m_s, xi, active_ids,
#             water_removed, no_cols, simpar, E_col_grid, data_paths['output'])
#    simpar['simulation_mode'] = "with_collision"
#    simpar['simulation_mode'] = "with_collision"
    

#%% MAIN SIMULATION    

if simpar['execute_simulation']: 

    data_paths, E_col_grid, no_cols, water_removed =\
        set_initial_sim_config(simpar)
    # set stdout file, if requested
    if simpar['set_std_out_file']:
        sys.stdout = open( data_paths['output'] + "std_out.log", 'w')
    if len(sys.argv) > 1:
        print("seed_SIP_gen set via argument =", simpar['seed_SIP_gen'])
    if len(sys.argv) > 2:
        print("seed_sim set via argument=", simpar['seed_sim'])
        
    grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
        load_grid_and_particles_full(simpar['t_start'], data_paths['grid'])
    
    simulate(grid, pos, vel, cells, m_w, m_s, xi, active_ids,
             water_removed, no_cols, simpar, E_col_grid, data_paths['output'])

