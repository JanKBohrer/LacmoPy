#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinetic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

MAIN SCRIPT

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
### BUILT IN
import sys

### CUSTOM MODULES
from init import set_config
from init import initialize_grid_and_particles_SinSIP
from integration import simulate
from file_handling import load_grid_and_particles_full

from config import config

# overwrite generation seed and simulation seed, if given by arguments
if len(sys.argv) > 1:
    config['seed_SIP_gen'] = int(sys.argv[1])
if len(sys.argv) > 2:
    config['seed_sim'] = int(sys.argv[2])
    
#%% GENERATE GRID AND PARTICLES, IF REQUESTED

if config['generate_grid']:
    config_mode = 'generation'

    # set stdout file, if requested
    set_config(config, config_mode)
    if config['set_std_out_file']:
        sys.stdout = open( config['paths']['grid'] + 'std_out.log', 'a')
    
    grid, pos, cells, cells_comb, vel, m_w, m_s, xi, active_ids =\
        initialize_grid_and_particles_SinSIP(config)
    
#%% SPIN UP, IF REQUESTED

if config['execute_spin_up']:
    config_mode = 'spin_up'
    act_collisions_sim = config['act_collisions']
    act_relaxation_sim = config['act_relaxation']
    E_col_grid, no_cols, water_removed =\
        set_config(config, config_mode)
    
    # set stdout file, if requested
    if config['set_std_out_file']:
        sys.stdout = open( config['paths']['output'] + 'std_out.log', 'a')
    if len(sys.argv) > 1:
        print('seed_SIP_gen set via argument =', config['seed_SIP_gen'])        
    
    grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
        load_grid_and_particles_full(config['t_start'],
                                     config['paths']['grid'])
    
    simulate(grid, pos, vel, cells, m_w, m_s, xi, active_ids,
             water_removed, no_cols, config, E_col_grid)
    
    config['act_collisions'] = act_collisions_sim
    config['act_relaxation'] = act_relaxation_sim
    config['spin_up_complete'] = True

#%% MAIN SIMULATION    

if config['execute_simulation']: 
    config_mode = 'simulation'
    E_col_grid, no_cols, water_removed =\
        set_config(config, config_mode)
    
    # set stdout file, if requested
    if config['set_std_out_file']:
        sys.stdout = open( config['paths']['output'] + 'std_out.log', 'a')
    if len(sys.argv) > 1:
        print('seed_SIP_gen set via argument =', config['seed_SIP_gen'])
    if len(sys.argv) > 2:
        print('seed_sim set via argument=', config['seed_sim'])
        
    grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
        load_grid_and_particles_full(config['t_start'],
                                     config['paths']['grid'])
    
    simulate(grid, pos, vel, cells, m_w, m_s, xi, active_ids,
             water_removed, no_cols, config, E_col_grid)