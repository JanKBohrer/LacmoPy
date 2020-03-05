
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinetic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

MAIN SIMULATION SCRIPT

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
### BUILT IN
import sys

### CUSTOM MODULES
from file_handling import load_grid_and_particles_full
from integration import simulate
from init import set_initial_sim_config

#%% IMPORT PARAMETERS FROM CONFIG FILE
from config_sim import simpar

# overwrite generation seed and simulation seed, when given by argument
if len(sys.argv) > 1:
    simpar['seed_SIP_gen'] = int(sys.argv[1])
if len(sys.argv) > 2:
    simpar['seed_sim'] = int(sys.argv[2])

#%% SPIN UP, IF REQUESTED

if simpar['execute_spin_up']:
    sim_mode = simpar['simulation_mode']
    simpar['simulation_mode'] = "spin_up"
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
             water_removed, no_cols, simpar, E_col_grid, data_paths)
    simpar['simulation_mode'] = sim_mode
    simpar['spin_up_complete'] = True

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
             water_removed, no_cols, simpar, E_col_grid, data_paths)
