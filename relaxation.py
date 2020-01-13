#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 11:40:20 2020

@author: bohrer
"""

import math
import numpy as np


from grid import Grid

def compute_relaxation_time_profile(z):
    return 300 * np.exp(z/200)

# field is an 2D array(x,z)
# profile0 is an 1D profile(z)
# t_relax is the relaxation time profile(z)
# dt is the time step
# return: relaxation source term profile(z) for one time step dt
def compute_relaxation_term(field, profile0, t_relax, dt):
    return dt * (profile0 - np.average(field, axis = 0)) / t_relax

#%%

#dx = 150    
#dy = 1
#dz = 150
#
#grid_ranges = [[0,1500],[0,1500]]
#grid_steps = [dx, dz]
#
#grid = Grid(grid_ranges, grid_steps, dy)
#
#grid.print_info()

#%%

from init import set_initial_sim_config

#%% IMPORT PARAMETERS FROM CONFIG FILE
from config_sim import simpar
    
from file_handling import load_grid_and_particles_full


simpar['simulation_mode'] = "spin_up"
data_paths, E_col_grid, no_cols, water_removed =\
    set_initial_sim_config(simpar)

grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
    load_grid_and_particles_full(simpar['t_start'], data_paths['grid'])

#grid.plot_thermodynamic_scalar_fields()

grid_centers_z = grid.centers[1][0]
relaxation_time_profile = compute_relaxation_time_profile(grid_centers_z)

init_rv = np.copy(grid.mixing_ratio_water_vapor)
init_profile_r_v = np.average( grid.mixing_ratio_water_vapor, axis = 0 )
init_profile_Theta = np.average( grid.potential_temperature, axis = 0 )

grid.mixing_ratio_water_vapor += 1E-3

r_v = grid.mixing_ratio_water_vapor
Theta = grid.potential_temperature

r_v_mean = np.average(r_v, axis = 0)

relax_term = compute_relaxation_term(r_v, init_profile_r_v,
                                     relaxation_time_profile, simpar['dt_adv'])

print(init_profile_r_v)
print(np.average(r_v, axis = 0))
print(relax_term)
print(relax_term/r_v_mean)


#%% RELAXATION TEST LOOP

r_v = grid.mixing_ratio_water_vapor
Theta = grid.potential_temperature

no_steps = 36000
save_every = 3600

r_v_vs_t = []

for step in range(no_steps):
    if step%save_every == 0:
        r_v_vs_t.append(np.copy(r_v))
    relax_term = compute_relaxation_term(
                     r_v, init_profile_r_v,
                     relaxation_time_profile, 1)
#    relax_term = compute_relaxation_term(
#                     r_v, init_profile_r_v,
#                     relaxation_time_profile, simpar['dt_adv'])
    # this works with numpy, because r_v[x,z] has "z" as second axis
    # and relax_term[z] is subtracted from each column of r_v
    # i.e. for each fixed "x", the same vector relax_term[z] is subtracted
    r_v += relax_term
    

for n in range(len(r_v_vs_t)):
    grid.plot_scalar_field_2D(r_v_vs_t[n] - init_rv)
    
    
    
    