#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  6 16:35:52 2019

@author: bohrer
"""

#%% MODULE IMPORTS
### BUILT IN
import os
import sys
import math
import numpy as np

### CUSTOM MODULES
import constants as c
from file_handling import load_grid_and_particles_full
from integration import simulate

#%% IMPORT PARAMETERS FROM COMFIG FILE

import config_sim as confpar

sys.stdout = open(confpar.save_path + "std_out.log", 'w')

#%% INIT GRID AND PARTICLES

grid, pos, cells, vel, m_w, m_s, xi, active_ids = \
    load_grid_and_particles_full(confpar.t_start,
                                 confpar.simdata_path + confpar.grid_folder)

#%% SIMULATION    
#simulate(grid, pos, vel, cells, m_w, m_s, xi,
#         confpar.solute_type, confpar.water_removed, confpar.active_ids,
#         confpar.dt, confpar.dt_col, confpar.scale_dt_cond,
#         confpar.no_col_per_adv,
#         confpar.t_start, confpar.t_end, confpar.no_iter_impl_mass,
#         confpar.g_set, confpar.act_collisions,
#         confpar.frame_every, confpar.dump_every, confpar.trace_ids, 
#         confpar.E_col_grid, confpar.no_kernel_bins,
#         confpar.R_kernel_low_log, confpar.bin_factor_R_log,
#         confpar.kernel_type, confpar.kernel_method,
#         confpar.no_cols, confpar.seed_sim,
#         confpar.save_path, confpar.simulation_mode)     