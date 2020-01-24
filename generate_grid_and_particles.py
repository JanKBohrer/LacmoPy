#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jan Bohrer
Script for grid and particle generation.
Set parameters in the corresponding config file "config_grid.py".
"""

#%% MODULE IMPORTS

import sys

from init import initialize_grid_and_particles_SinSIP
from init import set_initial_gen_config
from config_grid import genpar

if len(sys.argv) > 1:
    genpar['seed_SIP_gen'] = int(sys.argv[1])

data_paths = set_initial_gen_config(genpar)

if genpar['set_std_out_file']:
    sys.stdout = open(data_paths['grid'] + "std_out.log", 'w')

#%% GENERATE GRID AND PARTICLES

grid, pos, cells, cells_comb, vel, m_w, m_s, xi, active_ids  = \
    initialize_grid_and_particles_SinSIP(genpar, data_paths['grid'])

#%% PLOT PARTICLES WITH VELOCITY VECTORS

#fig_name = grid_path + "pos_vel_t0.png"
#plot_pos_vel_pt(pos, vel, grid,
#                    figsize=(8,8), no_ticks = [11,11],
#                    MS = 1.0, ARRSCALE=10, fig_name=fig_name)

#%% SIMPLE PLOTS
# IN WORK: ADD AUTOMATIC CREATION OF DRY SIZE SPECTRA COMPARED WITH EXPECT. PDF
# FOR ALL MODES AND SAVE AS FILE 
# grid.plot_thermodynamic_scalar_fields_grid()