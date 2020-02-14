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