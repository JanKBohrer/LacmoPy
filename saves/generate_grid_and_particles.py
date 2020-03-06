#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

GRID AND PARTICLE GENERATION
Set parameters in the corresponding config file "config_grid.py".

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
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
    
    