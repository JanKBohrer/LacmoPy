#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:40:09 2019

@author: bohrer
"""

import os
#import math
import numpy as np

import constants as c

from file_handling import load_kernel_data

#%% as dict

simpar = \
{
'seed_SIP_gen'      : 1011,
'seed_sim'          : 2011,
'simdata_path'      : '/Users/bohrer/sim_data_cloudMP/',
#simdata_path = '/Users/bohrer/sim_data_cloudMP_TEST191206/'

#no_cells = [75,75]
'no_cells'          : [5,5],

'solute_type'       : 'AS',

#no_spcm = [26,38]
'no_spcm'           : [2,2],

#simulation_mode = 'with_collision'
'simulation_mode'   : 'spin_up',
#simulation_mode = 'wo_collision'

# set True when starting from a spin-up state,
# this will be set to False below, if the simulation mode is 'spin_up'
#'spin_up_before'    : True,
'spin_up_before'    : False,

't_start'           : 0,
#t_end = 7200
't_end'             : 300,
'dt_adv'            : 1,
'no_cond_per_adv'   : 10,
'no_col_per_adv'    : 2,
# RENAME to no_iter_impl_mass or similar
'no_iter_impl_mass' : 2,
#frame_every = 300
'frame_every'       : 100,

# trace_ids can either be an integer or an array of ints
# if integer: ids are spread evenly over the whole amount of particles
# if array: this specific list of ids is used
#trace_ids = 80
'trace_ids'         : 20,
'dump_every'        : 10,

### collision kernel parameters
'kernel_type'       : 'Long_Bott',
# kernel_type = 'Hall_Bott'
# kernel_type = 'Hydro_E_const'
'kernel_method'     : 'Ecol_grid_R',
# value must be set, but is only used for kernel_method = 'Ecol_const',
'E_col_const'       : 0.5,
# this folder should be in the same directory,
# from which the program is executed
'save_folder_Ecol_grid': 'Ecol_grid_data',
#'save_dir_Ecol_grid': f'Ecol_grid_data/{kernel_type}/',

# if True: std out is written to file 'std_out.log' inside the save path
# if False: std out written to console
'set_std_out_file'  : True
}        