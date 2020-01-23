#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jan Bohrer
Configuration file for the Lagrangian cloud model simulation.
The corresponding simulation script is "cloudMP.py".
"""

#%% SET SIMULATION PARAMETERS IN DICTIONARY

simpar = \
{
### DATA PATH
# path to the parent directory, which was chosen for storing the grid data.
# the program will create subdirectories automatically, where simulation
# data will written.
#'simdata_path'      : '/Users/bohrer/sim_data_cloudMP/',
#'simdata_path'          : '/Users/bohrer/sim_data_cloudMP_TEST200108/',
'simdata_path'          : '/Users/bohrer/sim_data_cloudMP_ab_Jan20/',
#'simdata_path'          : '/vols/fs1/work/bohrer/sim_data_cloudMP_ab_Jan20/',

### RANDOM NUMBER GENERATION
# random number generator seeds for inital particle generation and simulations
# this number is overwritten, IF the script is executed with arguments:
# In the following example, the parameters 'seed_SIP_gen' and 'seed_sim'
# would be overwritten to 4711 and 5711:
# "python3 generate_grid_and_particles.py 4711 5711"
'seed_SIP_gen'          : 1001,
'seed_sim'              : 1001,

### GRID AND PARTICLE PARAMETERS
# need to set number of grid cells [x,z] for data path assignments
#'no_cells'              : [15,15],
#'no_cells'              : [50,50],
'no_cells'              : [75,75],

# solute material in the CCNs (possible is ammon. sulf. "AS" or "NaCl")
#'solute_type'           : 'AS',
'solute_type'           : 'NaCl',

# number of super particles per cell and mode as list [mode0, mode1, ..]
#'no_spcm'               : [2,2],
#'no_spcm'               : [6,8],
#'no_spcm'               : [10,14],
'no_spcm'               : [16,24],
#'no_spcm'               : [26,38],

### SIMULATION TYPE
# "wo_collisions": with gravity, no collisions
# "with_collisions": with gravity, with collisions
'simulation_mode'       : 'with_collisions',
#'simulation_mode'      : 'wo_collisions',

# "spin_up": no gravity, no collisions
# "spin_up" is executed before "simulation"
# "spin_up" can also be executed without subsequent simulation
# "simulation" can also be executed without preceding spin-up OR
# from an existing spin_up state
'execute_spin_up'       : True,
#'execute_spin_up'       : False,

'execute_simulation'    : True,
#'execute_simulation'    : False,

# set True, if a "simulation" is continued from a sim state, which was stored
# at t_start_sim given below. This is not meant for an existing spin-up. I.e., 
# if the simulation starts from an existing spin-up state, set this to False
#'continued_simulation'   : True,
'continued_simulation'  : False,

# activate the relaxation source term for r_v and Theta towards init. profiles
'act_relaxation'    : False,
#'act_relaxation'    : True,

# set this to "True" when starting from a spin-up state,
# will automatically be overwritten to "False" during the spin-up phase,
# if "execute_spin_up" = True
# this value gives the opportunity to simulate directly without spin-up
# in this case, set "execute_spin_up" False and "spin_up_complete" False
#'spin_up_complete'        : True,
'spin_up_complete'        : False,

### SIMULATION TIME AND INTEGRATION PARAMETERS
# time for reading the initial atmospheric profiles (used in the relax. term)
't_init'                : 0,
# start time of the simulation (seconds)
# when starting from a freshly generated grid, this must = 0
# when starting from a saved state (e.g. from a spin up state),
# this must correspond to the time of the saved data
# for a direct simulation without spin-up:
# set t_start_spin_up = 0, t_end_spin_up = 0, t_start_sim = 0

#'t_start_spin_up'        : 0, # seconds
#'t_end_spin_up'          : 3600, # seconds
#'t_start_sim'            : 3600, # seconds
#'t_end_sim'              : 7200, # seconds

#'t_start_spin_up'        : 0, # seconds
#'t_end_spin_up'          : 300, # seconds
#'t_start_sim'            : 300, # seconds
#'t_end_sim'              : 600, # seconds

't_start_spin_up'        : 0, # seconds
't_end_spin_up'          : 7200, # seconds
't_start_sim'            : 7200, # seconds
't_end_sim'              : 10800, # seconds

#'t_start_sim'            : 7200, # seconds
#'t_end_sim'              : 10800, # seconds
#'t_start_sim'            : 600, # seconds
#'t_end_sim'              : 900, # seconds

# advection time step (seconds),
# this is the largest time step of the simulation
# the refined time steps of condensation and collision are scaled with this ts
'dt_adv'                : 1, # seconds
# number of condensation steps per advection step.
# a "condensation step" includes particle mass growth and particle propagation
'no_cond_per_adv'       : 10,
# number of particle collisions steps per advection step
'no_col_per_adv'        : 2,
# number of Newton iterations in the implicit mass condensation algo.
'no_iter_impl_mass'     : 3,

### DATA AND TRACER STORAGE
'frame_every'           : 300, # full data output every X time steps "dt_adv"
# trace_ids can either be an integer or an array of ints
# if integer: ids are spread evenly over the whole amount of particles
# if array: this specific list of ids is used
'trace_ids'             : 40, # (80 = default)
'dump_every'            : 10, # tracer data output every X time steps "dt_adv"

### COLLISION KERNEL
# Long kernel modified by Bott (1997), J Atmos Sci 55, p.2284
'kernel_type'           : 'Long_Bott', 
# Hall kernel modified by Bott (1997), J Atmos Sci 55, p.2284
# 'kernel_type'         : 'Hall_Bott',
# constant collision kernel, set value below
# 'kernel_type'         : 'Hydro_E_const',
# interpolation method for the kernel, only available is "Ecol_grid_R"
'kernel_method'         : 'Ecol_grid_R',
# this value must be set to float,
# but is only used for kernel_type = 'Hydro_E_const'
'E_col_const'           : 0.5,
# this folder should be in the same directory,
# from which the program is executed
'save_folder_Ecol_grid' : 'Ecol_grid_data',
#'save_dir_Ecol_grid': f'Ecol_grid_data/{kernel_type}/',

### LOGGING
# if True: std out is written to file 'std_out.log' inside the save path
# if False: std out written to console
'set_std_out_file'      : True
#'set_std_out_file'     : False
}