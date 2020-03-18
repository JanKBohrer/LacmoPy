#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in a two-dimensional kinematic framework
Test Case 1, ICMW 2012, Muhlbauer et al. 2013, ‎Bull. Am. Meteorol. Soc. 94, 25

Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

CONFIGURATION FILE

the corresponding execution script is 'lacmo.py'

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ('mu')
all other quantities in SI units
"""

import numpy as np
from grid import compute_no_grid_cells_from_step_sizes

config = {
# path to parent directory for data output
'paths' : {
#    'simdata' : '/vols/fs1/work/bohrer/sim_data_cloudMP/'
#    'simdata' : '/Users/bohrer/sim_data_cloudMP/'
    'simdata' : '/home/jdesk/sim_data_cloudMP/'
},

'generate_grid'         : True,
#'generate_grid'         : False,
# spin-up: no gravity, no collisions, no relaxation
'execute_spin_up'       : True,
#'execute_spin_up'       : False,
'execute_simulation'    : True,
# set 'True' when starting from a spin-up state
# provides the opportunity to simulate directly without spin-up
# in this case, set 'execute_spin_up' and 'spin_up_complete' to 'False'
'spin_up_complete'      : False,
#'spin_up_complete'      : True,

# set 'False', if spin_up shall be executed
# set 'False', if the simulation starts from an existing spin-up state.
# set 'True', if a simulation is continued from a simulation state, 
# which was stored at t_start_sim given below.
'continued_simulation'  : False,
#'continued_simulation'  : True,

# collisions during simulation phase
'act_collisions'        : True,
# relaxation source term for r_v and Theta towards init. profiles
'act_relaxation'        : True,

#%% RANDOM NUMBER GENERATION SEEDS

# seeds are overwritten, if LaCMo.py is executed with arguments:
# 'python3 LaCMo.py 4711 5711'
# will overwrite 'seed_SIP_gen' by 4711 and 'seed_sim' by 5711
# random number generator seed for inital particle generation
'seed_SIP_gen'      : 1001,
# random number generator seed for particle collisions
'seed_sim'          : 1001,

#%% GRID AND PARTICLE GENERATION

# set, if the number generator should be reseeded at every z-level:
# 'False' is usually fine. 'True' is usually not necessary for the given setup.
'reseed'            : False, 

### GRID AND INITIAL ATMOSPHERIC PROFILE
# domain sizes (2D) [ [x_min, x_max], [z_min, z_max] ] in meter
'grid_ranges'       : [[0., 1500.], [0., 1500.]],

# spatial step sizes [dx, dz] in meter
# the number of grid cells is calculated from step sizes and domain sizes
'grid_steps'        : [150., 150.],# in meter
'dy'                : 1., # in meter, dy=1 is default in the 2D setup

# initial thermodynamic environment 
'p_0'               : 1015E2, # surface pressure in Pa
'p_ref'             : 1E5, # ref pressure for potential temperature in Pa
# total water mixing ratio (initial. constant over whole domain in setup)
'r_tot_0'           : 7.5E-3, # kg water / kg dry air 
# liquid potential temperature (initial. constant over whole domain in setup)
'Theta_l'           : 289., # K

### PARTICLES
# solute material of the CCNs
# options: 'AS' (ammon. sulf.) or 'NaCl'
'solute_type'       : 'NaCl',

# initial number of super particles per cell and mode (avg. values)
# list: [mode0, mode1, mode2...]
'no_spcm'           : [3, 3],
### particle dry size distribution
# distribution type (only 'lognormal' available)
'dist'              : 'lognormal', 
# droplet number concentration: number density of particles per mode
# array: [mode0, mode1, mode2...]
'DNC0'              : np.array( [60.0E6, 40.0E6] ), # 1/m^3
#'DNC0'              : np.array( [30.0E6, 20.0E6] ), # 1/m^3
# parameters of radial (R_s) lognormal dry size distribution:
# f_R(R_s) = \sum_k \frac{DNC_k}{\sqrt{2 \pi} \, \ln (\sigma_k) R_s}
# \exp [ -(\frac{\ln (R_s / \mu_k)}{\sqrt{2} \, \ln (\sigma_k)} )^2 ]
# -> will be converted to mass parameters in the process
# array: [mode0, mode1, mode2...]
'mu_R'              : 0.5 * np.array( [0.04, 0.15] ),
'sigma_R'           : np.array( [1.4, 1.6] ),
# SingleSIP init method parameters (cf. Unterstrasser 2017, GMD 10: 1521)
#'eta'               : 6E-10,
'eta'               : 1E-10,
'eta_threshold'     : 'fix',
#'eta_threshold'     : 'weak',
'r_critmin'         : np.array([1., 3.]) * 1E-3, # mu, # mode 1, mode 2, ..
'm_high_over_m_low' : 1.0E8,

### SATURATION ADJUSTMENT PARAMETERS
'S_init_max'        : 1.04, # upper cap threshold for saturation during init.
'dt_init'           : 0.1, # s  , time step for the sat. adjust. condensation
# number of iterations for the implicit Newton algorithm
# during mass condensation with the implicit method
'Newton_iterations' : 2,
# maximal allowed iter counts in initial particle water take up to equilibrium
# for sum(no_spcm) ~= 50, a value of iter_cnt_limit=1000 should be fine
# for smaller no_spcm, a higher number might be necessary to reach EQ
'iter_cnt_limit'     : 4000,

#%% SIMULATION PARAMETERS

# SIMULATION TIME AND INTEGRATION PARAMETERS
# time for reading the initial atmospheric profiles (used in the relax. term)
't_init'                : 0,
# start time of the simulation (seconds)
# when starting from a freshly generated grid, this must be = 0
# when starting from a saved state (e.g. from a spin up state),
# this must correspond to the time of the saved data
# for a direct simulation without spin-up:
# set t_start_spin_up = 0, t_end_spin_up = 0, t_start_sim = 0
't_start_spin_up'        : 0, # seconds
't_end_spin_up'          : 600, # seconds
't_start_sim'            : 600, # seconds
't_end_sim'              : 900, # seconds

# advection time step (seconds),
# this is the largest time step of the simulation
# the refined time steps of condensation and collision are scaled accordingly
'dt_adv'                : 1,
# number of condensation steps per advection step (only even integers possible)
# a 'condensation step' includes particle mass growth and particle propagation
'no_cond_per_adv'       : 10,
# number of particle collisions steps per advection step
# possible values: 1, 2 OR no_cond_per_adv
'no_col_per_adv'        : 2,
# number of Newton iterations in the implicit mass condensation algo.
'no_iter_impl_mass'     : 3,

### DATA AND TRACER STORAGE
'frame_every'           : 300, # full data output every X time steps 'dt_adv'
# trace_ids can either be an integer or an array of ints
# if integer: ids are spread evenly over the whole amount of particles
# if array: this specific list of ids is used
'trace_ids'             : 80,
'dump_every'            : 10, # tracer data output every X time steps 'dt_adv'

### COLLECTION KERNEL
# Long kernel modified by Bott (1997), J Atmos Sci 55, p.2284
#'kernel_type'           : 'Long_Bott', 
# Hall kernel modified by Bott (1997), J Atmos Sci 55, p.2284
 'kernel_type'         : 'Hall_Bott',
# constant collision kernel, set value below
# 'kernel_type'         : 'Hydro_E_const',
# interpolation method for the kernel,
# only option is 'Ecol_grid_R' (depending on radius)
'kernel_method'         : 'Ecol_grid_R',
# this value must always be set to some float.
# however, it is only used for kernel_type = 'Hydro_E_const'
'E_col_const'           : 0.5,
# this folder should be in the same directory,
# from which the program is executed
'save_folder_Ecol_grid' : 'Ecol_grid_data',

### LOGGING
# if True: std out is written to file 'std_out.log' inside the save path
# if False: std out written to console
'set_std_out_file'      : True
#'set_std_out_file'     : False
}

#%% DERIVED QUANTITIES (DO NOT SET MANUALLY)
config['no_cells'] = compute_no_grid_cells_from_step_sizes(
                         config['grid_ranges'], config['grid_steps'])
