#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Jan Bohrer
Configuration file for grid and particle generation.
The corresponding script is "generate_grid_and_particles.py".
"""

import numpy as np

#%% SET PARAMETERS FOR THE GRID AND PARTICLE GENERATION IN DICTIONARY

genpar = \
{
### RANDOM NUMBER GENERATION
# random number generator seed for inital particle generation
# this number is overwritten, if the script is executed with an argument:
# In the following example,
# the parameter 'seed_SIP_gen'would be overwritten to 4711:
# "python3 generate_grid_and_particles.py 4711"
'seed_SIP_gen'      : 1001,
# set, if the number generator should be reseeded at every z-level:
# 'False' is usually fine. 'True' is usually not necessary for the given setup.
'reseed'            : False, 

### DATA PATH
# path to the parent directory, where data of the grid and simulations shall
# be stored. in here, the program will create subdirectories automatically.
#'simdata_path'      : '/Users/bohrer/sim_data_cloudMP/',
#'simdata_path'      : '/Users/bohrer/sim_data_cloudMP_TEST191216/',
'simdata_path'          : '/Users/bohrer/sim_data_cloudMP_TEST200108/',

### GRID AND INITIAL ATMOSPHERIC PROFILE
# domain sizes (2D)
'x_min'             : 0.,
'x_max'             : 1500.,
'z_min'             : 0.,
'z_max'             : 1500.,

# spatial step sizes
# the number of grid cells is calculated from step sizes and domain sizes
'dx'                : 150.,
'dz'                : 150.,
#'dx'                : 20.,
#'dz'                : 20.,
'dy'                : 1., # dy=1 is default in the 2D setup

# initial thermodynamic environment 
'p_0'               : 1015E2, # surface pressure in Pa
'p_ref'             : 1E5, # ref pressure for potential temperature in Pa
# total water mixing ratio (initial. constant over whole domain in setup)
'r_tot_0'           : 7.5E-3, # kg water / kg dry air 
# liquid potential temperature (initial. constant over whole domain in setup)
'Theta_l'           : 289., # K

### PARTICLES
# solute material in the CCNs (possible is ammon. sulf. "AS" or "NaCl")
'solute_type'       : 'AS',

# initial number of super particles per cell and mode (avg. values)
# list: [mode0, mode1, mode2...]
#'no_spcm'           : [4,0],
'no_spcm'           : [2,2],
## particle size distribution
# distribution type (only "lognormal" available)
'dist'              : "lognormal", 
# droplet number concentration: number density of particles per mode
# array: [mode0, mode1, mode2...]
'DNC0'              : np.array( [60.0E6, 40.0E6] ), # 1/m^3
# parameters of radial (R_s) lognormal dry size distribution:
# f_R(R_s) = \sum_k \frac{DNC_k}{\sqrt{2 \pi} \, \ln (\sigma_k) R_s}
# \exp [ -(\frac{\ln (R_s / \mu_k)}{\sqrt{2} \, \ln (\sigma_k)} )^2 ]
# -> will be converted to mass parameters in the process
# array: [mode0, mode1, mode2...]
'mu_R'              : 0.5 * np.array( [0.04, 0.15] ),
'sigma_R'           : np.array( [1.4, 1.6] ),
# SingleSIP init method parameters (cf. Unterstrasser 2017, GMD (10), p.1521)
#'eta'               : 6E-10,
'eta'               : 1E-10,
'eta_threshold'     : "fix",
#'eta_threshold'     : "weak",
'r_critmin'         : np.array([1., 3.]) * 1E-3, # mu, # mode 1, mode 2, ..
'm_high_over_m_low' : 1.0E8,

# exponential distribution not yet implemented
# set DNC0 manually only for lognormal distr.
#elif distribution == "expo":
#    LWC0 = 1.0E-3 # kg/m^3
#    R_mean = 9.3 # in mu
#    r_critmin = 0.6 # mu
#    m_high_over_m_low = 1.0E6    
#
#    rho_w = 1E3
#    m_mean = compute_mass_from_radius_jit(R_mean, rho_w) # in 1E-18 kg
#    DNC0 = 1E18 * LWC0 / m_mean # in 1/m^3
#    # we need to hack here a little because of the units of m_mean and LWC0
#    LWC0_over_DNC0 = m_mean
#    DNC0_over_LWC0 = 1.0/m_mean
#
#    dst_par = (DNC0, DNC0_over_LWC0)

### SATURATION ADJUSTMENT PARAMETERS
'S_init_max'        : 1.04, # upper cap threshold for saturation during init.
'dt_init'           : 0.1, # s  , time step for the sat. adjust. condensation
# number of iterations for the root finding
# Newton algorithm during mass condensation with the implicit method
'Newton_iterations' : 2,
# maximal allowed iter counts in initial particle water take up to equilibrium
# for sum(no_spcm) ~= 50, a value of iter_cnt_limit=800 should be fine
# for smaller no_spcm, a higher number might be necessary to reach EQ
'iter_cnt_limit'     : 4000,

### LOGGING
# if True: std out is written to file 'std_out.log' inside the save path
# if False: std out written to console
'set_std_out_file'  : True

# IN WORK: decide here, if additional plots of the initial grid shall be
# generated directly after creation.
}        