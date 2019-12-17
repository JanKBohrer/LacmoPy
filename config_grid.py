#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 16:40:09 2019

@author: bohrer
"""

import numpy as np

#%% as dict

genpar = \
{
'seed_SIP_gen'      : 1013,
# set, if the number generator is reseeded at every z-level:
# 'False' is usually fine. 'True' is usually not necessary for the given setup
'reseed'            : False, 

#'simdata_path'      : '/Users/bohrer/sim_data_cloudMP/',
'simdata_path'         : '/Users/bohrer/sim_data_cloudMP_TEST191216/',

'solute_type'       : 'AS',

#'no_spcm'           : [4,0],
'no_spcm'           : [0,4],

'x_min'             : 0.,
'x_max'             : 1500.,
'z_min'             : 0.,
'z_max'             : 1500.,

'dx'                : 150.,
'dz'                : 150.,
#'dx'                : 20.,
#'dz'                : 20.,
'dy'                : 1.,

'p_0'                : 1015E2, # surface pressure in Pa
'p_ref'             : 1E5, # ref pressure for potential temperature in Pa
# total water mixing ratio (constant over whole domain in setup)
'r_tot_0'           : 7.5E-3, # kg water / kg dry air 
# liquid potential temperature (constant over whole domain in setup)
'Theta_l'           : 289., # K

'dist'      : "lognormal",

'r_critmin'         : np.array([1., 3.]) * 1E-3, # mu, # mode 1, mode 2, ..
'm_high_over_m_low' : 1.0E8,
# droplet number concentration 
# set DNC0 manually only for lognormal distr.
# number density of particles mode 1 and mode 2:
'DNC0'              : np.array([60.0E6, 40.0E6]), # 1/m^3
# parameters of radial lognormal distribution -> converted to mass
# parameters below
'mu_R'              : 0.5 * np.array( [0.04, 0.15] ),
'sigma_R'           : np.array( [1.4,1.6] ),

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

# SingleSIP init method parameters (cf. Unterstrasser 2017)
#eta = 6E-10
'eta'               : 1E-10,
'eta_threshold'     : "fix",
#'eta_threshold'     : "weak",

# Saturation adjustment phase parameters
'S_init_max'        : 1.04,
'dt_init'           : 0.1, # s
# number of iterations for the root finding
# Newton algorithm in the implicit method
'Newton_iterations' : 2,
# maximal allowed iter counts in initial particle water take up to equilibrium
'iter_cnt_limit'     : 2000,

# if True: std out is written to file 'std_out.log' inside the save path
# if False: std out written to console
'set_std_out_file'  : True
}        