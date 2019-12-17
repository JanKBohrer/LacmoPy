#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:07:21 2019

@author: jdesk
"""

#%% MODULE IMPORTS

import os
import numpy as np
import sys

from grid import compute_no_grid_cells_from_step_sizes

from init import initialize_grid_and_particles_SinSIP

from microphysics import compute_mass_from_radius_vec
#from microphysics import compute_mass_from_radius_jit
import constants as c

from config_grid import genpar
from init import set_initial_gen_config

if len(sys.argv) > 1:
    genpar['seed_SIP_gen'] = int(sys.argv[1])

data_paths = set_initial_gen_config(genpar)

if genpar['set_std_out_file']:
    sys.stdout = open(data_paths['grid'] + "std_out.log", 'w')

#    if inpar['distribution'] == "expo":
#        print("dist = expo", f"DNC0 = {DNC0:.3e}", "LWC0 =", LWC0,
#              "m_mean = ", m_mean)
#    elif inpar['distribution'] == "lognormal":
#        print("dist = lognormal", "DNC0 =", DNC0,
#              "mu_R =", mu_R, "sigma_R =", sigma_R,
#               "r_critmin =", r_critmin,
#               "m_high_over_m_low =", m_high_over_m_low)
#print("no_modes, idx_mode_nonzero:", no_modes, ",", idx_mode_nonzero)

#%% GENERATE GRID AND PARTICLES

grid, pos, cells, cells_comb, vel, m_w, m_s, xi, active_ids  = \
    initialize_grid_and_particles_SinSIP(genpar, data_paths['grid'])
        
#grid, pos, cells, cells_comb, vel, m_w, m_s, xi, active_ids  = \
#    initialize_grid_and_particles_SinSIP(
#        genpar,
#        
#        x_min, x_max, z_min, z_max, dx, dy, dz,
#        p_0, p_ref, r_tot_0, Theta_l, solute_type,
#        DNC0, no_spcm, no_modes, dist, dst_par,
#        eta, eta_threshold, r_critmin, m_high_over_m_low,
#        seed_SIP_gen, reseed,
#        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, grid_path)

#%% PLOT PARTICLES WITH VELOCITY VECTORS

#fig_name = grid_path + "pos_vel_t0.png"
#plot_pos_vel_pt(pos, vel, grid,
#                    figsize=(8,8), no_ticks = [11,11],
#                    MS = 1.0, ARRSCALE=10, fig_name=fig_name)

#grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids =\
#    initialize_grid_and_particles(
#        x_min, x_max, z_min, z_max, dx, dy, dz,
#        p_0, p_ref, r_tot_0, Theta_l,
#        n_p, no_spcm, dst, dst_par, 
#        P_min, P_max, r0, r1, dr, rnd_seed, reseed,
#        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, path)

#%% simple plots
# IN WORK: ADD AUTOMATIC CREATION OF DRY SIZE SPECTRA COMPARED WITH EXPECTED PDF
# FOR ALL MODES AND SAVE AS FILE 
# grid.plot_thermodynamic_scalar_fields_grid()