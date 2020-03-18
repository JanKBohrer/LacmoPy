#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

GRID AND PARTICLE INITIALIZATION

for particle initialization, the "SingleSIP" method is applied, as proposed by
Unterstrasser 2017, GMD 10: 1521–1548

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
import os
import numpy as np

import constants as c
from grid import Grid
from grid import interpolate_velocity_from_cell_bilinear

from integration import \
    compute_dml_and_gamma_impl_Newton_full

import material_properties as mat
import atmosphere as atm
import microphysics as mp
from generation_SIP_ensemble import \
    gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl    

from file_handling import save_grid_and_particles_full
from file_handling import load_kernel_data_Ecol

#%% CONFIG FILE PROCESSING

def set_config(config, config_mode):
    config['no_spcm'] = np.array(config['no_spcm'])
    no_spcm = config['no_spcm']
    no_cells = config['no_cells']
    solute_type = config['solute_type']
    seed_SIP_gen = config['seed_SIP_gen']
    seed_sim = config['seed_sim']

    grid_folder_init =\
        f'{solute_type}'\
        + f'/grid_{no_cells[0]}_{no_cells[1]}_'\
        + f'spcm_{no_spcm[0]}_{no_spcm[1]}/'\
        + f'{seed_SIP_gen}/' 
    grid_path_init = config['paths']['simdata'] + grid_folder_init
        
    ### grid and particle generation    
    if config_mode == 'generation':
        grid_path = grid_path_init
        config['paths']['grid'] = grid_path
        if not os.path.exists(grid_path):
            os.makedirs(grid_path)            
            
        if config['eta_threshold'] == 'weak':
            config['weak_threshold'] = True
        else: config['weak_threshold'] = False
        
        idx_mode_nonzero = np.nonzero(no_spcm)[0]
        config['idx_mode_nonzero'] = idx_mode_nonzero
        
        no_modes = len(idx_mode_nonzero)
    
        config['no_modes'] = no_modes
        
        if no_modes == 1:
            no_spcm = no_spcm[idx_mode_nonzero][0]
        else:    
            no_spcm = no_spcm[idx_mode_nonzero]
        
        if config['dist'] == 'lognormal':
            sigma_R = config['sigma_R']
            mu_R = config['mu_R']
            r_critmin = config['r_critmin']
            DNC0 = config['DNC0']
            if no_modes == 1:
                sigma_R = sigma_R[idx_mode_nonzero][0]
                mu_R = mu_R[idx_mode_nonzero][0]
                
                config['r_critmin'] = r_critmin[idx_mode_nonzero][0] # mu
                config['DNC0'] = DNC0[idx_mode_nonzero][0] # mu
                
            else:
                sigma_R = sigma_R[idx_mode_nonzero]
                mu_R = mu_R[idx_mode_nonzero]
                r_critmin = r_critmin[idx_mode_nonzero] # mu
                DNC0 = DNC0[idx_mode_nonzero] # mu
        
            sigma_R_log = np.log( sigma_R )    
            # derive parameters of lognormal distribution of mass f_m(m)
            # assuming mu_R in mu and density in kg/m^3
            # mu_m in 1E-18 kg
            mu_m_log = np.log(mp.compute_mass_from_radius_vec(
                                  mu_R, c.mass_density_NaCl_dry))
            sigma_m_log = 3.0 * sigma_R_log
            dist_par = (mu_m_log, sigma_m_log)    
        
        config['dist_par'] = dist_par

    elif config_mode == 'spin_up':
        config['simulation_mode'] = 'spin_up'
        grid_path = grid_path_init
        config['paths']['grid'] = grid_path
        config['paths']['output'] = grid_path + 'spin_up_wo_col_wo_grav/'
        
        config['t_start'] = config['t_start_spin_up']
        config['t_end'] = config['t_end_spin_up']
        config['spin_up_complete'] = False
        config['g_set'] = 0.0
        config['act_collisions'] = False
        config['act_relaxation'] = False
        
    elif config_mode == 'simulation':
        config['simulation_mode'] = 'simulation'
        if config['act_collisions']:
            if config['spin_up_complete']:
                output_folder = 'w_spin_up_w_col/' + f'{seed_sim}/'
            else:
                output_folder = 'wo_spin_up_w_col/' + f'{seed_sim}/'
        else:
            if config['spin_up_complete']:
                output_folder = 'w_spin_up_wo_col/'
            else:
                output_folder = 'wo_spin_up_wo_col/'
        
        if config['continued_simulation']:
            grid_path = grid_path_init + output_folder
        else:    
            grid_path = grid_path_init + 'spin_up_wo_col_wo_grav/'
        
        config['paths']['grid'] = grid_path
        config['paths']['output'] = grid_path_init + output_folder

        config['t_start'] = config['t_start_sim']
        config['t_end'] = config['t_end_sim']
        config['g_set'] = c.earth_gravity
    
    if config_mode in ['spin_up', 'simulation']:
        config['paths']['init'] = grid_path_init
    
        if not os.path.exists(config['paths']['output']):
                os.makedirs(config['paths']['output'])    
        
        t_start = config['t_start']
        if t_start > 0.:
            water_removed = np.load(grid_path
                                    + f'water_removed_{int(t_start)}.npy')
        else:        
            water_removed = np.array([0.0])
        
        config['scale_dt_cond'] = config['no_cond_per_adv'] // 2
        config['dt_col'] = config['dt_adv'] / config['no_col_per_adv']
        
        ### load collection kernel data
        E_col_grid, radius_grid, \
        R_kernel_low, bin_factor_R, \
        R_kernel_low_log, bin_factor_R_log, \
        no_kernel_bins =\
            load_kernel_data_Ecol(config['kernel_method'],
                                  config['save_folder_Ecol_grid'] + '/' 
                                  + config['kernel_type'] + '/',
                                  config['E_col_const'])
        
        config['no_kernel_bins'] = no_kernel_bins
        config['R_kernel_low_log'] = R_kernel_low_log
        config['bin_factor_R_log'] = bin_factor_R_log    
        no_cols = np.array((0,0))
    
        return E_col_grid, no_cols, water_removed

#%% KINEMATIC MASS FLUX FIELD

# stream function
pi_inv = 1.0/np.pi
def compute_stream_function(x_, z_, j_max_, x_domain_, z_domain_):
    return -j_max_ * x_domain_ * pi_inv * np.sin(np.pi * z_ / z_domain_)\

# dry mass flux  j_d = rho_d * vel
# Z = domain height z
# X = domain width x
# X_over_Z = X/Z
# k_z = pi/Z
# k_x = 2 * pi / X
# j_max = Amplitude
def compute_mass_flux_air_dry( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
    j_x = j_max_ * X_over_Z_ * np.cos(k_x_ * x_) * np.cos(k_z_ * z_ )
    j_z = 2 * j_max_ * np.sin(k_x_ * x_) * np.sin(k_z_ * z_)
    return j_x, j_z

def compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1(grid_,
                                                                   j_max_):
    X = grid_.sizes[0]
    Z = grid_.sizes[1]
    k_x = 2.0 * np.pi / X
    k_z = np.pi / Z
    X_over_Z = X / Z
    vel_pos_u = [grid_.corners[0], grid_.corners[1] + 0.5 * grid_.steps[1]]
    vel_pos_w = [grid_.corners[0] + 0.5 * grid_.steps[0], grid_.corners[1]]
    j_x = compute_mass_flux_air_dry( *vel_pos_u, j_max_,
                                            k_x, k_z, X_over_Z )[0]
    j_z = compute_mass_flux_air_dry( *vel_pos_w, j_max_,
                                            k_x, k_z, X_over_Z )[1]
    return j_x, j_z

# j_max_base = the appropriate value for a grid of 1500 x 1500 m^2
def compute_j_max(j_max_base, grid):
    return np.sqrt( grid.sizes[0] * grid.sizes[0]
                  + grid.sizes[1] * grid.sizes[1] ) \
           / np.sqrt(2.0 * 1500.0 * 1500.0)

#%% RANDOM PARTICLE POSITIONS
          
# no_spcm = super-part/cell in modes [N_mode1_per_cell, N_mode2_per_cell, ...]
# no_spc = # super-part/cell
# no_spt = # SP total in full domain
# returns positions of particles of shape (2, no_spt)
def generate_random_positions(grid, no_spc, seed, set_seed = False):
    if isinstance(no_spc, (list, tuple, np.ndarray)):
        no_spt = np.sum(no_spc)
    else:
        no_spt = grid.no_cells_tot * no_spc
        no_spc = np.ones(grid.no_cells, dtype = np.int64) * no_spc
    if set_seed:
        np.random.seed(seed)        
    rnd_x = np.random.rand(no_spt)
    rnd_z = np.random.rand(no_spt)
    dx = grid.steps[0]
    dz = grid.steps[1]
    x = []
    z = []
    cells = [[],[]]
    n = 0
    for j in range(grid.no_cells[1]):
        z0 = grid.corners[1][0,j]
        for i in range(grid.no_cells[0]):
            x0 = grid.corners[0][i,j]
            for k in range(no_spc[i,j]):
                x.append(x0 + dx * rnd_x[n])
                z.append(z0 + dz * rnd_z[n])
                cells[0].append(i)
                cells[1].append(j)
                n += 1
    pos = np.array([x, z])
    rel_pos = np.array([rnd_x, rnd_z])
    
    return pos, rel_pos, np.array(cells)

#%% COMPUTE VERTICAL PROFILES WITHOUT LIQUID
def compute_profiles_T_p_rhod_S_without_liquid(
        z, z_0, p_0, p_ref, Theta_l, r_tot, SCALE_FACTOR = 1.0 ):

    p_0_over_p_ref = p_0 / p_ref
    kappa_tot = atm.compute_kappa_air_moist(r_tot) # [-]
    kappa_tot_inv = 1.0 / kappa_tot # [-]
    p_0_over_p_ref_to_kappa_tot = p_0_over_p_ref**(kappa_tot) # [-]
    beta_tot = atm.compute_beta_without_liquid(r_tot, Theta_l) # 1/m

    # analytically integrated profiles
    # for hydrostatic system with constant water vapor mixing ratio
    # r_v = r_tot and r_l = 0
    T_over_Theta_l = p_0_over_p_ref_to_kappa_tot - beta_tot * (z - z_0)
    T = T_over_Theta_l * Theta_l
    p_over_p_ref = T_over_Theta_l**(kappa_tot_inv)
    p = p_over_p_ref * p_ref
    rho_dry = p * beta_tot \
            / (T_over_Theta_l * c.earth_gravity * kappa_tot )
    S = atm.compute_pressure_vapor( rho_dry * r_tot, T )\
        / mat.compute_saturation_pressure_vapor_liquid(T)
            
    return T, p, rho_dry, S

#%% INITIALIZE: GENERATE INIT GRID AND SUPER-PARTICLES
# p_0 = 101500 # surface pressure in Pa
# p_ref = 1.0E5 # ref pressure for potential temperature in Pa
# r_tot_0 = 7.5E-3 # kg water / kg dry air, r_tot = r_v + r_l (spatially const)
# Theta_l = 289.0 # K, Liquid potential temperature (spatially constant)
# n_p: list of number density of particles in the modes [n1, n2, ..]
# e.g. n_p = np.array([60.0E6, 40.0E6]) # m^3
# no_spcm = list of number of super particles per cell [no_spc_1, no_spc_2, ..]
# where no_spc_k is the number of SP per cell in mode k
# dst: function or list of functions of the distribution to use,
# e.g. [dst1, dst2, ..]
# dst_par: parameters of the distributions, dst_par = [par1, par2, ...], 
# where par1 = [par11, par12, ...] are pars of dst1
# P_min = cutoff probability for the generation of random radii
# P_max = cutoff probability for the generation of random radii
# r0, r1, dr: parameters for the integration to set the quantiles during
# generation of random radii
# reseed: renew the random seed after the generation of the first mode
## for initialization phase
# S_init_max = 1.05
# dt_init = 0.1 # s
# Newton iterations for the implicit mass growth during initialization phase
# maximal allowed iter counts in initial particle water take up to equilibrium
# iter_cnt_limit = 500

# grid_file_list = ['grid_basics.txt', 'arr_file1.npy', 'arr_file2.npy']
# grid_file_list = [path + s for s in grid_file_list]

# particle_file = 'stored_particles.txt'
# particle_file = path + particle_file

def initialize_grid_and_particles_SinSIP(config):
    ##########################################################################
    ### 1. set base grid
    ##########################################################################
    
    # grid dimensions ('ranges')
    # the step size of all cells is the same
    # the grid dimensions will be adjusted such that they are
    # AT LEAST x_max - x_min, etc... but may also be larger,
    # if the sizes are no integer multiples of the step sizes
    grid_ranges = config['grid_ranges']
    x_min = grid_ranges[0][0]
    x_max = grid_ranges[0][1]
    z_min = grid_ranges[1][0]
    z_max = grid_ranges[1][1]
    
    grid_steps = config['grid_steps']
    dx = grid_steps[0]
    dz = grid_steps[1]
    dy = config['dy']
    
    p_0 = config['p_0']
    p_ref = config['p_ref']
    r_tot_0 = config['r_tot_0']
    Theta_l = config['Theta_l']
    solute_type = config['solute_type']
    DNC0 = config['DNC0']
    no_spcm = config['no_spcm']
    
    no_modes = config['no_modes']
    idx_mode_nonzero = config['idx_mode_nonzero']
    dist_name = config['dist']
    dist_par = config['dist_par']
    eta = config['eta']
    eta_threshold = config['eta_threshold']
    r_critmin = config['r_critmin']
    
    m_high_over_m_low = config['m_high_over_m_low']
    rnd_seed = config['seed_SIP_gen']
    reseed = config['reseed']
    
    S_init_max = config['S_init_max']
    dt_init = config['dt_init']
    Newton_iterations = config['Newton_iterations']
    iter_cnt_limit = config['iter_cnt_limit']
    save_path = config['paths']['grid']
        
    if not os.path.exists(save_path):
        os.makedirs(save_path)
 
    log_file = save_path + f'log_grid.txt'

    grid = Grid( grid_ranges, grid_steps, dy )
    grid.print_info()
    
    with open(log_file, 'w+') as f:
        f.write('grid basic parameters:\n')
        f.write(f'grid ranges [x_min, x_max] [z_min, z_max]:\n')
        for gr_ in grid_ranges:
            for gr__ in gr_:
                f.write(f'{gr__} ')
        f.write('\n')
        f.write('number of cells: ')
        f.write(f'{grid.no_cells[0]}, {grid.no_cells[1]} \n')
        f.write('grid steps: ')
        f.write(f'{grid_steps[0]}, {dy}, {grid_steps[1]}\n\n')
    
    ##########################################################################
    ### 2. Set initial profiles without liquid water
    ##########################################################################
    
    # INITIAL PROFILES
    
    levels_z = grid.corners[1][0]
    centers_z = grid.centers[1][0]
    
    # for testing on a smaller grid: r_tot as array in centers
    r_tot_centers = np.ones_like( centers_z ) * r_tot_0
    
    p_env_init_bottom = np.zeros_like(levels_z)
    p_env_init_bottom[0] = p_0
    
    p_env_init_center = np.zeros_like(centers_z)
    T_env_init_center = np.zeros_like(centers_z)
    rho_dry_env_init_center = np.zeros_like(centers_z)
    r_v_env_init_center = np.ones_like(centers_z) * r_tot_centers
    r_l_env_init_center = np.zeros_like(centers_z)
    S_env_init_center = np.zeros_like(centers_z)
    
    ##########################################################################
    ### 3. Go through levels from the ground and place particles 
    ##########################################################################
    
    print(
    '\n### particle placement and saturation adjustment for each z-level ###')
    print('timestep for sat. adj.: dt_init = ', dt_init)
    with open(log_file, 'a') as f:
        f.write(
    '### particle placement and saturation adjustment for each z-level ###\n')
        f.write(f'solute material = {solute_type}\n')
        f.write(f'nr of modes in dry distribution = {no_modes}\n')
    # derived parameters sip init
    if dist_name == 'lognormal':
        mu_m_log = dist_par[0]
        sigma_m_log = dist_par[1]
        
        # derive scaling parameter kappa from no_spcm
        if no_modes == 1:
            kappa_dst = np.ceil( no_spcm[idx_mode_nonzero][0] / 20 * 28) * 0.1
            kappa_dst = np.maximum(kappa_dst, 0.1)
        elif no_modes == 2:        
            kappa_dst = np.ceil( no_spcm / 20 * np.array([33,25])) * 0.1
            kappa_dst = np.maximum(kappa_dst, 0.1)
        else:        
            kappa_dst = np.ceil( no_spcm / 20 * 28) * 0.1
            kappa_dst = np.maximum(kappa_dst, 0.1)
        print('kappa =', kappa_dst)
        with open(log_file, 'a') as f:
            f.write('intended SIPs per mode and cell = ')
            for k_ in no_spcm:
                f.write(f'{k_:.1f} ')
            f.write('\n')
        with open(log_file, 'a') as f:
            f.write('kappa = ')
            if no_modes == 1:
                f.write(f'{kappa_dst:.1f} ')
            else:
                for k_ in kappa_dst:
                    f.write(f'{k_:.1f} ')
            f.write('\n')

    with open(log_file, 'a') as f:
        f.write(f'timestep for sat. adj.: dt_init = {dt_init}\n')
    
    if eta_threshold == 'weak':
        weak_threshold = True
    else: weak_threshold = False
    
    # start at level 0 (from surface!)
    
    # produce random numbers for relative location:
    np.random.seed( rnd_seed )
    
    V0 = grid.volume_cell
    
    # total number of real particles per mode in one grid cell:
    # total number of real particles in mode 'k' (= 1,2) in cell [i,j]
    # reference value at p_ref (marked by 0)
    no_rpcm_0 =  (np.ceil( V0*DNC0 )).astype(int)
    
    print('no_rpcm_0 = ', no_rpcm_0)
    with open(log_file, 'a') as f:
        f.write('no_rpcm_0 = ')
        if no_modes == 1:
            f.write(f'{no_rpcm_0:.2e} ')
        else:
            for no_rpcm_ in no_rpcm_0:
                f.write(f'{no_rpcm_:.2e} ')
        f.write('\n')
        f.write('\n')
    no_rpct_0 = np.sum(no_rpcm_0)
    
    # empty cell list
    ids_in_cell = []
    for i in range(grid.no_cells[0]):
        row_list = []
        for j in range(grid.no_cells[1]):
            row_list.append( [] )
        ids_in_cell.append(row_list)
    
    ids_in_level = []    
    for j in range(grid.no_cells[1]):
        ids_in_level.append( [] )
    
    mass_water_liquid_levels = []
    mass_water_vapor_levels = []
    
    # start with a cell [i,j]
    # we start with fixed 'z' value to assign levels as well
    iter_cnt_max = 0
    iter_cnt_max_level = 0
    rho_dry_0 = p_0 / (c.specific_gas_constant_air_dry * 293.0)
    no_rpt_should = np.zeros_like(no_rpcm_0)
    
    np.random.seed(rnd_seed)
    
    m_w = []
    m_s = []
    xi = []
    cells_x = []
    cells_z = []
    modes = []
    
    no_spc = np.zeros(grid.no_cells, dtype = np.int64)
    
    no_rpcm_scale_factors_lvl_wise = np.zeros(grid.no_cells[1])
    
    ### go through z-levels from the ground, j is the cell index resp. z
    for j in range(grid.no_cells[1]):

        n_top = j + 1
        n_bot = j
        
        z_bot = levels_z[n_bot]
        z_top = levels_z[n_top]
        
        r_tot = r_tot_centers[j]
        
        kappa_tot = atm.compute_kappa_air_moist(r_tot) # [-]
        kappa_tot_inv = 1.0 / kappa_tot # [-]
        
        p_bot = p_env_init_bottom[n_bot]

        # initial guess at borders of new level from analytical
        # integration without liquid: r_v = r_tot
        # with boundary condition P(z_bot) = p_bot is set
        # calc values for bottom of level
        T_bot, p_bot, rho_dry_bot, S_bot = \
            compute_profiles_T_p_rhod_S_without_liquid(
                z_bot, z_bot, p_bot, p_ref, Theta_l, r_tot)
        
        # calc values for top of level
        T_top, p_top, rho_dry_top, S_top = \
            compute_profiles_T_p_rhod_S_without_liquid(
                z_top, z_bot, p_bot, p_ref, Theta_l, r_tot)
        
        p_avg = (p_bot + p_top) * 0.5
        T_avg = (T_bot + T_top) * 0.5
        rho_dry_avg = (rho_dry_bot + rho_dry_top  ) * 0.5
        S_avg = (S_bot + S_top) * 0.5
        r_v_avg = r_tot
        
        # calc mass dry from 
        # int dV (rho_dry) = dx * dy * int dz rho_dry
        # rho_dry = dp/dz / [g*( 1+r_tot )]
        # => m_s = dx dy / (g*(1+r_tot)) * (p(z_1)-p(z_2))
        mass_air_dry_level = grid.sizes[0] * dy * (p_bot - p_top )\
                             / ( c.earth_gravity * (1 + r_tot) )
        mass_water_vapor_level = r_v_avg * mass_air_dry_level # in kg
        mass_water_liquid_level = 0.0
        mass_particles_level = 0.0
        dm_l_level = 0.0
        dm_p_level = 0.0
        
        print('\n### level', j, '###')
        print('S_env_init0 = ', S_avg)
        with open(log_file, 'a') as f:
            f.write(f'### level {j} ###\n')
            f.write(f'S_env_init0 = {S_avg:.5f}\n')
        ##################################################################
        ### 3a. (first initialization setting S_eq = S_amb if possible)
        ##################################################################
        
        # nr of real particle per cell and mode is now given for rho_dry_avg
        no_rpcm_scale_factor = rho_dry_avg / rho_dry_0
        no_rpcm = np.rint( no_rpcm_0 * no_rpcm_scale_factor ).astype(int)
        print('no_rpcm = ', no_rpcm)
        with open(log_file, 'a') as f:
            f.write('no_rpcm = ')
            if no_modes == 1:
                f.write(f'{no_rpcm:.2e} ')
            else:
                for no_rpcm_ in no_rpcm:
                    f.write(f'{no_rpcm_:.2e} ')
            f.write('\n')
        
        no_rpcm_scale_factors_lvl_wise[j] = no_rpcm_scale_factor
        no_rpt_should += no_rpcm * grid.no_cells[0]
        
        ### create SIP ensemble for this level
        if solute_type == 'NaCl':
            mass_density_dry = c.mass_density_NaCl_dry
        elif solute_type == 'AS':
            mass_density_dry = c.mass_density_AS_dry
        
        print('kappa_dst')
        print(kappa_dst)

        # m_s_lvl = [ m_s[0,j], m_s[1,j], ... ]
        # xi_lvl = ...
        if dist_name == 'lognormal':
            m_s_lvl, xi_lvl, cells_x_lvl, modes_lvl, no_spc_lvl = \
                gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl(
                        no_modes,
                        mu_m_log, sigma_m_log, mass_density_dry,
                        grid.volume_cell, kappa_dst, eta,
                        weak_threshold, r_critmin,
                        m_high_over_m_low, rnd_seed,
                        grid.no_cells[0], no_rpcm,
                        setseed=reseed)
        # expo dist. not implemented
        # elif dist_name == 'expo':
            
        no_spc[:,j] = no_spc_lvl
        if solute_type == 'NaCl':
            w_s_lvl = mp.compute_initial_mass_fraction_solute_m_s_NaCl(
                              m_s_lvl, S_avg, T_avg)
        elif solute_type == 'AS':            
            w_s_lvl = mp.compute_initial_mass_fraction_solute_m_s_AS(
                              m_s_lvl, S_avg, T_avg)
        m_p_lvl = m_s_lvl / w_s_lvl
        m_w_lvl = m_p_lvl - m_s_lvl
        dm_l_level += np.sum(m_w_lvl * xi_lvl)
        dm_p_level += np.sum(m_p_lvl * xi_lvl)
        for mode_n in range(no_modes):
            no_pt_mode_ = len(xi_lvl[modes_lvl==mode_n])
            print('placed', no_pt_mode_,
                  f'SIPs in mode {mode_n}')
            with open(log_file, 'a') as f:
                f.write(f'placed {no_pt_mode_} ')
                f.write(f'SIPs in mode {mode_n}, ')
                f.write(f'-> {no_pt_mode_/grid.no_cells[0]:.2f} ')
                f.write('SIPs per cell\n')
                
        # convert from 10^-18 kg to kg
        dm_l_level *= 1.0E-18 
        dm_p_level *= 1.0E-18 
        mass_water_liquid_level += dm_l_level
        mass_particles_level += dm_p_level
        
        # now we distributed the particles between levels j and j+1
        # thereby sampling liquid water,
        # i.e. the mass of water vapor has dropped
        mass_water_vapor_level -= dm_l_level
        
        # and the ambient fluid is heated by Q = m_l * L_v = C_p dT
        # C_p = m_dry_end*c_p_dry + m_v*c_p_v + m_p_end*c_p_p
        heat_capacity_level =\
            mass_air_dry_level * c.specific_heat_capacity_air_dry_NTP\
            + mass_water_vapor_level *c.specific_heat_capacity_water_vapor_20C\
            + mass_particles_level * c.specific_heat_capacity_water_NTP
        heat_of_vaporization = mat.compute_heat_of_vaporization(T_avg)
        
        dT_avg = dm_l_level * heat_of_vaporization / heat_capacity_level
        
        # assume homogeneous heating: bottom, mid and top are heated equally
        # i.e. the slope (lapse rate) of the temperature in the level
        # remains constant,
        # but the whole (linear) T-curve is shifted by dT_avg
        T_avg += dT_avg
        T_bot += dT_avg
        T_top += dT_avg
        
        r_v_avg = mass_water_vapor_level / mass_air_dry_level
        
        # assume linear T-profile in cells
        p_top = p_bot * (T_top/T_bot)**(((1 + r_tot) * c.earth_gravity * dz)\
                / ( (1 + r_v_avg / atm.epsilon_gc)
                    * c.specific_gas_constant_air_dry * (T_bot - T_top)) )

        p_avg = 0.5 * (p_bot + p_top)
        
        # from integration of dp/dz = - (1 + r_tot) g rho_dry
        rho_dry_avg = (p_bot - p_top) / ( dz * (1 + r_tot) * c.earth_gravity )
        mass_air_dry_level = grid.sizes[0] * dy * dz * rho_dry_avg
        
        r_l_avg = mass_water_liquid_level / mass_air_dry_level
        r_v_avg = r_tot - r_l_avg
        mass_water_vapor_level = r_v_avg * mass_air_dry_level
        
        e_s_avg = mat.compute_saturation_pressure_vapor_liquid(T_avg)
        
        S_avg = atm.compute_pressure_vapor( rho_dry_avg * r_v_avg, T_avg ) \
                / e_s_avg
        
        #####################################################################
        ### 3b. saturation adjustment in level, CELL WISE
        # this was the initial placement of particles
        # now comes the saturation adjustment incl. condensation/vaporization
        # due to supersaturation/subsaturation
        # note that there will also be subsaturation if S < S_act,
        # because water vapor was taken from the atm. in the intitial
        # particle placement step
        #####################################################################
    
        # initialize for saturation adjustment loop
        # loop until the change in dm_l_level is sufficiently small:
        grid.mixing_ratio_water_vapor[:,j] = r_v_avg
        grid.mixing_ratio_water_liquid[:,j] = r_l_avg
        grid.saturation_pressure[:,j] = e_s_avg
        grid.saturation[:,j] = S_avg
    
        dm_l_level = mass_water_liquid_level
        iter_cnt = 0
        
        print('S_env_init after placement =', S_avg)
        print('sat. adj. start')
        with open(log_file, 'a') as f:
            f.write(f'S_env_init after placement = {S_avg:.5f}\n')
            f.write('sat. adj. start\n')
        # need to define criterium -> use relative change in liquid water
        while ( np.abs(dm_l_level/mass_water_liquid_level) > 1e-5
                and iter_cnt < iter_cnt_limit ):
            ## loop over particles in level:
            dm_l_level = 0.0
            dm_p_level = 0.0
            
            D_v = mat.compute_diffusion_constant( T_avg, p_avg )
            K = mat.compute_thermal_conductivity_air(T_avg)
            # c_p = compute_specific_heat_capacity_air_moist(r_v_avg)
            L_v = mat.compute_heat_of_vaporization(T_avg)
    
            for i in range(grid.no_cells[0]):
                cell = (i,j)
                e_s_avg = grid.saturation_pressure[cell]
                # set arbitrary maximum saturation during spin up to
                # avoid overshoot
                # at high saturations, since S > 1.05 can happen initially
                S_avg = grid.saturation[cell]
                S_avg2 = np.min([S_avg, S_init_max ])
                
                ind_x = cells_x_lvl == i
                
                m_w_cell = m_w_lvl[ind_x]
                m_s_cell = m_s_lvl[ind_x]
                
                R_p_cell, w_s_cell, rho_p_cell =\
                    mp.compute_R_p_w_s_rho_p(m_w_cell, m_s_cell, T_avg,
                                          solute_type)
                sigma_p_cell = mat.compute_surface_tension_water(T_avg)
                dm_l, gamma_ =\
                compute_dml_and_gamma_impl_Newton_full(
                    dt_init, Newton_iterations, m_w_cell,
                    m_s_cell, w_s_cell, R_p_cell, T_avg,
                    rho_p_cell,
                    T_avg, p_avg, S_avg2, e_s_avg,
                    L_v, K, D_v, sigma_p_cell, solute_type)
                
                m_w_lvl[ind_x] += dm_l
                
                dm_l_level += np.sum(dm_l * xi_lvl[ind_x])
                dm_p_level += np.sum(dm_l * xi_lvl[ind_x])
                
            # convert from 10^-18 kg to kg
            dm_l_level *= 1.0E-18 
            dm_p_level *= 1.0E-18 
            mass_water_liquid_level += dm_l_level
            mass_particles_level += dm_p_level
    
            # now we distributed the particles between levels j and j+1
            # thereby sampling liquid water,
            # i.e. the mass of water vapor has dropped
            mass_water_vapor_level -= dm_l_level
    
            # and the ambient fluid is heated by Q = m_l * L_v = C_p dT
            # C_p = m_dry_end*c_p_dry + m_v*c_p_v + m_p_end*c_p_p
            heat_capacity_level =\
                mass_air_dry_level * c.specific_heat_capacity_air_dry_NTP \
                + mass_water_vapor_level\
                  * c.specific_heat_capacity_water_vapor_20C \
                + mass_particles_level * c.specific_heat_capacity_water_NTP
            heat_of_vaporization = mat.compute_heat_of_vaporization(T_avg)
    
            dT_avg = dm_l_level * heat_of_vaporization / heat_capacity_level
    
            # assume homogeneous heating:
            # bottom, mid and top are heated equally,
            # i.e. the slope (lapse rate) of the temperature in the level
            # remains constant,
            # but the whole (linear) T-curve is shifted by dT_avg
            T_avg += dT_avg
            T_bot += dT_avg
            T_top += dT_avg
    
            r_v_avg = mass_water_vapor_level / mass_air_dry_level
    
            # assume linear T-profile
            # the T-curve was shifted upwards in total
            p_top = p_bot * (T_top/T_bot)**( kappa_tot_inv
                                             * ( atm.epsilon_gc + r_tot )
                                             / ( atm.epsilon_gc + r_v_avg ) )
            p_avg = 0.5 * (p_bot + p_top)
    
            # from integration of dp/dz = - (1 + r_tot) * g * rho_dry
            rho_dry_avg = (p_bot - p_top)\
                          / ( dz * (1 + r_tot) * c.earth_gravity )
            mass_air_dry_level = grid.sizes[0] * dy * dz * rho_dry_avg
            mass_air_dry_cell = dx * dy * dz * rho_dry_avg
            
            r_l_avg = mass_water_liquid_level / mass_air_dry_level
            r_v_avg = r_tot - r_l_avg
            
            grid.mixing_ratio_water_liquid[:,j].fill(0.0)
            for i in range(grid.no_cells[0]):
                cell = (i,j)
                ind_x = cells_x_lvl == i
                grid.mixing_ratio_water_liquid[cell] +=\
                    np.sum(m_w_lvl[ind_x] * xi_lvl[ind_x])
    
            grid.mixing_ratio_water_liquid[:,j] *= 1.0E-18 / mass_air_dry_cell
            grid.mixing_ratio_water_vapor[:,j] =\
                r_tot - grid.mixing_ratio_water_liquid[:,j]
            
            grid.saturation_pressure[:,j] =\
                mat.compute_saturation_pressure_vapor_liquid(T_avg)
            grid.saturation[:,j] =\
                atm.compute_pressure_vapor(
                    rho_dry_avg * grid.mixing_ratio_water_vapor[:,j], T_avg )\
                / grid.saturation_pressure[:,j]
            
            mass_water_vapor_level = r_v_avg * mass_air_dry_level
            
            e_s_avg = mat.compute_saturation_pressure_vapor_liquid(T_avg)
    
            S_avg = atm.compute_pressure_vapor( rho_dry_avg * r_v_avg, T_avg ) \
                    / e_s_avg
            
            iter_cnt += 1
        
        if iter_cnt_max < iter_cnt: 
            iter_cnt_max = iter_cnt
            iter_cnt_max_level = j
        print('sat. adj. end: iter_cnt = ', iter_cnt,
              ', S_avg_end = ', S_avg,
              '\ndm_l_level = ', dm_l_level,
              ', m_l_level = ', mass_water_liquid_level,
              '\ndm_l_level/mass_water_liquid_level = ',
              dm_l_level/mass_water_liquid_level)
        with open(log_file, 'a') as f:
            f.write(f'sat. adj. end: iter_cnt = {iter_cnt}, ')
            f.write(f'S_avg_end = {S_avg:.5f}\n')
            f.write(f'dm_l_level = {dm_l_level}\n')
            f.write(f'm_l_level = {mass_water_liquid_level}\n')
            f.write(f'dm_l_level/mass_water_liquid_level = ')
            f.write(f'{dm_l_level/mass_water_liquid_level}\n\n')
        mass_water_liquid_levels.append(mass_water_liquid_level)
        mass_water_vapor_levels.append( mass_water_vapor_level )
    
        p_env_init_bottom[n_top] = p_top
        p_env_init_center[j] = p_avg
        T_env_init_center[j] = T_avg
        r_v_env_init_center[j] = r_v_avg
        r_l_env_init_center[j] = r_l_avg
        rho_dry_env_init_center[j] = rho_dry_avg
        S_env_init_center[j] = S_avg
        
        m_w.append(m_w_lvl)
        m_s.append(m_s_lvl)
        xi.append(xi_lvl)
        
        cells_x.append(cells_x_lvl)
        cells_z.append(np.ones_like(cells_x_lvl) * j)        
        
        modes.append(modes_lvl)        
        
    m_w = np.concatenate(m_w)        
    m_s = np.concatenate(m_s)        
    xi = np.concatenate(xi)        
    cells_x = np.concatenate(cells_x)        
    cells_z = np.concatenate(cells_z)   
    cells_comb = np.array( (cells_x, cells_z) )
    modes = np.concatenate(modes)   
    
    print('')
    print('### Saturation adjustment ended for all lvls ###')
    print('iter count max = ', iter_cnt_max, ' level = ', iter_cnt_max_level)
    # total number of particles in grid
    no_particles_tot = np.size(m_w)
    print('last particle ID = ', len(m_w) - 1 )
    print ('no_super_particles_tot placed = ', no_particles_tot)
    print('no_cells x N_p_cell_tot =', np.sum(no_rpct_0)*grid.no_cells[0]\
                                                        *grid.no_cells[1] )
    
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write('### Saturation adjustment ended for all lvls ###\n')
        f.write(f'iter cnt max = {iter_cnt_max}, at lvl {iter_cnt_max_level}')
        f.write('\n')
        f.write(f'last particle ID = {len(m_w) - 1}\n' )
        f.write(f'no_super_particles_tot placed = {no_particles_tot}\n')
        f.write('no_cells x N_p_cell_tot = ')
        f.write(
        f'{np.sum(no_rpct_0) * grid.no_cells[0] * grid.no_cells[1]:.3e}\n')
    
    for i in range(grid.no_cells[0]):
        grid.pressure[i] = p_env_init_center
        grid.temperature[i] = T_env_init_center
        grid.mass_density_air_dry[i] = rho_dry_env_init_center
        
    p_dry = grid.mass_density_air_dry * c.specific_gas_constant_air_dry\
            * grid.temperature
    
    grid.potential_temperature = grid.temperature\
                                 * ( 1.0E5 / p_dry )**atm.kappa_air_dry
                                 
    grid.saturation_pressure =\
        mat.compute_saturation_pressure_vapor_liquid(grid.temperature)
       
    rho_dry_env_init_bottom = 0.5 * ( rho_dry_env_init_center[0:-1]
                                      + rho_dry_env_init_center[1:])
    rho_dry_env_init_bottom =\
        np.insert( rho_dry_env_init_bottom, 0, 
                   0.5 * ( 3.0 * rho_dry_env_init_center[0]
                           - rho_dry_env_init_center[1] ))
    rho_dry_env_init_bottom =\
        np.append( rho_dry_env_init_bottom,
                   0.5 * ( 3.0 * rho_dry_env_init_center[-1]
                           - rho_dry_env_init_center[-2] ) )
    
    print()
    print('placed ', len(m_w.flatten()), 'super particles' )
    print('representing ', np.sum(xi.flatten()), 'real particles:' )
    print('mode real_part_placed, real_part_should, ' + 
          'rel_dev:')
    
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(f'placed {len(m_w.flatten())} super particles\n')
        f.write(f'representing {np.sum(xi.flatten()):.3e} real particles:\n')
        f.write('mode real_part_placed, ')
        f.write('real_part_should, rel dev:\n')
    
    if no_modes == 1:
        rel_dev_ = (np.sum(xi) - no_rpt_should)/no_rpt_should
        print(0, f'{np.sum(xi):.3e}',
              f'{no_rpt_should:.3e}',
              f'{rel_dev_:.3e}')
        with open(log_file, 'a') as f:
            f.write(f'0 {np.sum(xi):.3e} {no_rpt_should:.3e} ')
            f.write(f'{rel_dev_:.3e}\n')
    else:
        for mode_n in range(no_modes):
            ind_mode = modes == mode_n
            rel_dev_ = (np.sum(xi[ind_mode]) - no_rpt_should[mode_n])\
                      /no_rpt_should[mode_n]
            print(mode_n, f'{np.sum(xi[ind_mode]):.3e}',
                  f'{no_rpt_should[mode_n]:.3e}',
                  f'{rel_dev_:.3e}')
            with open(log_file, 'a') as f:
                f.write(f'{mode_n} {np.sum(xi[ind_mode]):.3e} ')
                f.write(f'{no_rpt_should[mode_n]:.3e} ')
                f.write(f'{rel_dev_:.3e}\n')
    
    r_tot_err =\
        np.sum(np.abs(grid.mixing_ratio_water_liquid
                      + grid.mixing_ratio_water_vapor
                      - r_tot_0))
    print()
    print(f'accumulated abs. error'
          + '|(r_l + r_v) - r_tot_should| over all cells = '
          + f'{r_tot_err:.4e}')
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(f'accumulated abs. error |(r_l + r_v) - r_tot_should| ')
        f.write(f'over all cells ')
        f.write(f'= {r_tot_err:.4e}\n')
    
    ########################################################
    ### 4. set mass flux and velocity grid
    ######################################################## 
    
    j_max = 0.6 * np.sqrt(grid.sizes[0] * grid.sizes[0] +
                          grid.sizes[1] * grid.sizes[1])\
                / np.sqrt(2.0 * 1500.0*1500.0)
    print()
    print('j_max')
    print(j_max)
    with open(log_file, 'a') as f:
        f.write(f'\nj_max = {j_max}')
    grid.mass_flux_air_dry =\
        compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1( grid,
                                                                        j_max )
    
    # grid.mass_flux_air_dry[1]:
    # j_z - positions taken at the z-bottom positions
    # every element mass_flux_air_dry[1][i] is a z-profile at fix x
    # and gets divided by the dry density profile at the bottoms
    # note that this is checked to provide the right division by numpy:
    # since mass_flux_air_dry[1] is an 2D array of shape (Nx, Nz)
    # and rho_dry_env_init_bottom is a 1D array of shape (Nz,)
    # each element of mass_flux_air_dry[1] (which is itself an array of dim Nz)
    # is divided BY THE FULL ARRAY 'rho_dry_env_init_bottom'
    # grid.mass_flux_air_dry[0]:
    # j_x - positions taken at the z-center positions
    # every element mass_flux_air_dry[0][i] is a z-profile at fix x_i
    # and gets divided by the dry density profile at the centers
    # note that this is checked to provide the right division by numpy:
    # since mass_flux_air_dry[0] is an 2D array of shape (Nx, Nz)
    # and rho_dry_env_init_bottom is a 1D array of shape (Nz,)
    # each element of mass_flux_air_dry[0] (which is itself an array of dim Nz)
    # is divided BY THE FULL ARRAY 'rho_dry_env_init_bottom' BUT
    # we need to add a dummy density for the center in a cell above
    # the highest cell, because velocity has
    # the same dimensions as grid.corners and not grid.centers
    grid.velocity[0] = grid.mass_flux_air_dry[0] / rho_dry_env_init_bottom
    grid.velocity[1] =\
        grid.mass_flux_air_dry[1]\
        / np.append(rho_dry_env_init_center, rho_dry_env_init_center[-1])
    
    grid.update_material_properties()
    V0_inv = 1.0 / grid.volume_cell
    grid.rho_dry_inv = np.ones_like(grid.mass_density_air_dry)\
                       / grid.mass_density_air_dry
    grid.mass_dry_inv = V0_inv * grid.rho_dry_inv
    
    # assign random positions to particles
    
    pos, rel_pos, cells = generate_random_positions(grid, no_spc, rnd_seed,
                                                    set_seed=False)
    # init velocities
    vel = interpolate_velocity_from_cell_bilinear(cells, rel_pos,
                                                      grid.velocity,
                                                      grid.no_cells)
    active_ids = np.full( len(m_s), True )
    
    t = 0
    save_grid_and_particles_full(t, grid, pos, cells, vel,
                                 m_w, m_s, xi,
                                 active_ids, save_path)
    
    np.save(save_path + 'modes_0', modes)
    np.save(save_path + 'no_rpcm_scale_factors_lvl_wise',
            no_rpcm_scale_factors_lvl_wise)
    
    paras = [x_min, x_max, z_min, z_max, dx, dy, dz, p_0, p_ref, r_tot_0,
             Theta_l, DNC0, no_spcm, dist_name,
             dist_par, kappa_dst, eta, eta_threshold,
             r_critmin, m_high_over_m_low,
             eta, rnd_seed, S_init_max, dt_init,
             Newton_iterations, iter_cnt_limit]
    para_names = 'x_min, x_max, z_min, z_max, dx, dy, dz, p_0, p_ref, ' \
    + 'r_tot_0, Theta_l, DNC0, no_super_particles_cell_mode, dist, ' \
    + 'mu_m_log, sigma_m_log, kappa_dst, eta, eta_threshold, ' \
    + 'r_critmin, m_high_over_m_low, rnd_seed, S_init_max, dt_init, ' \
    + 'Newton_iterations_init, iter_cnt_limit'
    
    grid_para_file = save_path + 'grid_paras.txt'
    with open(grid_para_file, 'w') as f:
        f.write( para_names + '\n' )
        for item in paras:
            type_ = type(item)
            if type_ is list or type_ is np.ndarray or type_ is tuple:
                for el in item:
                    tp_e = type(el)
                    if tp_e is list or tp_e is np.ndarray or tp_e is tuple:
                        for el_ in el:
                            f.write( f'{el_} ' )
                    else:
                        f.write( f'{el} ' )
            else: f.write( f'{item} ' )        
    
    return grid, pos, cells, cells_comb, vel, m_w, m_s, xi,\
           active_ids 
