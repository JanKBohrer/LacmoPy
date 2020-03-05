#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

GRID AND PARTICLE INITIALIZATION

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
import os
import numpy as np
import math
from numba import njit

import constants as c
from grid import Grid
from grid import interpolate_velocity_from_cell_bilinear

from integration import \
    compute_dml_and_gamma_impl_Newton_full

import materialproperties as mat
import atmosphere as atm
import microphysics as mp
import distributions as dist
from generate_SIP_ensemble_dst import \
    gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl    

from file_handling import save_grid_and_particles_full
from file_handling import load_kernel_data


#%% FUNCTION DEFS    
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

def compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1( grid_,
                                                                    j_max_ ):
    X = grid_.sizes[0]
    Z = grid_.sizes[1]
    k_x = 2.0 * np.pi / X
    k_z = np.pi / Z
    X_over_Z = X / Z
    # j_max = 0.6 # (m/s)*(m^3/kg)
    # grid only has corners as positions...
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
            load_kernel_data(config['kernel_method'],
                             config['save_folder_Ecol_grid'] + '/' 
                             + config['kernel_type'] + '/',
                             config['E_col_const'])
        
        config['no_kernel_bins'] = no_kernel_bins
        config['R_kernel_low_log'] = R_kernel_low_log
        config['bin_factor_R_log'] = bin_factor_R_log    
        no_cols = np.array((0,0))
    
        return E_col_grid, no_cols, water_removed


#%% STOCHASTICS

# input: prob = [p1, p2, p3, ...] = quantile probab. -> need to set no_q = None
# OR (if prob = None)
# input: no_q = number of quantiles (including prob = 1.0)
# returns N-1 quantiles q with given (prob)
# or equal probability distances (no_q):
# N = 4 -> P(X < q[0]) = 1/4, P(X < q[1]) = 2/4, P(X < q[2]) = 3/4
# this function was checked with the std normal distribution
def compute_quantiles(func, par, x0, x1, dx, prob, no_q=None):
    # probabilities of the quantiles
    if par is not None:
        f = lambda x : func(x, par)
    else: f = func
    
    if prob is None:
        prob = np.linspace(1.0/no_q,1.0,no_q)[:-1]
    print ('quantile probabilities = ', prob)
    
    intl = 0.0
    x = x0
    q = []
    
    cnt = 0
    for i,p in enumerate(prob):
        while (intl < p and p < 1.0 and x <= x1 and cnt < 1E8):
            intl += dx * f(x)
            x += dx
            cnt += 1
        # the quantile value is somewhere between x - dx and x
        # q.append(x)    
        # q.append(x - 0.5 * dx)    
        q.append(max(x - dx,x0))    
    
    print ('quantile values = ', q)
    return q, prob
          
# for a given grid with grid center positions grid_centers[i,j]:
# create random radii rad[i,j] = array of shape (Nx, Ny, no_spc)
# and the corresp. values of the distribution p[i,j]
# input:
# p_min, p_max: quantile probs to cut off e.g. (0.001,0.999)
# no_spc: number of super-droplets per cell (scalar!)
# dst: distribution to use
# par: params of the distribution par -> 'None' possible if dst = dst(x)
# r0, r1, dr: parameters for the numerical integration to find the cutoffs
def generate_random_radii_monomodal(grid, dst, par, no_spc, p_min, p_max,
                                    r0, r1, dr, seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    if par is not None:
        func = lambda x : dst(x, par)
    else: func = dst
    
    qs, Ps = compute_quantiles(func, None, r0, r1, dr, [p_min, p_max], None)
    
    r_min = qs[0]
    r_max = qs[1]
    
    bins = np.linspace(r_min, r_max, no_spc+1)
    
    rnd = []
    for i,b in enumerate(bins[0:-1]):
        # we need 1 radius value for each bin for each cell
        rnd.append( np.random.uniform(b, bins[i+1], grid.no_cells_tot) )
    
    rnd = np.array(rnd)
    
    rnd = np.transpose(rnd)    
    shape = np.hstack( [np.array(np.shape(grid.centers)[1:]), [no_spc]] )
    rnd = np.reshape(rnd, shape)
    
    weights = func(rnd)
    
    for p_row in weights:
        for p_cell in p_row:
            p_cell /= np.sum(p_cell)
    
    return rnd, weights #, bins

# no_spc = super particles per cell
# creates no_spc random radii per cell and the no_spc weights per cell
# where sum(weight_i) = 1.0
# the weights are drawn from a normal distribution with mu = 1.0, sigma = 0.2
# and rejected if weight < 0. The weights are then normalized such that sum = 1
# par = (mu*, ln(sigma*)), where mu* and sigma* are the
# GEOMETRIC expectation value and standard dev. of the lognormal distr. resp.
def generate_random_radii_monomodal_lognorm(grid, par, no_spc,
                                            seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    no_spt = grid.no_cells_tot * no_spc
    
    # draw random numbers from log normal distr. by this procedure
    Rs = np.random.normal(0.0, par[1], no_spt)
    Rs = np.exp(Rs)
    Rs *= par[0]
    # draw random weights
    weights = np.abs(np.random.normal(1.0, 0.2, no_spt))
    
    # bring to shape Rs[i,j,k], where Rs[i,j] = arr[k] k = 0...no_spc-1
    shape = np.hstack( [np.array(np.shape(grid.centers)[1:]), [no_spc]] )
    Rs = np.reshape(Rs, shape)
    weights = np.reshape(weights, shape)
    
    for p_row in weights:
        for p_cell in p_row:
            p_cell /= np.sum(p_cell)
    
    return Rs, weights #, bins

# no_spcm MUST be a list/array with [no_1, no_2, .., no_N], N = number of modes
# no_spcm[k] = 0 is possible for some mode k. Then no particles are generated
# for this mode...
# par MUST be a list with [par1, par2, .., parN]
# where par1 = [p11, p12, ...] , par2 = [...] etc
# everything else CAN be a list or a scalar single float/int
# if given as scalar, all modes will use the same value
# the seed is set at least once
# reseed = True will reset the seed everytime for every new mode with the value
# given in the seed list, it can also be used with seed being a single scalar
# about the seeds:
# using np.random.seed(seed) in a function sets the seed globaly
# after leaving the function, the seed remains the set seed
def generate_random_radii_multimodal(grid, dst, par, no_spcm, p_min, p_max, 
                                     r0, r1, dr, seed, reseed = False):
    no_modes = len(no_spcm)
    if not isinstance(p_min, (list, tuple, np.ndarray)):
        p_min = [p_min] * no_modes
    if not isinstance(p_max, (list, tuple, np.ndarray)):
        p_max = [p_max] * no_modes
    if not isinstance(dst, (list, tuple, np.ndarray)):
        dst = [dst] * no_modes
    if not isinstance(r0, (list, tuple, np.ndarray)):
        r0 = [r0] * no_modes
    if not isinstance(r1, (list, tuple, np.ndarray)):
        r1 = [r1] * no_modes
    if not isinstance(dr, (list, tuple, np.ndarray)):
        dr = [dr] * no_modes
    if not isinstance(seed, (list, tuple, np.ndarray)):
        seed = [seed] * no_modes        

    rad = []
    weights = []
    print(p_min)
    # set seed always once
    setseed = True
    for k in range(no_modes):
        if no_spcm[k] > 0:
            r, w = generate_random_radii_monomodal(
                       grid, dst[k], par[k], no_spcm[k], p_min[k], p_max[k],
                       r0[k], r1[k], dr[k], seed[k], setseed)
            rad.append( r )
            weights.append( w )
            setseed = reseed
    # we need the different modes separately, because they 
    # are weighted with different total concentrations
    # if len(rad)>1:
    #     rad = np.concatenate(rad, axis = 2)
    #     weights = np.concatenate(weights, axis = 2)
    
    return np.array(rad), np.array(weights)

def generate_random_radii_multimodal_lognorm(grid, par, no_spcm,
                                             seed, reseed = False):
    no_modes = len(no_spcm)
    if not isinstance(seed, (list, tuple, np.ndarray)):
        seed = [seed] * no_modes       
        
    rad = []
    weights = []
    # set seed always once
    setseed = True
    for k in range(no_modes):
        if no_spcm[k] > 0:
            r, w = generate_random_radii_monomodal_lognorm(
                       grid, par[k], no_spcm[k], seed[k], setseed)
            rad.append( r )
            weights.append( w )
            setseed = reseed
    # we need the different modes separately, because they 
    # are weighted with different total concentrations
    # if len(rad)>1:
    #     rad = np.concatenate(rad, axis = 2)
    #     weights = np.concatenate(weights, axis = 2)
    
    return np.array(rad), np.array(weights)

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


#%% GENERATE SIP ENSEMBLES

### NEW APPROACH AFTER UNTERSTRASSER CODE 
    
#############################################################################
### OLD APPROACH

# par = 'rate' parameter 'k' of the expo distr: k*exp(-k*m) (in 10^18 kg)
# no_rpc = number of real particles in cell
def generate_SIP_ensemble_expo_SingleSIP_weak_threshold(
        par, no_rpc, r_critmin, m_high_by_m_low, kappa,
        eta, seed, setseed):
    
    if setseed: np.random.seed(seed)
    m_low = mp.compute_mass_from_radius(r_critmin,
                                     c.mass_density_water_liquid_NTP)
    # m_high = num_int_expo_impl_right_border(0.0, p_max, 1.0/par*0E-6, par,
    #                                            cnt_lim=1E8)
    
    m_high = m_low * m_high_by_m_low
    # since we consider only particles with m > m_low, the total number of
    # placed particles and the total placed mass will be underestimated
    # to fix this, we could adjust the PDF
    # by e.g. one of the two possibilities:
    # 1. decrease the total number of particles and thereby the total pt conc.
    # 2. keep the total number of particles, and thus increase the PDF
    # in the interval [m_low, m_high]
    # # For 1.: decrease the total number of particles by multiplication 
    # # with factor num_int_expo_np(m_low, m_high, par, steps=1.0E6)
    # print(no_rpc)
    # no_rpc *= num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    # for 2.:
    # increase the total number of
    # real particle 'no_rpc' by 1.0 / int_(m_low)^(m_high)
    # no_rpc *= 1.0/num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    kappa_inv = 1.0/kappa
    bin_fac = 10**kappa_inv
    
    masses = []
    xis = []
    bins = [m_low]
    no_rp_set = 0
    no_sp_set = 0
    bin_n = 0
    
    m = np.random.uniform(m_low/2, m_low)
    xi = dist.dst_expo(m,par) * m_low * no_rpc
    masses.append(m)
    xis.append(xi)
    no_rp_set += xi
    no_sp_set += 1    
    
    m_left = m_low

    while(m < m_high):
        m_right = m_left * bin_fac
        m = np.random.uniform(m_left, m_right)
        # we do not round to integer here, because of the weak threshold below
        # the rounding is done afterwards
        xi = dist.dst_expo(m,par) * (m_right - m_left) * no_rpc
        masses.append(m)
        xis.append(xi)
        no_rp_set += xi
        no_sp_set += 1
        m_left = m_right
        bins.append(m_right)
        bin_n += 1
    
    xis = np.array(xis)
    masses = np.array(masses)
    
    xi_max = xis.max()
    
    xi_critmin = int(xi_max * eta)
    if xi_critmin < 1: xi_critmin = 1
    
    for bin_n,xi in enumerate(xis):
        if xi < xi_critmin:
            p = xi / xi_critmin
            if np.random.rand() >= p:
                xis[bin_n] = 0
            else: xis[bin_n] = xi_critmin
    
    ind = np.nonzero(xis)
    xis = xis[ind].astype(np.int64)
    masses = masses[ind]
    
    # now, some bins are empty, thus the total number sum(xis) is not right
    # moreover, the total mass sum(masses*xis) is not right ...
    # FIRST: test, how badly the total number and total mass of the cell
    # is violated, if this is bad:
    # -> create a number of particles, such that the total number is right
    # and assign the average mass to it
    # then reweight all of the masses such that the total mass is right.
    
    return masses, xis, m_low, m_high, bins



# par = 'rate' parameter 'k' of the expo distr: k*exp(-k*m) (in 10^18 kg)
# no_rpc = number of real particles in cell
# NON INTERGER WEIGHTS ARE POSSIBLE A SIN UNTERSTRASSER 2017
def generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint(
        par, no_rpc, r_critmin=0.6, m_high_by_m_low=1.0E6, kappa=40,
        eta=1.0E-9, seed=4711, setseed = True):
    
    if setseed: np.random.seed(seed)
    m_low = mp.compute_mass_from_radius(r_critmin,
                                     c.mass_density_water_liquid_NTP)
    # m_high = num_int_expo_impl_right_border(0.0, p_max, 1.0/par*0E-6, par,
    #                                            cnt_lim=1E8)
    
    m_high = m_low * m_high_by_m_low
    # since we consider only particles with m > m_low, the total number of
    # placed particles and the total placed mass will be underestimated
    # to fix this, we could adjust the PDF
    # by e.g. one of the two possibilities:
    # 1. decrease the total number of particles and thereby the total pt conc.
    # 2. keep the total number of particles, and thus increase the PDF
    # in the interval [m_low, m_high]
    # # For 1.: decrease the total number of particles by multiplication 
    # # with factor num_int_expo_np(m_low, m_high, par, steps=1.0E6)
    # print(no_rpc)
    # no_rpc *= num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    # for 2.:
    # increase the total number of
    # real particle 'no_rpc' by 1.0 / int_(m_low)^(m_high)
    # no_rpc *= 1.0/num_int_expo(m_low, m_high, par, steps=1.0E6)
    
    kappa_inv = 1.0/kappa
    bin_fac = 10**kappa_inv
    
    masses = []
    xis = []
    bins = [m_low]
    no_rp_set = 0
    no_sp_set = 0
    bin_n = 0
    
    m = np.random.uniform(m_low/2, m_low)
    xi = dist.dst_expo(m,par) * m_low * no_rpc
    masses.append(m)
    xis.append(xi)
    no_rp_set += xi
    no_sp_set += 1    
    
    m_left = m_low
    # m = 0.0
    
    # print('no_rpc =', no_rpc)
    # while(no_rp_set < no_rpc and m < m_high):
    while(m < m_high):
        # m_right = m_left * 10**kappa_inv
        m_right = m_left * bin_fac
        # print('missing particles =', no_rpc - no_rp_set)
        # print('m_left, m_right, m_high')
        # print(bin_n, m_left, m_right, m_high)
        m = np.random.uniform(m_left, m_right)
        # print('m =', m)
        # we do not round to integer here, because of the weak threshold below
        # the rounding is done afterwards
        xi = dist.dst_expo(m,par) * (m_right - m_left) * no_rpc
        # xi = int((dst_expo(m,par) * (m_right - m_left) * no_rpc))
        # if xi < 1 : xi = 1
        # if no_rp_set + xi > no_rpc:
        #     xi = no_rpc - no_rp_set
            # print('no_rpc reached')
        # print('xi =', xi)
        masses.append(m)
        xis.append(xi)
        no_rp_set += xi
        no_sp_set += 1
        m_left = m_right
        bins.append(m_right)
        bin_n += 1
    
    xis = np.array(xis)
    masses = np.array(masses)
    
    xi_max = xis.max()
    # print('xi_max =', f'{xi_max:.2e}')
    
    # xi_critmin = int(xi_max * eta)
    xi_critmin = xi_max * eta
    # if xi_critmin < 1: xi_critmin = 1
    # print('xi_critmin =', xi_critmin)
    
    valid_ind = []
    
    for bin_n,xi in enumerate(xis):
        if xi < xi_critmin:
            # print('')
            p = xi / xi_critmin
            if np.random.rand() >= p:
                xis[bin_n] = 0
            else:
                xis[bin_n] = xi_critmin
                valid_ind.append(bin_n)
        else: valid_ind.append(bin_n)
    
    # ind = np.nonzero(xis)
    # xis = xis[ind].astype(np.int64)
    valid_ind = np.array(valid_ind)
    xis = xis[valid_ind]
    masses = masses[valid_ind]
    
    # now, some bins are empty, thus the total number sum(xis) is not right
    # moreover, the total mass sum(masses*xis) is not right ...
    # FIRST: test, how badly the total number and total mass of the cell
    # is violated, if this is bad:
    # -> create a number of particles, such that the total number is right
    # and assign the average mass to it
    # then reweight all of the masses such that the total mass is right.
    
    return masses, xis, m_low, m_high, bins

@njit()
def generate_SIP_ensemble_expo_SingleSIP_weak_threshold_nonint2(
        par, no_rpc, r_critmin=0.6, m_high_by_m_low=1.0E6, kappa=40,
        eta=1.0E-9, seed=4711, setseed = True):
    bin_factor = 10**(1.0/kappa)
    m_low = mp.compute_mass_from_radius(r_critmin,
                                        c.mass_density_water_liquid_NTP)
    m_left = m_low
    l_max = int(kappa * np.log10(m_high_by_m_low)) + 1
    rnd = np.random.rand( l_max )
    
    masses = np.zeros(l_max, dtype = np.float64)
    xis = np.zeros(l_max, dtype = np.float64)
    bins = np.zeros(l_max+1, dtype = np.float64)
    bins[0] = m_left
    
    for l in range(l_max):
        m_right = m_left * bin_factor
        bins[l+1] = m_right
        dm = m_right - m_left
        
        m = m_left + dm * rnd[l]
        masses[l] = m
        xis[l] = no_rpc * dm * dist.dst_expo(m, par)
        
        m_left = m_right
    
    xi_max = xis.max()
    xi_critmin = xi_max * eta
    
    switch = np.ones(l_max, dtype=np.int64)
    
    for l in range(l_max):
        if xis[l] < xi_critmin:
            if np.random.rand() < xis[l] / xi_critmin:
                xis[l] = xi_critmin
            else: switch[l] = 0
    
    ind = np.nonzero(switch)[0]
    
    xis = xis[ind]
    masses = masses[ind]
    
    
    return masses, xis, m_low, bins

# no_spc is the intended number of super particles per cell,
# this will right on average, but will vary due to the random assigning 
# process of the xi_i
# m0, m1, dm: parameters for the numerical integration to find the cutoffs
# and for the numerical integration of integrals
# dV = volume size of the cell (in m^3)
# n0: initial particle number distribution (in 1/m^3)
# IN WORK: make the bin size smaller for low masses to get
# finer resolution here...
# -> 'steer against' the expo PDF for low m -> something in between
# we need higher resolution in very small radii (Unterstrasser has
# R_min = 0.8 mu
# at least need to go down to 1 mu (resolution there...)
# eps = parameter for the bin linear spreading:
# bin_size(m = m_high) = eps * bin_size(m = m_low)
# @njit()
def generate_SIP_ensemble_expo_my_xi_rnd(par, no_spc, no_rpc,
                                         total_mass_in_cell,
                                         p_min, p_max, eps,
                                         m0, m1, dm, seed, setseed = True):
    if setseed: np.random.seed(seed)
    
    m_low = 0.0
    m_high = dist.num_int_expo_impl_right_border(m_low,p_max, dm, par, 1.0E8)
    
    # the bin mean size must be adjusted by ~np.log10(eps)
    # to get to a SIP number in the cell of about the intended number
    # if the reweighting is not done, the SIP number will be much larger
    # due to small bins for low masses
    bin_size_mean = (m_high - m_low) / no_spc * np.log10(eps)
    
    a = bin_size_mean * (eps - 1) / (m_high * 0.5 * (eps + 1))
    b = bin_size_mean / (0.5 * (eps + 1))
    
    # the current position of the left bin border
    m_left = m_low
    
    # generate list of masses and multiplicities
    masses = []
    xis = []
    
    # no of particles placed:
    no_pt = 0
    
    # let f(m) be the PDF!! (m can be the mass or the radius, depending,
    # which distr. is given), NOTE that f(m) is not the number concentration..
    
    pt_n = 0
    while no_pt < no_rpc:
        ### i) determine the next xi_mean by
        # integrating xi_mean = no_rpc * int_(m_left)^(m_left+dm) dm f(m)
        # print(pt_n, 'm_left=', m_left)
        # m = m_left
        bin_size = a * m_left + b
        m_right = m_left + bin_size
        
        intl = dist.num_int_expo(m_left, m_right, par, steps=1.0E5)
        
        xi_mean = no_rpc * intl
        if xi_mean > 10:
            xi = np.random.normal(xi_mean, 0.2*xi_mean)
        else:
            xi = np.random.poisson(xi_mean)
        xi = int(math.ceil(xi))
        if xi <= 0: xi = 1
        if no_pt + xi >= no_rpc:
            xi = no_rpc - no_pt
            M_sys = np.sum( np.array(xis)*np.array(masses) )
            M_should = no_rpc * dist.num_int_expo_mean(0.0,m_left,par,1.0E7)
            masses = [m * M_should / M_sys for m in masses]
            # M = p_max * total_mass_in_cell - M
            M_diff = total_mass_in_cell - M_should
            if M_diff <= 0.0:
                M_diff = 1.0/par                
            mu = M_diff/xi
            if mu <= 1.02*masses[pt_n-1]:
                xi_sum = xi + xis[pt_n-1]
                m_sum = xi*mu + xis[pt_n-1] * masses[pt_n-1]
                xis[pt_n-1] = xi_sum
                masses[pt_n-1] = m_sum / xi_sum
                no_pt += xi
            else:                
                masses.append(mu)    
                xis.append(xi)
                no_pt += xi
                pt_n += 1            
        else:            
            ### iii) set the right right bin border
            # by no_rpc * int_(m_left)^m_right dm f(m) = xi
            
            m_right = dist.num_int_expo_impl_right_border(m_left, xi/no_rpc,
                                                          dm, par)
            
            intl = dist.num_int_expo_mean(m_left, m_right, par)
            
            mu = intl * no_rpc / xi
            masses.append(mu)
            xis.append(xi)
            no_pt += xi
            pt_n += 1
            m_left = m_right
        
        if m_left >= m_high and no_pt < no_rpc:

            xi = no_rpc - no_pt
            # last = True
            M_sys = np.sum( np.array(xis)*np.array(masses) )
            M_should = no_rpc * dist.num_int_expo_mean(0.0,m_left,par,1.0E7)
            masses = [m * M_should / M_sys for m in masses]
            # M = p_max * total_mass_in_cell - M
            M_diff = total_mass_in_cell - M_should
            if M_diff <= 0.0:
                M_diff = 1.0/par
            
            # mu = max(p_max*M/xi,m_left)
            mu = M_diff/xi
            if mu <= 1.02*masses[pt_n-1]:
                xi_sum = xi + xis[pt_n-1]
                m_sum = xi*mu + xis[pt_n-1] * masses[pt_n-1]
                xis[pt_n-1] = xi_sum
                masses[pt_n-1] = m_sum / xi_sum
                no_pt += xi
            else:                
                masses.append(mu)    
                xis.append(xi)
                no_pt += xi
                pt_n += 1
    
    return np.array(masses), np.array(xis, dtype=np.int64), m_low, m_high

#%% COMPUTE VERTICAL PROFILES WITHOUT LIQUID
def compute_profiles_T_p_rhod_S_without_liquid(
        z_, z_0_, p_0_, p_ref_, Theta_l_, r_tot_, SCALE_FACTOR_ = 1.0 ):

    z_0 = z_0_

    # constants:
    p_ref = p_ref_ # ref pressure for potential temperature in Pa
    p_0 = p_0_ # surface pressure in Pa
    p_0_over_p_ref = p_0 / p_ref

    r_tot = r_tot_ # kg water / kg dry air
    Theta_l = Theta_l_ # K

    kappa_tot = atm.compute_kappa_air_moist(r_tot) # [-]
    kappa_tot_inv = 1.0 / kappa_tot # [-]
    p_0_over_p_ref_to_kappa_tot = p_0_over_p_ref**(kappa_tot) # [-]
    beta_tot = atm.compute_beta_without_liquid(r_tot, Theta_l) # 1/m

    # analytically integrated profiles
    # for hydrostatic system with constant water vapor mixing ratio
    # r_v = r_tot and r_l = 0
    T_over_Theta_l = p_0_over_p_ref_to_kappa_tot - beta_tot * (z_ - z_0)
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
    dist = config['dist']
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
                f.write(f'{gr__}')
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
    
    ##############################################
    ### 3. Go through levels from the ground and place particles 
    ##############################################
    
    print(
    '\n### particle placement and saturation adjustment for each z-level ###')
    print('timestep for sat. adj.: dt_init = ', dt_init)
    with open(log_file, 'a') as f:
        f.write(
    '### particle placement and saturation adjustment for each z-level ###\n')
        f.write(f'solute material = {solute_type}\n')
        f.write(f'nr of modes in dry distribution = {no_modes}\n')
    # derived parameters sip init
    if dist == 'lognormal':
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
        ###############################################################
        ### 3a. (first initialization setting S_eq = S_amb if possible)
        ###############################################################
        
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
        if dist == 'lognormal':
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
        # elif dist == 'expo':
            
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
             Theta_l, DNC0, no_spcm, dist,
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
