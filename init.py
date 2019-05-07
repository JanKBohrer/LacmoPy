#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 11:31:49 2019

@author: jdesk
"""

import numpy as np
import math
import matplotlib.pyplot as plt

import constants as c
from grid import Grid
from grid import interpolate_velocity_from_cell_bilinear_jit,\
                 interpolate_velocity_from_position_bilinear_jit,\
                 compute_cell_and_relative_position_jit
from microphysics import compute_mass_from_radius,\
                         compute_initial_mass_fraction_solute_NaCl,\
                         compute_radius_from_mass,\
                         compute_density_particle,\
                 compute_delta_water_liquid_and_mass_rate_implicit_Newton_full,\
                         compute_R_p_w_s_rho_p
                         
from atmosphere import compute_kappa_air_moist,\
                       compute_diffusion_constant,\
                       compute_thermal_conductivity_air,\
                       compute_specific_heat_capacity_air_moist,\
                       compute_heat_of_vaporization,\
                       compute_saturation_pressure_vapor_liquid,\
                       compute_pressure_vapor,\
                       compute_pressure_ideal_gas,\
                       epsilon_gc, compute_surface_tension_water,\
                       kappa_air_dry,\
                       compute_beta_without_liquid,\
                       compute_temperature_from_potential_temperature_moist
                       
from file_handling import save_particles_to_files,\
                          save_grid_and_particles_full,\
                          load_grid_and_particles_full
    
# IN WORK: what do you need from grid.py?
# from grid import *
# from physical_relations_and_constants import *

# stream function
pi_inv = 1.0/np.pi
def compute_stream_function_Arabas(x_, z_, j_max_, x_domain_, z_domain_):
    return -j_max_ * x_domain_ * pi_inv * np.sin(np.pi * z_ / z_domain_)\
                                        * np.cos( 2 * np.pi * x_ / x_domain_)

# dry mass flux  j_d = rho_d * vel
# Z = domain height z
# X = domain width x
# X_over_Z = X/Z
# k_z = pi/Z
# k_x = 2 * pi / X
# j_max = Amplitude
def compute_mass_flux_air_dry_Arabas( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
    j_x = j_max_ * X_over_Z_ * np.cos(k_x_ * x_) * np.cos(k_z_ * z_ )
    j_z = 2 * j_max_ * np.sin(k_x_ * x_) * np.sin(k_z_ * z_)
    return j_x, j_z

#def compute_mass_flux_air_dry_x( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
#    return j_max_ * X_over_Z_ * np.cos(k_x_ * x_) * np.cos(k_z_ * z_ )
#def compute_mass_flux_air_dry_z( x_, z_, j_max_, k_x_, k_z_, X_over_Z_ ):
#    return 2 * j_max_ * np.sin(k_x_ * x_) * np.sin(k_z_ * z_)
#def mass_flux_air_dry(x_, z_):
#    return compute_mass_flux_air_dry_Arabas(x_, z_, j_max, k_x, k_z, X_over_Z)
#
#def mass_flux_air_dry_x(x_, z_):
#    return compute_mass_flux_air_dry_x(x_, z_, j_max, k_x, k_z, X_over_Z)
#def mass_flux_air_dry_z(x_, z_):
#    return compute_mass_flux_air_dry_z(x_, z_, j_max, k_x, k_z, X_over_Z)

def compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1( grid_,
                                                                    j_max_ ):
    X = grid_.sizes[0]
    Z = grid_.sizes[1]
    k_x = 2.0 * np.pi / X
    k_z = np.pi / Z
    X_over_Z = X / Z
#    j_max = 0.6 # (m/s)*(m^3/kg)
    # grid only has corners as positions...
    vel_pos_u = [grid_.corners[0], grid_.corners[1] + 0.5 * grid_.steps[1]]
    vel_pos_w = [grid_.corners[0] + 0.5 * grid_.steps[0], grid_.corners[1]]
    j_x = compute_mass_flux_air_dry_Arabas( *vel_pos_u, j_max_,
                                            k_x, k_z, X_over_Z )[0]
    j_z = compute_mass_flux_air_dry_Arabas( *vel_pos_w, j_max_,
                                            k_x, k_z, X_over_Z )[1]
    return j_x, j_z

# j_max_base = the appropriate value for a grid of 1500 x 1500 m^2
def compute_j_max(j_max_base, grid):
    return np.sqrt( grid.sizes[0] * grid.sizes[0]
                  + grid.sizes[1] * grid.sizes[1] ) \
           / np.sqrt(2.0 * 1500.0 * 1500.0)

####
# par[0] = mu
# par[1] = sigma
two_pi_sqrt = math.sqrt(2.0 * math.pi)
def dst_normal(x, par):
    return np.exp( -0.5 * ( ( x - par[0] ) / par[1] )**2 ) \
           / (two_pi_sqrt * par[1])

# par[0] = mu^* = geometric mean of the log-normal dist
# par[1] = ln(sigma^*) = lognat of geometric std dev of log-normal dist
def dst_log_normal(x, par):
    # sig = math.log(par[1])
    f = np.exp( -0.5 * ( np.log( x / par[0] ) / par[1] )**2 ) \
        / ( x * math.sqrt(2 * math.pi) * par[1] )
    return f

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
    print ("quantile probabilities = ", prob)
    
    intl = 0.0
    x = x0
    q = []
    
    cnt = 0
    for i,p in enumerate(prob):
        while (intl < p and p < 1.0 and x <= x1 and cnt < 1E8):
            intl += dx * f(x)
            x += dx
            cnt += 1
        # the quantile value is somewhere between x - dx and x -> choose middle
        # q.append(x)    
        # q.append(x - 0.5 * dx)    
        q.append(x - dx)    
    
    print ("quantile values = ", q)
    return q, prob
          
# for a given grid with grid center positions grid_centers[i,j]:
# create random radii rad[i,j] = array of shape (Nx, Ny, no_spc)
# and the corresp. values of the distribution p[i,j]
# input:
# p_min, p_max: quantile probs to cut off e.g. (0.001,0.999)
# no_spc: number of super-droplets per cell (scalar!)
# dst: distribution to use
# par: params of the distribution par -> "None" possible if dst = dst(x)
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

# no_spcm = super-part/cell in modes [N_mode1_per_cell, N_mode2_per_cell, ...]
# no_spc = # super-part/cell
# no_spt = # SP total in full domain
# returns positions of particles of shape (2, no_spt)
def generate_random_positions(grid, no_spc, seed):
    no_spt = grid.no_cells_tot * no_spc
    rnd_x = np.random.rand(no_spt)
    rnd_z = np.random.rand(no_spt)
    dx = grid.steps[0]
    dz = grid.steps[1]
    x = []
    z = []
    n = 0
    for j in range(grid.no_cells[1]):
        z0 = grid.corners[1][0,j]
        for i in range(grid.no_cells[0]):
            x0 = grid.corners[0][i,j]
            for k in range(no_spc):
                x.append(x0 + dx * rnd_x[n])
                z.append(z0 + dz * rnd_z[n])
                n += 1
    pos = np.array([x, z])
    rel_pos = np.array([rnd_x, rnd_z])
    return pos, rel_pos

# # c "per cell", t "tot"
# # rad and weights are (Nx,Ny,no_spc) array
# # pos is (no_spt,) array
# def generate_particle_positions_and_dry_radii(grid, p_min, p_max, no_spc,
#                                               dst, par, r0, r1, dr, seed):
#     rad, weights = generate_random_radii(grid, p_min, p_max, no_spc,
#                                          dst, par, r0, r1, dr, seed)
    
#     return pos, rad, weights

#%%
def compute_profiles_T_p_rhod_S_without_liquid(z_, z_0_, p_0_, p_ref_, Theta_l_, r_tot_, SCALE_FACTOR_ = 1.0 ):

#    dz = grid_.steps[1] * SCALE_FACTOR_
#    z_0 = grid_.ranges[1,0]
    z_0 = z_0_
#    z_inversion_height = grid_.ranges[1,1]
#    zg = grid_.corners[1][0]
#    Nz = grid_.no_cells[1] + 1

    # constants:
    p_ref = p_ref_ # ref pressure for potential temperature in Pa
    p_0 = p_0_ # surface pressure in Pa
    p_0_over_p_ref = p_0 / p_ref

    r_tot = r_tot_ # kg water / kg dry air
    Theta_l = Theta_l_ # K

    kappa_tot = compute_kappa_air_moist(r_tot) # [-]
    kappa_tot_inv = 1.0 / kappa_tot # [-]
    p_0_over_p_ref_to_kappa_tot = p_0_over_p_ref**(kappa_tot) # [-]
    beta_tot = compute_beta_without_liquid(r_tot, Theta_l) # 1/m

    # in K
    T_0 = compute_temperature_from_potential_temperature_moist(Theta_l, p_0, p_ref, r_tot)

    #################################################

    # analytically integrated profiles
    # for hydrostatic system with constant water vapor mixing ration r_v = r_tot and r_l = 0
    T_over_Theta_l = p_0_over_p_ref_to_kappa_tot - beta_tot * (z_ - z_0)
    T = T_over_Theta_l * Theta_l
    p_over_p_ref = T_over_Theta_l**(kappa_tot_inv)
    p = p_over_p_ref * p_ref
    rho_dry = p * beta_tot \
            / (T_over_Theta_l * c.earth_gravity * kappa_tot )
    S = compute_pressure_vapor( rho_dry * r_tot, T ) / compute_saturation_pressure_vapor_liquid(T)
            
#    def compute_T_over_Theta_l_init( z_ ):
#        return p_0_over_p_ref_to_kappa_tot - beta_tot * (z_ - z_0)
#    def compute_p_over_p_ref_init( z_ ):
#        return compute_T_over_Theta_l_init(z_)**(kappa_tot_inv)
#    def compute_density_dry_init( z_ ):
#        return compute_p_over_p_ref_init(z_) * p_ref * beta_tot \
#                / (compute_T_over_Theta_l_init( z_ ) * earth_gravity * kappa_tot )
#    def compute_saturation_init( z_ ):
#        T = compute_T_over_Theta_l_init( z_ ) * Theta_l
#        return \
#            compute_pressure_vapor(compute_density_dry_init( z_ ) * r_tot, T) \
#            / compute_saturation_pressure_vapor_liquid(T)
            
    return T, p, rho_dry, S
###############################################################################
#%%
# p_0 = 101500 # surface pressure in Pa
# p_ref = 1.0E5 # ref pressure for potential temperature in Pa
# r_tot_0 = 7.5E-3 # kg water / kg dry air, r_tot = r_v + r_l (spatially const)
# Theta_l = 289.0 # K, Liquid potential temperature (spatially constant)
# n_p: list of number density of particles in the modes [n1, n2, ..]
# e.g. n_p = np.array([60.0E6, 40.0E6]) # m^3
# no_spcm = list of number of super particles per cell: [no_spc_1, no_spc_2, ..]
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

# grid_file_list = ["grid_basics.txt", "arr_file1.npy", "arr_file2.npy"]
# grid_file_list = [path + s for s in grid_file_list]

# particle_file = "stored_particles.txt"
# particle_file = path + particle_file

def initialize_grid_and_particles(
        x_min, x_max, z_min, z_max, dx, dy, dz,
        p_0, p_ref, r_tot_0, Theta_l,
        n_p, no_spcm, dst, dst_par, 
        P_min, P_max, r0, r1, dr, rnd_seed, reseed,
        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, path):
    # VERSION WITH PARTICLE PROPERTIES IN ARRAYS, not particle class
    ##############################################
    # 1. set base grid
    ##############################################
    # grid dimensions ("ranges")
    # note that the step size is fix.
    # The grid dimensions will be adjusted such that they are
    # AT LEAST x_max - x_min, etc... but may also be larger,
    # if the sizes are no integer multiples of the step sizes
    grid_ranges = [ [x_min, x_max],
    #                 [y_min, y_max], 
                    [z_min, z_max] ]
    grid_steps = [dx, dz]

    grid = Grid( grid_ranges, grid_steps, dy )
    grid.print_info()
    
    ##############################################
    # 2. Set initial profiles without liquid water
    ##############################################
    
    # INITIAL PROFILES
    
    # z_0 = grid.ranges[1,0]
    levels_z = grid.corners[1][0]
    centers_z = grid.centers[1][0]
    # Nz = grid.no_cells[1] + 1
    
    # for testing on a smaller grid: r_tot as array in centers
    r_tot_centers = np.ones_like( centers_z ) * r_tot_0
    # r_tot_centers = np.linspace(1.15, 1.3, np.shape(centers_z)[0]) * r_tot_0
    
    p_env_init_bottom = np.zeros_like(levels_z)
    p_env_init_bottom[0] = p_0
    
    p_env_init_center = np.zeros_like(centers_z)
    T_env_init_center = np.zeros_like(centers_z)
    rho_dry_env_init_center = np.zeros_like(centers_z)
    r_v_env_init_center = np.ones_like(centers_z) * r_tot_centers
    r_l_env_init_center = np.zeros_like(centers_z)
    S_env_init_center = np.zeros_like(centers_z)
    
    # p_env_init_center = np.zeros_like(levels_z)
    # T_env_init_center = np.zeros_like(levels_z)
    # rho_dry_env_init = np.zeros_like(levels_z)
    # r_v_env_init = np.ones_like(levels_z) * r_tot
    # r_l_env_init = np.zeros_like(r_v_env_init)
    # S_env_init = np.zeros_like(levels_z)
    # T_env_init = np.zeros_like(levels_z)
    # rho_dry_env_init = np.zeros_like(levels_z)
    # r_v_env_init = np.ones_like(levels_z) * r_tot
    # r_l_env_init = np.zeros_like(r_v_env_init)
    # S_env_init = np.zeros_like(levels_z)
    
    # T_env_init[0], p_env_init[0], rho_dry_env_init[0], S_env_init[0] =\
    #     compute_profiles_T_p_rhod_S_without_liquid(z_0, z_0, p_0, p_ref, Theta_l, r_tot )
    
    ##############################################
    # 3. Go through levels from the ground and place particles 
    ##############################################
    
    # start at level 0 (from surface!)
    
    # produce random numbers for relative location:
    np.random.seed( rnd_seed )
    
    # IN WORK: remove?
    V0 = grid.volume_cell
    
    # total number of real particles per mode in one grid cell:
    # total number of real particles in mode 'k' (= 1,2) in cell [i,j]
    # reference value at p_ref (marked by 0)
    no_rpcm_0 =  (np.ceil( V0*n_p )).astype(int)
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
    
    # particle_list_by_id = []
    
    # cell numbers is sorted as [ [0,0], [1,0], [2,0] ]  
    # i.e. keeping z- level fixed and run through x-range
    # cell_numbers = []
    
    # running particle ID
    ID = 0
    
    # number of super particles
    no_spcm = np.array(no_spcm) # input: no super part. per cell and mode
    no_spct = np.sum(no_spcm) # no super part. per cell (total)
    no_spt = no_spct * grid.no_cells_tot # no super part total in full domain
    
    ### generate particle radii and weights for the whole grid
    R_s, weights_R_s = generate_random_radii_multimodal(
                               grid, dst, dst_par, no_spcm, P_min, P_max, 
                               r0, r1, dr, rnd_seed, reseed)
    # convert to dry masses
#    print()
#    print("type(R_s)")
#    print(type(R_s))
#    print(R_s)
#    print()
                   
    m_s = compute_mass_from_radius(R_s, c.mass_density_NaCl_dry)
    w_s = np.zeros_like(m_s) # init weight fraction
    m_w = np.zeros_like(m_s) # init particle water masses to zero
    m_p = np.zeros_like(m_s) # init particle full mass
    R_p = np.zeros_like(m_s) # init particle radius
    rho_p = np.zeros_like(m_s) # init particle density
    xi = np.zeros_like(m_s).astype(int)
    
    mass_water_liquid_levels = []
    mass_water_vapor_levels = []
    # start with a cell [i,j]
    # we start with fixed 'z' value to assign levels as well
    iter_cnt_max = 0
    iter_cnt_max_level = 0
    # maximal allowed iter counts
    # iter_cnt_limit = 500
    rho_dry_0 = p_0 / (c.specific_gas_constant_air_dry * 293)
    for j in range(grid.no_cells[1]):
    #     print('next j')
    #     we are now in column 'j', fixed z-level
        n_top = j + 1
        n_bot = j
        
        z_bot = levels_z[n_bot]
        z_top = levels_z[n_top]
        
        r_tot = r_tot_centers[j]
        
        kappa_tot = compute_kappa_air_moist(r_tot) # [-]
        kappa_tot_inv = 1.0 / kappa_tot # [-]
        
    #     T_prev = T_env_init[n_prev]
        # p at bottom of level
        p_bot = p_env_init_bottom[n_bot]
    #     rho_dry_prev = rho_dry_env_init[n_prev]
    #     S_prev = S_env_init[n_prev]
    #     r_v_prev = r_v_env_init[n_prev]
    
        # initial guess at borders of new level from analytical integration without liquid: r_v = r_tot
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
        
        # ambient properties for this level
        # diffusion_constant = compute_diffusion_constant( T_avg, p_avg )
        # thermal_conductivity_air = compute_thermal_conductivity_air(T_avg)
        # specific_heat_capacity_air = compute_specific_heat_capacity_air_moist(r_v_avg)
        # adiabatic_index = 1.4
        # accomodation_coefficient = 1.0
        # condensation_coefficient = 0.0415
        
        # calc mass dry from 
        # int dV (rho_dry) = dx * dy * int dz rho_dry
        # rho_dry = dp/dz / [g*( 1+r_tot )]
        # => m_s = dx dy / (g*(1+r_tot)) * (p(z_1)-p(z_2))
        mass_air_dry_level = grid.sizes[0] * dy * (p_bot - p_top )\
                             / ( c.earth_gravity * (1 + r_tot) )
    #     mass_air_dry_level = rho_dry_avg * grid.sizes[0] * dy * dz # in kg !!!
        mass_water_vapor_level = r_v_avg * mass_air_dry_level # in kg
    #     mass_water_vapor = 1.0E18*r_v_avg * rho_dry_avg\
    #                        * grid.sizes[0] * dy * dz # in 10^-18 kg !!!
        
        mass_water_liquid_level = 0.0
        mass_particles_level = 0.0
        dm_l_level = 0.0
        dm_p_level = 0.0
        
        print('\nlevel ', j, ', S_env_init0 = ', S_avg)
        ########################################################
        # 3a. (first initialization setting S_eq = S_amb if possible)
        ########################################################
    #     print(mass_water_vapor_level)
        no_rpcm = np.rint( no_rpcm_0 * rho_dry_avg / rho_dry_0 ).astype(int)
        
        # multiplicities in level j:
        
        
        # for level j: set mass_fraction, particle mass, water mass for each pt

                                 
        for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
            # print("weights_R_s[l][:,j]")
            # print("no_rpcm[l]")
            # print(weights_R_s[l][:,j])
            # print(no_rpcm[l])
            xi[l][:,j] = np.rint(weights_R_s[l][:,j] * no_rpcm[l]).astype(int)
            # print("xi[l][:,j]")
            # print(xi[l][:,j])
            # initial weight fraction of this level dependent on S_amb
            # -> try to place the particles with m_w such that S = S_eq
            w_s[l][:,j] = compute_initial_mass_fraction_solute_NaCl(
                          R_s[l][:,j], S_avg, T_avg)
            m_p[l][:,j] = m_s[l][:,j] / w_s[l][:,j]
            rho_p[l][:,j] = compute_density_particle(w_s[l][:,j], T_avg)
            R_p[l][:,j] = compute_radius_from_mass(m_p[l][:,j], rho_p[l][:,j])
            m_w[l][:,j] = m_p[l][:,j] - m_s[l][:,j]
            dm_l_level += np.sum(m_w[l][:,j] * xi[l][:,j])
            dm_p_level += np.sum(m_p[l][:,j] * xi[l][:,j])
        
        # if j == 9:
            # print()
            # print("xi at lvl 9:")
            # print(xi)
            # print()
        # no_sub_particles_per_particle = np.rint(  * rho_dry_avg / rho_dry_0).astype(int)
        print('level = ', j)
        # print('no_sub_particles_per_particle = ')
        # print(f'{no_sub_particles_per_particle[0]:.2e} {no_sub_particles_per_particle[1]:.2e}')
        
        
    #     for i in range(grid.no_cells[0]):
    # #         print('next i')
    #         # cell [i,j]
    #         # cell numbers is sorted as [ [0,0], [1,0], [2,0] ]  
    #         # i.e. keeping z- level fixed and run through x-range
    #         cell_numbers.append ( [i,j] )
            
    #         # take average values for cell [i,j]
    # #         T_amb = T_init0[j+1] - T_init0[j]
    # #         p_amb = p_init0[j+1] - p_init0[j]
    # #         S_amb = S_init0[j+1] - S_init0[j]
    # #         rv_amb = r_v T_init0[j+1] - T_init0[j]
    # #         p_amb = grid.pressure[i,j]
    # #         S_amb = grid.saturation[i,j]
    # #         rv_amb = grid.mixing_ratio_water_vapor[i,j]
    # #         T_amb = grid.temperature[i,j]
    # #         p_amb = grid.pressure[i,j]
    # #         S_amb = grid.saturation[i,j]
    # #         rv_amb = grid.mixing_ratio_water_vapor[i,j]
            
    #     # N_sp =  number of super particles in the modes = [N_1, N_2]
    #         for k, N_k in enumerate(no_super_particles_cell):
    # #         for k, N_k in enumerate(N_p_cell):
    # #             print('next k = ', k)
    #             # cell [i,j] mode k
    #             sigma = par_sigma[k]
    #             r0 = par_r0[k]
                
    #             # produce array of arrays with N_k entries of arrays dim 2 with random numbers in [0.0,1.0)   [open, closed)
    #             relative_locations = np.random.rand(N_k,2)
                
    #             # compute from distribution: ln(r(r_i)) is normal distributed with mu = 0, sigma = sigma_i for i = 1,2
    #             # see Arabas 2015
    #             # produce array with N_k entries of random numbers drawn 
    #             # from normal distribution with mu = 0.0, std_dev  = sigma
    #             radii_solute_dry = np.random.normal(0.0, sigma, N_k)
    # #             print( radii_solute_dry )
    #             radii_solute_dry = np.exp( radii_solute_dry )
    #             radii_solute_dry *= r0
    #             # now we have N_k values for the dry radii
                
    #             # loop through N_k:
    #             for n_k in range(N_k):
    # #                 print('next n_k = ', n_k)
    #                 # cell [i,j], mode k, particle number n_k in mode k
    #                 relative_location = relative_locations[n_k]
    #                 location = grid.compute_location(i,j,*relative_location)
    #                 velocity = grid.interpolate_velocity_from_cell_bilinear(i,j,*relative_location )
    #                 radius_solute_dry = radii_solute_dry[n_k]
    
    # #                 droplet = Particle(ID, location, velocity, grid,
    # #                    T_amb, p_amb, S_amb,
    # #                    diffusion_constant, thermal_conductivity_air,
    # #                    specific_heat_capacity_air, adiabatic_index,
    # #                    accomodation_coefficient, condensation_coefficient,
    # #                    radius_solute_dry)
    # #                 print (i,j,k, n_k, ID, len(particle_list_by_id))
                    
    #                 particle = Particle(ID, no_sub_particles_per_particle[k], location, velocity, grid,
    #                                                T_avg, p_avg, S_avg,
    #                                                diffusion_constant, thermal_conductivity_air,
    #                                                specific_heat_capacity_air, adiabatic_index,
    #                                                accomodation_coefficient, condensation_coefficient,
    #                                                radius_solute_dry)
    # #                 particle_list_by_id.append( Particle(ID, location, velocity, grid,
    # #                                                T_avg, p_avg, S_avg,
    # #                                                diffusion_constant, thermal_conductivity_air,
    # #                                                specific_heat_capacity_air, adiabatic_index,
    # #                                                accomodation_coefficient, condensation_coefficient,
    # #                                                radius_solute_dry) )
    #                 particle_list_by_id.append( particle )
    # #                 print('assigned ID', ID)
    #                 ids_in_cell[i][j].append(ID)
    #                 ids_in_level[j].append(ID)
    # #                 dm_l_level += particle_list_by_id[ID].mass_water
    # #                 dm_p_level += particle_list_by_id[ID].mass
    #                 multiplicity_ID = particle.multiplicity
    #                 dm_l_level += particle.mass_water * multiplicity_ID
    #                 dm_p_level += particle.mass * multiplicity_ID
                    
    # #                 print (i,j,k, n_k, ID, len(particle_list_by_id))
    # #                 about the ID increase bug: the origin was from overuse of ID: 
    # #                 once as stored variable increasing during looping and once as loop index below..
    #                 ID += 1
    # #                 print('ID increases to', ID)
    # #                 print (i,j,k, n_k, ID, len(particle_list_by_id))
    
        # convert from 10^-18 kg to kg
        dm_l_level *= 1.0E-18 
        dm_p_level *= 1.0E-18 
        mass_water_liquid_level += dm_l_level
        mass_particles_level += dm_p_level
        
        # now we distributed the particles between levels 0 and 1
        # thereby sampling liquid water, i.e. the mass of water vapor has dropped
        mass_water_vapor_level -= dm_l_level
        
        # and the ambient fluid is heated by Q = m_l * L_v = C_p dT
        # C_p = m_dry_end*c_p_dry + m_v*c_p_v + m_p_end*c_p_p
        heat_capacity_level =\
            mass_air_dry_level * c.specific_heat_capacity_air_dry_NTP\
            + mass_water_vapor_level * c.specific_heat_capacity_water_vapor_20C\
            + mass_particles_level * c.specific_heat_capacity_water_NTP
        heat_of_vaporization = compute_heat_of_vaporization(T_avg)
        
        dT_avg = dm_l_level * heat_of_vaporization / heat_capacity_level
        
        # assume homogeneous heating: bottom, mid and top are heated equally
        # i.e. the slope (lapse rate) of the temperature in the level
        # remains constant,
        # but the whole (linear) T-curve is shifted by dT_avg
        T_avg += dT_avg
        T_bot += dT_avg
        T_top += dT_avg
        
        r_v_avg = mass_water_vapor_level / mass_air_dry_level
        
    #     R_m_avg = compute_specific_gas_constant_air_moist(r_v_avg)
        
        # EVEN BETTER:
        # assume linear T-profile
        # then (p/p1) = (T/T1) ^ [(eps + r_t)/ (kappa_t * (eps + r_v))]
        # BETTER: exponential for const T and r_v:
        # p_2 = p_1 np.exp(-a/b * dz/T)
        # also poss.:
        # implicit formula for p
        # p_n+1 = p_n (1 - a/(2b) * dz/T) / (1 + a/(2b) * dz/T)
        # a = (1 + r_t) g, b = R_d(1 + r_v/eps)
        # chi = ( 1 + r_tot / (1 + r_v_avg ) )
        # this is equal to expo above up to order 2 in taylor!
        
        # IN WORK: check if kappa_tot_inv is right or it should be kappa(r_v)

        p_top = p_bot * (T_top/T_bot)**( kappa_tot_inv * ( epsilon_gc + r_tot )\
                / (epsilon_gc + r_v_avg) )
        
        p_avg = 0.5 * (p_bot + p_top)
        
        # from integration of dp/dz = - (1 + r_tot) g rho_dry
        rho_dry_avg = (p_bot - p_top) / ( dz * (1 + r_tot) * c.earth_gravity )
        mass_air_dry_level = grid.sizes[0] * dy * dz * rho_dry_avg
        
        r_l_avg = mass_water_liquid_level / mass_air_dry_level
        r_v_avg = r_tot - r_l_avg
        mass_water_vapor_level = r_v_avg * mass_air_dry_level
        
        e_s_avg = compute_saturation_pressure_vapor_liquid(T_avg)
        
        S_avg = compute_pressure_vapor( rho_dry_avg * r_v_avg, T_avg ) \
                / e_s_avg
        
    #     rho_m_avg = p_avg / (R_m_avg * T_avg)
    #     rho_tot_avg = rho_m_avg * ((1 + r_tot)/(1 + r_v_avg))
    #     p = p_prev  - earth_gravity * rho_tot_avg * dz
        
        ########################################################
        # 3b. 
        # this was the initial placement of particles
        # now comes the saturation adjustment incl. condensation/vaporization
        # due to supersaturation/subsaturation
        # note that there will also be subsaturation if S is smaller than S_act,
        # because water vapor was taken from the atm. in the intitial
        # particle placement step
        ########################################################
    
        # adjust material properties etc. in level due to new T and p:
        D_v = compute_diffusion_constant( T_avg, p_avg )
        K = compute_thermal_conductivity_air(T_avg)
        # c_p = compute_specific_heat_capacity_air_moist(r_v_avg)
        L_v = compute_heat_of_vaporization(T_avg)
        sigma_w = compute_surface_tension_water(T_avg)
        # diffusion_constant = compute_diffusion_constant( T_avg, p_avg )
        # thermal_conductivity_air = compute_thermal_conductivity_air(T_avg)
        # specific_heat_capacity_air = compute_specific_heat_capacity_air_moist(r_v_avg)
        # heat_of_vaporization = compute_heat_of_vaporization(T_avg)
        
        
        # adiabatic_index = 1.4
        # accomodation_coefficient = 1.0
        # condensation_coefficient = 0.0415
        
        # loop until the change in dm_l_level is sufficiently small:
        # need to define criterium -> use relative change in liquid water
        dm_l_level = mass_water_liquid_level
        print('dt_init = ', dt_init)
        print('level ', j, ', S_env_init = ', S_avg)
        iter_cnt = 0
        
        ### NEW
        grid.mixing_ratio_water_vapor[:,j] = r_v_avg
        grid.mixing_ratio_water_liquid[:,j] = r_l_avg
        grid.saturation_pressure[:,j] = e_s_avg
        grid.saturation[:,j] = S_avg
    
        while ( np.abs(dm_l_level/mass_water_liquid_level) > 1e-5
                and iter_cnt < iter_cnt_limit ):
    #         print('level ', j, ', iter_cnt = ', iter_cnt,
    #               ', S_avg = ', S_avg,
    #               '\ndm_l_level = ', dm_l_level, ', m_l_level = ', mass_water_liquid_level,
    #               '\ndm_l_level/mass_water_liquid_level = ', dm_l_level/mass_water_liquid_level)
    #         print('level ', j, ', iter_cnt = ', iter_cnt, '\nabs(dm_l_level/mass_water_liquid_level) = ', np.abs(dm_l_level/mass_water_liquid_level))
            #         print('np.abs(dm_l_level/mass_water_liquid_level) = ', np.abs(dm_l_level/mass_water_liquid_level))
            ## loop over particles in level:
            dm_l_level = 0.0
            dm_p_level = 0.0
            
            D_v = compute_diffusion_constant( T_avg, p_avg )
            K = compute_thermal_conductivity_air(T_avg)
            # c_p = compute_specific_heat_capacity_air_moist(r_v_avg)
            L_v = compute_heat_of_vaporization(T_avg)
            sigma_w = compute_surface_tension_water(T_avg)
    
            # set arbitrary maximum saturation during spin up to avoid overshoot at 
            # high saturations, since S > 1.05 can happen initially
#            S_avg2 = np.min([S_avg, S_init_max ])
    #         print('S_avg2 = ', S_avg2, '\n')
            # VERY IMPORTANT: use 'ID_j' (or similar) NOT 'ID' as loop variable,
            # because ID is still a running counter in the outer 'j' loop
            
            # we have m_s, m_p, 
            
            for i in range(grid.no_cells[0]):
                cell = (i,j)
                e_s_avg = grid.saturation_pressure[cell]
                S_avg = grid.saturation[cell]
                S_avg2 = np.min([S_avg, S_init_max ])

                
                for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
                    m_w_cell = m_w[l][cell]
                    m_s_cell = m_s[l][cell]
                    
                    R_p_cell, w_s_cell, rho_p_cell =\
                        compute_R_p_w_s_rho_p(m_w_cell, m_s_cell, T_avg)
                    
                    dm_l, gamma_ =\
                    compute_delta_water_liquid_and_mass_rate_implicit_Newton_full(
                        dt_init, Newton_iterations, m_w_cell,
                        m_s_cell, w_s_cell, R_p_cell, T_avg,
                        rho_p_cell,
                        T_avg, p_avg, S_avg2, e_s_avg, L_v, K, D_v, sigma_w)
                    
                    m_w[l][cell] += dm_l
                    # m_p[l][cell] += dm_l
                    
                    # particle_list_by_id[ID_j].mass_water += dm_l
                    # particle_list_by_id[ID_j].mass += dm_l
                    # if (particle_list_by_id[ID_j].mass_water < 0.0):
                    #     print('ID = ', ID_j, ', mass_water = ', particle_list_by_id[ID_j].mass_water )
                    # multiplicity_ID = particle_list_by_id[ID_j].multiplicity
                
#                grid.mixing_ratio_water_liquid[cell] 
                
                    dm_l_level += np.sum(dm_l * xi[l][cell])
                    dm_p_level += np.sum(dm_l * xi[l][cell])
                    # dm_l_level += dm_l * multiplicity_ID
                    # dm_p_level += dm_l * multiplicity_ID
                
    #         for ID_j in ids_in_level[j]:
    #             # implicit mass growth with fixed ambient values
    #             particle = particle_list_by_id[ID_j]
    #             cell = tuple(particle.cell)
    #             e_s_avg = grid.saturation_pressure[cell]
    #             S_avg = grid.saturation[cell]
    #             S_avg2 = np.min([S_avg, S_init_max ])
    #             mass_water = particle_list_by_id[ID_j].mass_water
    #             mass_solute = particle_list_by_id[ID_j].mass_solute            
    #             particle_list_by_id[ID_j].temperature = T_avg
    #             mass_particle = particle_list_by_id[ID_j].mass
    #             mass_fraction_solute = mass_solute / mass_particle
    #             density_particle = compute_density_particle(mass_fraction_solute, T_avg)
    #             radius_particle = compute_radius_from_mass(mass_particle, density_particle)
    #             surface_tension = compute_surface_tension_water(T_avg)
    # #             if (j==46 and ID_j == 17252):
    # #                 print(mass_water)
    # #                 print(mass_solute)
    # #                 print(T_avg)
    # #                 print(p_avg)
    # #                 print(S_avg)
    # #                 print(e_s_avg)
    
    #             # condensation step:
    #             # mass change liquid water dm_l in femtogram during timestep dt_init
    #             # particles are held fixed at position
    # #             if (j==46 and ID_j == 17252):
    # #                 dm_l = compute_delta_water_liquid_implicit_linear( 
    # #                                                       dt_init, mass_water, mass_solute, #  in femto gram
    # #                                                       T_avg, # this is the particle temperature, here equal to T_amb
    # #                                                       T_avg, p_avg,
    # #                                                       S_avg2, e_s_avg,
    # #                                                       diffusion_constant,
    # #                                                       thermal_conductivity_air,
    # #                                                       specific_heat_capacity_air,
    # #                                                       adiabatic_index,
    # #                                                       accomodation_coefficient,
    # #                                                       condensation_coefficient, 
    # #                                                       heat_of_vaporization, True)
    # #             else:
    # #             dm_l = compute_delta_water_liquid_imex_linear( 
    #             # we use implicit here, because it takes much less iterations
    #             # for the low levels, because there we have small S and m_w and in that
    #             # region a LARGE NEGATIVE d/dm (dmdt) leading to oscillations 
    #             # IN WORK: further tests with adjusted dt_init ???
    #             # IN WORK: also test with explicit euler only ??
    #             # IN WORK: INSERTED NEWTON IMPLICIT HERE -> TESTING with 6 iterations
    #             dm_l, gamma_ = compute_delta_water_liquid_and_mass_rate_implicit_Newton_full(dt_init, Newton_iterations, mass_water, mass_solute,
    #                                                               mass_particle, mass_fraction_solute, radius_particle,
    #                                                               T_avg, density_particle,
    #                                                               T_avg, p_avg,
    #                                                               S_avg2, e_s_avg,
    #                                                               diffusion_constant,
    #                                                               thermal_conductivity_air,
    #                                                               specific_heat_capacity_air,
    #                                                               heat_of_vaporization,
    #                                                               surface_tension,
    #                                                               adiabatic_index,
    #                                                               accomodation_coefficient,
    #                                                               condensation_coefficient)
                
#                initialize_profiles_and_write_to_file_kinematic_2D_ICMW_2012_case1_with_droplets(
#        x_min, x_max, z_min, z_max, dx, dy, dz, p_0, p_ref, r_tot_0, Theta_l,
#        n_p, no_super_particles_cell, par_sigma, par_r0, rnd_seed, S_init_max, dt_init, iter_cnt_limit,
#        grid_file_list, particle_file, active_ids_file, removed_ids_file)
                
                 # REPLACED BY NEW IMPLICIT NEWTON METHOD
#                dm_l = compute_delta_water_liquid_implicit_linear( 
#                                                          dt_init, mass_water, mass_solute, #  in femto gram
#                                                          T_avg, # this is the particle temperature, here equal to T_amb
#                                                          T_avg, p_avg,
#                                                          S_avg2, e_s_avg,
#                                                          diffusion_constant,
#                                                          thermal_conductivity_air,
#                                                          specific_heat_capacity_air,
#                                                          adiabatic_index,
#                                                          accomodation_coefficient,
#                                                          condensation_coefficient, 
#                                                          heat_of_vaporization)
    
            # IN WORK: check if the temperature, pressure etc is right BEFORE the level starts and check the addition to these values!
            # after 'sampling' water from all particles: heat the cell volume:
                # convert from 10^-18 kg to kg
            dm_l_level *= 1.0E-18 
            dm_p_level *= 1.0E-18 
            mass_water_liquid_level += dm_l_level
            mass_particles_level += dm_p_level
    
            # now we distributed the particles between levels 0 and 1
            # thereby sampling liquid water,
            # i.e. the mass of water vapor has dropped
            mass_water_vapor_level -= dm_l_level
    
            # and the ambient fluid is heated by Q = m_l * L_v = C_p dT
            # C_p = m_dry_end*c_p_dry + m_v*c_p_v + m_p_end*c_p_p
            heat_capacity_level =\
                mass_air_dry_level * c.specific_heat_capacity_air_dry_NTP \
                + mass_water_vapor_level * c.specific_heat_capacity_water_vapor_20C \
                + mass_particles_level * c.specific_heat_capacity_water_NTP
            heat_of_vaporization = compute_heat_of_vaporization(T_avg)
    
            dT_avg = dm_l_level * heat_of_vaporization / heat_capacity_level
    
            # assume homogeneous heating: bottom, mid and top are heated equally
            # i.e. the slope (lapse rate) of the temperature in the level remains constant,
            # but the whole (linear) T-curve is shifted by dT_avg
            T_avg += dT_avg
            T_bot += dT_avg
            T_top += dT_avg
    
            r_v_avg = mass_water_vapor_level / mass_air_dry_level
    
        #     R_m_avg = compute_specific_gas_constant_air_moist(r_v_avg)
    
            # EVEN BETTER:
            # assume linear T-profile
            # then (p/p1) = (T/T1) ^ [(eps + r_t)/ (kappa_t * (eps + r_v))]
            # BETTER: exponential for const T and r_v:
            # p_2 = p_1 np.exp(-a/b * dz/T)
            # also poss.:
            # implicit formula for p
            # p_n+1 = p_n (1 - a/(2b) * dz/T) / (1 + a/(2b) * dz/T)
            # a = (1 + r_t) g, b = R_d(1 + r_v/eps)
            # chi = ( 1 + r_tot / (1 + r_v_avg ) )
            # this is equal to expo above up to order 2 in taylor!
    
            # IN WORK: check if kappa_tot_inv is right or it should be kappa(r_v)
            # -> should be right, because the lapse rate dT/dt ~ beta_tot is not altered!
            # the T curve is shifted upwards in total
            p_top = p_bot * (T_top/T_bot)**( kappa_tot_inv * ( epsilon_gc + r_tot ) / (epsilon_gc + r_v_avg) )
    
            p_avg = 0.5 * (p_bot + p_top)
    
            # from integration of dp/dz = - (1 + r_tot) * g * rho_dry
            rho_dry_avg = (p_bot - p_top) / ( dz * (1 + r_tot) * c.earth_gravity )
            mass_air_dry_level = grid.sizes[0] * dy * dz * rho_dry_avg
            mass_air_dry_cell = dx * dy * dz * rho_dry_avg
            
            r_l_avg = mass_water_liquid_level / mass_air_dry_level
            r_v_avg = r_tot - r_l_avg
            
            grid.mixing_ratio_water_liquid[:,j].fill(0.0)
            for i in range(grid.no_cells[0]):
                cell = (i,j)
                for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
                    grid.mixing_ratio_water_liquid[cell] +=\
                        np.sum(m_w[l][cell] * xi[l][cell])
                    
            # for ID_ in ids_in_level[j]:
            #     par = particle_list_by_id[ID_]
            #     cell = tuple(par.cell)
            #     grid.mixing_ratio_water_liquid[cell] += par.mass_water * par.multiplicity
    
            grid.mixing_ratio_water_liquid[:,j] *= 1.0E-18 / mass_air_dry_cell
            grid.mixing_ratio_water_vapor[:,j] =\
                r_tot - grid.mixing_ratio_water_liquid[:,j]
            
            grid.saturation_pressure[:,j] =\
                compute_saturation_pressure_vapor_liquid(T_avg)
            grid.saturation[:,j] =\
                compute_pressure_vapor(
                    rho_dry_avg * grid.mixing_ratio_water_vapor[:,j], T_avg )\
                / grid.saturation_pressure[:,j]
            
            mass_water_vapor_level = r_v_avg * mass_air_dry_level
            
            e_s_avg = compute_saturation_pressure_vapor_liquid(T_avg)
    
            S_avg = compute_pressure_vapor( rho_dry_avg * r_v_avg, T_avg ) \
                    / e_s_avg
            # diffusion_constant = compute_diffusion_constant(T_avg, p_avg)
            # thermal_conductivity_air = compute_thermal_conductivity_air(T_avg)
            # specific_heat_capacity_air = compute_specific_heat_capacity_air_moist(r_v_avg)
            # heat_of_vaporization = compute_heat_of_vaporization(T_avg)
            
            iter_cnt += 1
        
        if iter_cnt_max < iter_cnt: 
            iter_cnt_max = iter_cnt
            iter_cnt_max_level = j
    #     print('iter_cnt_max: ', iter_cnt_max)
        print('level ', j, ', iter_cnt = ', iter_cnt,
              ', S_avg = ', S_avg,
              '\ndm_l_level = ', dm_l_level,
              ',m_l_level = ', mass_water_liquid_level,
              '\ndm_l_level/mass_water_liquid_level = ',
              dm_l_level/mass_water_liquid_level)
    #     print('level ', j, ', iter_cnt = ', iter_cnt,
    #           '\ndm_l_level/mass_water_liquid_level = ', dm_l_level/mass_water_liquid_level)
        
        mass_water_liquid_levels.append(mass_water_liquid_level)
        mass_water_vapor_levels.append( mass_water_vapor_level )
    
        p_env_init_bottom[n_top] = p_top
        p_env_init_center[j] = p_avg
        T_env_init_center[j] = T_avg
        r_v_env_init_center[j] = r_v_avg
        r_l_env_init_center[j] = r_l_avg
        rho_dry_env_init_center[j] = rho_dry_avg
        S_env_init_center[j] = S_avg
    
    print('')
    print('iter count max = ', iter_cnt_max, ' level = ', iter_cnt_max_level)
    # total number of particles in grid
    no_particles_tot = np.size(m_w)
    print('last particle ID = ', len(m_w.flatten()) - 1 )
    print ('no_super_particles_tot placed = ', no_particles_tot)
    # print('no_super_particles_tot should',
    #       no_super_particles_tot  )
    print('no_cells x N_p_cell_tot =', np.sum(no_rpct_0)*grid.no_cells[0]\
                                                        *grid.no_cells[1] )
    for i in range(grid.no_cells[0]):
        grid.pressure[i] = p_env_init_center
        grid.temperature[i] = T_env_init_center
        grid.mass_density_air_dry[i] = rho_dry_env_init_center
#        grid.mixing_ratio_water_vapor[i] = r_v_env_init_center
    #     grid.mixing_ratio_water_liquid[i] = r_l_env_init_center # see below
#        grid.saturation[i] = S_env_init_center
        
    # we need a special treatment for r_l because, the atmosphere is generated 
    # as z-profile only, but the particle sizes vary also in x-direction 
#    for par in particle_list_by_id:
#        cell = tuple(par.cell)
#        grid.mixing_ratio_water_liquid[cell] += par.mass_water * par.multiplicity
#    
#    grid.mixing_ratio_water_liquid *= 1.0E-18 / (grid.volume_cell * grid.mass_density_air_dry)
    p_dry = grid.mass_density_air_dry * c.specific_gas_constant_air_dry\
            * grid.temperature
            
    
    grid.potential_temperature = grid.temperature\
                                 * ( 1.0E5 / p_dry )**kappa_air_dry
                                 
    grid.saturation_pressure =\
        compute_saturation_pressure_vapor_liquid(grid.temperature)
    
       
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
    
    ########################################################
    # 4. 
    # set mass flux and velocity grid
    ######################################################## 
    
#    print()
#    print()
#    print()
#    print("xi")
#    print(xi)
#    print()
#    print()
#    print()
    
    j_max = 0.6 * np.sqrt(grid.sizes[0] * grid.sizes[0] +
                          grid.sizes[1] * grid.sizes[1])\
                / np.sqrt(2.0 * 1500.0*1500.0)
    print()
    print('j_max')
    print(j_max)
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
    # we need to add a dummy density for the center in a cell above the highest cell,
    # because velocity has the same dimensions as grid.corners and not grid.centers
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
    pos, rel_pos = generate_random_positions(grid, no_spct, rnd_seed)
    # init velocities
    #vel = np.zeros_like(pos)
    
    # generate cell array [ [i1,j1], [i2,j2], ... ] and flatten particle masses
    ID = 0
    m_w_flat = np.zeros(len(pos[0]))
    m_s_flat = np.zeros(len(pos[0]))
    xi_flat = np.zeros(len(pos[0]), dtype = np.int64)
    cell_list = np.zeros( (2, len(pos[0])), dtype = int )
    
    # @njit()
    # def flatten_m_w_m_s(m_w,m_s,m_w_flat,m_s_flat, no_spcm, grid_no_cells):
    #     for j in range(grid_no_cells[1]):
    #         for i in range(grid_no_cells[0]):
    #             for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
    #                 for n, m_w_ in m_w[l][i,j]:
    #                     cell_list[0,ID] = i
    #                     cell_list[1,ID] = j
    #                     m_w_flat[ID] = m_w[l][i,j][n]
    #                     m_s_flat[ID] = m_s[l][i,j][n]
                        
    for j in range(grid.no_cells[1]):
        for i in range(grid.no_cells[0]):
            for l, N_l in enumerate( no_spcm[ np.nonzero(no_spcm) ] ):
                if isinstance(m_w[l][i,j], (list, tuple, np.ndarray)):
                    for n, m_w_ in enumerate(m_w[l][i,j]):
                        cell_list[0,ID] = i
                        cell_list[1,ID] = j
                        m_w_flat[ID] = m_w[l][i,j][n]
                        m_s_flat[ID] = m_s[l][i,j][n]
                        xi_flat[ID] = xi[l][i,j][n]
                        ID += 1
                else:
                    cell_list[0,ID] = i
                    cell_list[1,ID] = j
                    m_w_flat[ID] = m_w[l][i,j]
                    m_s_flat[ID] = m_s[l][i,j]
                    xi_flat[ID] = xi[l][i,j]
                    ID += 1
    
    vel = interpolate_velocity_from_cell_bilinear_jit(cell_list, rel_pos,
                                                      grid.velocity,
                                                      grid.no_cells)
    
    # for particle_ in particle_list_by_id:
    #     cell = tuple(particle_.cell)
    #     particle_.velocity = np.array(grid.interpolate_velocity_from_cell_bilinear(*particle_.cell, *particle_.relative_location ))
    #     particle_.temperature = grid.temperature[cell]
    #     particle_.mass_fraction_solute = particle_.mass_solute / particle_.mass
    #     particle_.update_physical_properties()
    #     particle_.radius = compute_radius_from_mass(particle_.mass, particle_.density)
    #     particle_.equilibrium_temperature = particle_.temperature
        
#    grid_file_list = ["grid_basics.txt", "arr_file1.npy", "arr_file2.npy"]
#    grid_file_list = [path + s for s in grid_file_list  ]
#    particle_file = "stored_particles.txt"
#    particle_file = path + particle_file
    
    active_ids = list(range(len(m_s_flat)))
    removed_ids = []
    
#    print(grid_file_list)
#    print(particle_file)
#    
#    save_grid_to_files(grid, 0.0, *grid_file_list)
#    # grid = load_grid_from_files(*grid_file_list)
#    save_particle_list_to_files(particle_list_by_id, active_ids, removed_ids, particle_file, active_ids_file, removed_ids_file)
    # print("active_ids")
    # print(active_ids)
    t = 0
    save_grid_and_particles_full(t, grid, pos, cell_list, vel,
                                 m_w_flat, m_s_flat, xi_flat,
                                 active_ids, removed_ids, path)
    
    return grid, pos, cell_list, vel, m_w_flat, m_s_flat, xi_flat, active_ids, removed_ids

#%%

### storage directories -> need to assign "simdata_path" and "fig_path"
# my_OS = "Linux_desk"
my_OS = "Mac"

if(my_OS == "Linux_desk"):
    home_path = '/home/jdesk/'
    simdata_path = home_path + "OneDrive/python/sim_data/"
    fig_path = home_path + 'Onedrive/Uni/Masterthesis/latex/Report/Figures/'
elif (my_OS == "Mac"):
    home_path = "/Users/bohrer/"
    simdata_path = home_path + "OneDrive - bwedu/python/sim_data/"
    fig_path = home_path + 'OneDrive - bwedu/Uni/Masterthesis/latex/Report/Figures/'

#simdata_path = "/home/jdesk/OneDrive/python/sim_data/"

# domain size
x_min = 0.0
x_max = 1500.0
z_min = 0.0
z_max = 1500.0

# grid steps
dx = 20.0
dy = 1.0
dz = 20.0

p_0 = 101500 # surface pressure in Pa
p_ref = 1.0E5 # ref pressure for potential temperature in Pa
r_tot_0 = 7.5E-3 # kg water / kg dry air
# r_tot_0 = 22.5E-3 # kg water / kg dry air
# r_tot_0 = 7.5E-3 # kg water / kg dry air
Theta_l = 289.0 # K
# number density of particles mode 1 and mode 2:
n_p = np.array([60.0E6, 100.0E6]) # m^3
# n_p = np.array([60.0E6, 40.0E6]) # m^3

# no_super_particles_cell = [N1,N2] is a list with N1 = no super part. per cell in mode 1 etc.
no_spcm = np.array([10, 10])
# no_super_particles_cell = [0, 4]


# parameters of log-normal distribution:
dst = dst_log_normal
# in log(mu)
par_sigma = np.log( [1.4,1.6] )
# par_sigma = np.log( [1.0,1.0] )
# in mu
par_r0 = 0.5 * np.array( [0.04, 0.15] )
dst_par = []
for i,sig in enumerate(par_sigma):
    dst_par.append([par_r0[0],sig])

P_min = 0.01
P_max = 0.99

dr = 1E-4
r0 = dr
r1 = 10 * par_sigma

reseed = False
rnd_seed = 4711

## for initialization phase
S_init_max = 1.05
dt_init = 0.1 # s
# number of iterations for the root finding (full) Newton algorithm in the implicit method
Newton_iterations = 1
# maximal allowed iter counts in initial particle water take up to equilibrium
iter_cnt_limit = 800

##### save sim data to files

folder = "190506/test1/"
path = simdata_path + folder
# path = folder1 + folder2
#####

grid_para_file = path + "grid_paras.txt"
paras = [x_min, x_max, z_min, z_max, dx, dy, dz, p_0, p_ref, r_tot_0,
         Theta_l, n_p,
         no_spcm,
         par_sigma, par_r0, rnd_seed, S_init_max, dt_init, iter_cnt_limit]
para_names = 'x_min, x_max, z_min, z_max, dx, dy, dz, p_0, p_ref, r_tot_0, \
Theta_l, n_p, no_super_particles_cell, \
par_sigma, par_r0, rnd_seed, S_init_max, dt_init, iter_cnt_limit'

with open(grid_para_file, "w") as f:
    f.write( para_names + '\n' )
    for item in paras:
        if type(item) is list or type(item) is np.ndarray:
            for el in item:
                f.write( f'{el} ' )
        else: f.write( f'{item} ' )

grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids =\
    initialize_grid_and_particles(
        x_min, x_max, z_min, z_max, dx, dy, dz,
        p_0, p_ref, r_tot_0, Theta_l,
        n_p, no_spcm, dst, dst_par, 
        P_min, P_max, r0, r1, dr, rnd_seed, reseed,
        S_init_max, dt_init, Newton_iterations, iter_cnt_limit, path)


#%%

from file_handling import load_grid_and_particles_full
    
grid, pos, cells, vel, m_w, m_s, xi, active_ids, removed_ids = \
    load_grid_and_particles_full(0, path)

# mapping from temperature grid -> list of temperatures with dimension of m_w

#%%

cells = cells.astype(int)
print(cells)

#%%
print(grid.temperature[cells])


#%%
R_p, w_s, rho_p = compute_R_p_w_s_rho_p(m_w, m_s, )

#%%
    
def plot_pos_vel_pt(pos, vel, grid,
                    figsize=(8,8), no_ticks = [6,6],
                    MS = 1.0, ARRSCALE=2):
    u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
    v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
    ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
    # ax.quiver(*pos, *vel, scale=ARRSCALE, pivot="mid")
    # ax.quiver(*grid.centers, u_g[:,0:-1], v_g[0:-1],
              # scale=ARRSCALE, pivot="mid", color="red")
    # ax.quiver(grid.corners[0], grid.corners[1] + 0.5*grid.steps[1],
    #           grid.velocity[0], np.zeros_like(grid.velocity[0]),
    #           scale=0.5, pivot="mid", color="red")
    # ax.quiver(grid.corners[0] + 0.5*grid.steps[0], grid.corners[1],
    #           np.zeros_like(grid.velocity[1]), grid.velocity[1],
    #           scale=0.5, pivot="mid", color="blue")
    x_min = grid.ranges[0,0]
    x_max = grid.ranges[0,1]
    y_min = grid.ranges[1,0]
    y_max = grid.ranges[1,1]
    ax.set_xticks( np.linspace(x_min, x_max, no_ticks[0]) )
    ax.set_yticks( np.linspace(y_min, y_max, no_ticks[1]) )
    # ax.set_xticks(grid.corners[0][:,0])
    # ax.set_yticks(grid.corners[1][0,:])
    ax.set_xticks(grid.corners[0][:,0], minor = True)
    ax.set_yticks(grid.corners[1][0,:], minor = True)
    # plt.minorticks_off()
    # plt.minorticks_on()
    ax.grid()
    # ax.grid(which="minor")
    plt.show()
    
plot_pos_vel_pt(pos, vel, grid, no_ticks=[11,11], MS = 0.1, ARRSCALE=50)

# grid.plot_thermodynamic_scalar_fields_grid()
  

#%%
# particles: pos, vel, masses, multi,

# base grid without THD or flow values

# from grid import Grid
# from grid import interpolate_velocity_from_cell_bilinear_jit,\
#                  interpolate_velocity_from_position_bilinear_jit,\
#                  compute_cell_and_relative_position_jit
#                  # compute_cell_and_relative_position,\

# # domain size
# x_min = 0.0
# x_max = 120.0
# z_min = 0.0
# z_max = 200.0

# # grid steps
# dx = 20.0
# dy = 1.0
# dz = 20.0

# grid_ranges = [ [x_min, x_max], [z_min, z_max] ]
# grid_steps = [dx, dz]
# grid = Grid( grid_ranges, grid_steps, dy )

# # set mass flux, note that the velocity requires the dry density field first
# j_max = 0.6
# j_max = compute_j_max(j_max, grid)
# grid.mass_flux_air_dry = np.array(
#     compute_initial_mass_flux_air_dry_kinematic_2D_ICMW_2012_case1(grid, j_max))
# grid.velocity = grid.mass_flux_air_dry / c.mass_density_air_dry_NTP


# print("grid.velocity[0]")
# print(np.shape(grid.velocity))
# print(grid.velocity[0])

# # gen. positions of no_spc Sp per cell
# no_spc = 20
# seed = 4713
# pos, rel_pos = generate_random_positions(grid, no_spc, seed)
# cell_list = np.zeros_like(pos).astype(int)

# n = 0
# for j in range(grid.no_cells[1]):
#     for i in range(grid.no_cells[0]):
#         for sp_N in range(no_spc):
#             cell_list[0,n] = i
#             cell_list[1,n] = j
#             n += 1

# # cells = compute_cell(*pos,
# #                                            grid.steps[0],
# #                                            grid.steps[1])

# # i, j, x_rel, y_rel = compute_cell_and_relative_position_jit(*pos,
# cells, rel_pos = compute_cell_and_relative_position_jit(pos,
#                                                         grid.ranges,
#                                                         grid.steps)
#                                              # grid.ranges[0,0],
#                                              # grid.ranges[1,0],
#                                              # grid.steps[0],
#                                              # grid.steps[1])

# print()
# print("cells")
# print(cells)


# # print(i)
# # print(j)
# # print(x_rel)
# # print(y_rel)


# # cell_list = np.array((i, j))
# # rel_post = np.array((x_rel, y_rel))

# # vel = interpolate_velocity_from_position_bilinear_jit(
# #           grid.velocity, grid.no_cells, grid.ranges, grid.steps, pos)
# print()
# print(type(grid.velocity))
# print(type(grid.no_cells))
# print(type(cells))
# print(type(rel_pos))
# print()

# print("cells[0,1]")
# print(cells[0,1])


# vel3 = interpolate_velocity_from_cell_bilinear_jit(grid.velocity,
#                                                   grid.no_cells,
#                                                   cells, rel_pos)
# np.set_printoptions(precision=4)
# print()
# print("vel")
# print(vel)

# vel = interpolate_velocity_from_position_bilinear_jit(pos, grid.velocity,
#                                                        grid.no_cells,
#                                                        grid.ranges,
#                                                        grid.steps,
#                                                        )



# vel2 = np.zeros_like(vel)
# for n,x in enumerate(pos[0]):
#     u,v = grid.interpolate_velocity_from_location_bilinear(x, pos[1,n])
#     vel2[0,n] = u
#     vel2[1,n] = v

# print()
# print("vel2 - vel1")
# print(vel2 - vel)
# max_dev = 0.0
# max_n = 0
# for n,v in enumerate(vel[0]):
#     if abs(vel2[0,n] - vel[0,n]) > max_dev:
#         max_dev = vel2[0,n] - vel[0,n]
#         max_n = n
#     if abs( vel2[1,n] - vel[1,n]) > max_dev:
#         max_dev = vel2[0,n] - vel[0,n]
#         max_n = n
# print(max_n, max_dev)
# print()

# #%%

# # visualize particles in grid
# # %matplotlib inline
# def plot_pos_pt(pos, grid, figsize=(6,6), MS = 1.0):
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
#     ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
#     ax.set_xticks(grid.corners[0][:,0])
#     ax.set_yticks(grid.corners[1][0,:])
#     # ax.set_xticks(grid.corners[0][:,0], minor = True)
#     # ax.set_yticks(grid.corners[1][0,:], minor = True)
#     plt.minorticks_off()
#     # plt.minorticks_on()
#     ax.grid()
#     # ax.grid(which="minor")
#     plt.show()
# def plot_pos_vel_pt(pos, vel, grid, figsize=(8,8), MS = 1.0):
#     u_g = 0.5 * ( grid.velocity[0,0:-1] + grid.velocity[0,1:] )
#     v_g = 0.5 * ( grid.velocity[1,:,0:-1] + grid.velocity[1,:,1:] )
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.plot(grid.corners[0], grid.corners[1], "x", color="red", markersize=MS)
#     ax.plot(pos[0],pos[1], "o", color="k", markersize=2*MS)
#     ax.quiver(*pos, *vel, scale=2, pivot="mid")
#     ax.quiver(*grid.centers, u_g[:,0:-1], v_g[0:-1],
#               scale=2, pivot="mid", color="red")
#     # ax.quiver(grid.corners[0], grid.corners[1] + 0.5*grid.steps[1],
#     #           grid.velocity[0], np.zeros_like(grid.velocity[0]),
#     #           scale=0.5, pivot="mid", color="red")
#     # ax.quiver(grid.corners[0] + 0.5*grid.steps[0], grid.corners[1],
#     #           np.zeros_like(grid.velocity[1]), grid.velocity[1],
#     #           scale=0.5, pivot="mid", color="blue")
#     ax.set_xticks(grid.corners[0][:,0])
#     ax.set_yticks(grid.corners[1][0,:])
#     # ax.set_xticks(grid.corners[0][:,0], minor = True)
#     # ax.set_yticks(grid.corners[1][0,:], minor = True)
#     plt.minorticks_off()
#     # plt.minorticks_on()
#     ax.grid()
#     # ax.grid(which="minor")
#     plt.show()
    
# plot_pos_vel_pt(pos, vel, grid)

# #%%



# print(grid.corners[0][:,0])

# print("radii distribution")

# p_min = [0.001] * 2
# p_max = [0.999] * 2
# no_spcm = [4, 3]

# dst = [dst_log_normal, dst_log_normal]
# no_modes = 2
# mus = [0.02, 0.075]
# sigmas = [1.4, 1.6]
# pars = []
# for i,mu in enumerate(mus):
#     pars.append( [mu, math.log(sigmas[i])] )
# print("pars = ", pars)
# dr = [1E-4] * 2
# r0 = dr
# r1 = [ pars[0][1]*10, pars[1][1]*10 ]
# seed = [4711] * 2

# rad1,weights1 = generate_random_radii_monomodal(grid, p_min[0], p_max[0],
#                                                 no_spcm[0],
#                                 dst[0], pars[0], r0[0], r1[0], dr[0], seed[0])

# rad2,weights2 = generate_random_radii_monomodal(grid, p_min[1], p_max[1],
#                                                 no_spcm[1],
#                                 dst[1], pars[1], r0[1], r1[1], dr[0], seed[1])

# rad_merged = np.concatenate( (rad1, rad2),axis=2 )

# np.set_printoptions(precision=5)

# print(np.shape(rad1))
# print(np.shape(rad2))
# print(np.shape(rad_merged))
# print()
# # for i in range(grid.no_cells[0]):
# #     for j in range(grid.no_cells[1]):
# #         print(rad1[i,j])
# # print()
# # for i in range(grid.no_cells[0]):
# #     for j in range(grid.no_cells[1]):
# #         print(rad2[i,j])
# # print()
# # for i in range(grid.no_cells[0]):
# #     for j in range(grid.no_cells[1]):
# #         print(rad_merged[i,j])
# # print()

# p_min = 0.001
# p_max = 0.999

# no_modes = 2
# no_spcm = [4, 3]
# dst = dst_log_normal
# mus = [0.02, 0.075]
# sigmas = [1.4, 1.6]
# dist_pars = []
# for i,mu in enumerate(mus):
#     dist_pars.append( [mu, math.log(sigmas[i])] )
# print("pars = ", pars)
# dr = 1E-4
# r0 = dr
# r1 = [ 10 * p_[1] for p_ in pars ]
# print(r1)
# seed = 4711

# rad, weight = generate_random_radii_multimodal(grid, p_min, p_max, no_spcm,
#                                      dst, dist_pars, r0, r1, dr, seed)

# print(np.shape(rad))
# print(np.shape(weight))

# for i in range(grid.no_cells[0]):
#     for j in range(grid.no_cells[1]):
#         print(rad_merged[i,j])
# print()
# for i in range(grid.no_cells[0]):
#     for j in range(grid.no_cells[1]):
#         print(rad[i,j])
# print()
# for i in range(grid.no_cells[0]):
#     for j in range(grid.no_cells[1]):
#         print(rad[i,j] - rad_merged[i,j])

# # generate_random_radii_multimodal(grid, p_min, p_max, no_spcm,
# #                                      dst, pars, r0, r1, dr, seed, no_modes)
    
#     # for var in [p_min, p_max, no_spcm, dst, par, r0, r1, dr, seed]:
#     #     if not isinstance(var, (list, tuple, np.ndarray)):
#     #         var = [var] * no_modes
    
#     # print([p_min, p_max, no_spcm, dst, par, r0, r1, dr, seed])
# # for i in range(grid.no_cells[0]):
# #     for j in range(grid.no_cells[1]):
# #         print(weights1[i,j], np.sum(weights1[i,j]))


# # plot_pos_pt(pos[:,0:8*8], grid, MS = 2.0)
    
# #%%
# # print(1, np.random.get_state()[1][0])
# # print()

# # for seeds in np.arange(1,1000000,123456):
# #     np.random.seed(seeds)
# #     print(seeds, np.random.get_state()[1][0])


# # def test1(seed):
# #     print(np.random.get_state()[1][0])
# #     np.random.seed(seed)
# #     print(np.random.get_state()[1][0])
# # def test2(seed,reseed):
# #     print(np.random.get_state()[1][0])
# #     if reseed:
# #         test1(seed)
# #     print(np.random.get_state()[1][0])

# # print(np.random.get_state()[1][0])
# # print()
# # test1(4713)
# # print()
# # print(np.random.get_state()[1][0])
# # print()
# # test2(4715, True)
# # print()
# # print(np.random.get_state()[1][0])
