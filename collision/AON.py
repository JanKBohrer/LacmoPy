#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:25:13 2019

@author: jdesk
"""

#%% IMPORTS
import math
import numpy as np
from numba import njit
import kernel
from kernel import compute_kernel_Long_Bott_m, compute_kernel_hydro, \
                   compute_E_col_Long_Bott
from microphysics import compute_radius_from_mass_jit
from microphysics import compute_radius_from_mass_vec

#%% COLLISION ALGORITHMS

# one collision step for a number of SIPs in dV during dt
# the multiplicities are collected in "xis" (rational, non-integer)
# the SIP-masses are collected in "masses" (in 1E-18 kg)
# the collection pair (i-j) order is the same as in Unterstrasser
def collision_step_Long_Bott_m_np(xis, masses, mass_density, dt_over_dV,
                                  no_cols):
    # check each i-j combination for a possible collection event
    no_SIPs = xis.shape[0]
    # ind = generate_permutation(no_SIPs)

    rnd = np.random.rand( (no_SIPs*(no_SIPs-1))//2 )
    # print(rnd[0])
    cnt = 0
    for i in range(0,no_SIPs-1):
#        ind_kernel_i = ind_kernel[i]
        for j in range(i+1, no_SIPs):
            if xis[i] <= xis[j]:
                ind_min = i
                ind_max = j
            else:
                ind_min = j
                ind_max = i
            xi_min = xis[ind_min] # = nu_i in Unt
            xi_max = xis[ind_max] # = nu_j in Unt
            m_min = masses[ind_min] # = mu_i in Unt
            m_max = masses[ind_max] # = mu_j in Unt

            p_crit = xi_max \
                     * compute_kernel_Long_Bott_m(m_min, m_max, mass_density) \
                     * dt_over_dV

            if p_crit > 1.0:
                # multiple collection
                no_cols[1] += 1
                xi_col = p_crit * xi_min
                masses[ind_min] = (xi_min*m_min + xi_col*m_max) / xi_min
                xis[ind_max] -= xi_col
                # print(j,i,p_crit)
            elif p_crit > rnd[cnt]:
                no_cols[0] += 1
                xi_rel_dev = (xi_max-xi_min)/xi_max
                # if xis are equal (or nearly equal)
                # the two droplets are diveded in two droplets of the same mass
                # with xi1 = 0.7 * 0.5 * (xi1 + xi2)
                # xi2 = 0.3 * 0.5 * (xi1 + xi2)
                # the separation in weights of 0.7 and 0.3 
                # is to avoid the same situation for the two droplets during 
                # the next two collisions
                if xi_rel_dev < 1.0E-5:
                    print("xi_i approx xi_j, xi_rel_dev =",
                          xi_rel_dev,
                          " in collision")
                    xi_ges = xi_min + xi_max
                    masses[ind_min] = 2.0 * ( xi_min*m_min + xi_max*m_max ) \
                                      / xi_ges
                    masses[ind_max] = masses[ind_min]
                    xis[ind_max] = 0.5 * 0.7 * xi_ges
                    xis[ind_min] = 0.5 * xi_ges - xis[ind_max]
                else:
                    masses[ind_min] += m_max
                    xis[ind_max] -= xi_min
                # print(j,i,p_crit,rnd[cnt])
            cnt += 1
collision_step_Long_Bott_m = njit()(collision_step_Long_Bott_m_np)

# x is e.g. mass or radius
# depending on which log-grid is used
# x_kernel_low_log is the log of the minimum of the discrete kernel grid-range
@njit()
def compute_kernel_index(x, x_kernel_low_log,
                         bin_factor_x_log, no_kernel_bins):
    ind = int( ( math.log(x) - x_kernel_low_log ) / bin_factor_x_log + 0.5 )
    if ind < 0 : ind = 0
    elif ind > no_kernel_bins - 1: ind = no_kernel_bins - 1
    return ind

# x is an array of masses or radii
@njit()
def create_kernel_index_array(x, no_SIPs, x_kernel_low_log,
                              bin_factor_x_log, no_kernel_bins):
    ind_kernel = np.zeros( no_SIPs, dtype = np.int64 )
    for i in range(no_SIPs):
        ind_kernel[i] =\
            compute_kernel_index(x[i], x_kernel_low_log,
                                 bin_factor_x_log, no_kernel_bins)
    return ind_kernel



# given the kernel grid for masses
def collision_step_Long_Bott_kernel_grid_m_np(
        xis, masses, dt_over_dV,
        kernel_grid, no_kernel_bins,
        m_kernel_low_log, bin_factor_m_log, no_cols):
    
    no_SIPs = xis.shape[0]

    rnd = np.random.rand( (no_SIPs*(no_SIPs-1))//2 )
    
    ind_kernel = create_kernel_index_array(
                     masses, no_SIPs,
                     m_kernel_low_log, bin_factor_m_log, no_kernel_bins)
    
    # check each i-j combination for a possible collection event
    cnt = 0
    for i in range(0,no_SIPs-1):
        for j in range(i+1, no_SIPs):
            if xis[i] <= xis[j]:
                ind_min = i
                ind_max = j
            else:
                ind_min = j
                ind_max = i
            xi_min = xis[ind_min] # = nu_i in Unt
            xi_max = xis[ind_max] # = nu_j in Unt
            m_min = masses[ind_min] # = mu_i in Unt
            m_max = masses[ind_max] # = mu_j in Unt

            p_crit = xi_max \
                     * kernel_grid[ind_kernel[i], ind_kernel[j]] * dt_over_dV

            if p_crit > 1.0:
                # multiple collection
                xi_col = p_crit * xi_min
                masses[ind_min] = (xi_min*m_min + xi_col*m_max) / xi_min
                xis[ind_max] -= xi_col
                # mass of ind_min changed -> update kernel index:
                ind_kernel[ind_min] = \
                    compute_kernel_index(masses[ind_min], m_kernel_low_log,
                                         bin_factor_m_log, no_kernel_bins)
                no_cols[1] += 1
            elif p_crit > rnd[cnt]:
                no_cols[0] += 1
                xi_rel_dev = (xi_max-xi_min)/xi_max
                if xi_rel_dev < 1.0E-5:
                    print("xi_i approx xi_j, xi_rel_dev =",
                          xi_rel_dev,
                          " in collision")
                    xi_ges = xi_min + xi_max
                    masses[ind_min] = 2.0 * ( xi_min*m_min + xi_max*m_max ) \
                                      / xi_ges
                    masses[ind_max] = masses[ind_min]
                    xis[ind_max] = 0.5 * 0.7 * xi_ges
                    xis[ind_min] = 0.5 * xi_ges - xis[ind_max]
                    # mass of ind_min AND ind_max changed to same masses
                    # -> update kernel ind:
                    ind_kernel[ind_min] = \
                        compute_kernel_index(masses[ind_min], m_kernel_low_log,
                                             bin_factor_m_log, no_kernel_bins)
                    ind_kernel[ind_max] = ind_kernel[ind_min]
                else:
                    masses[ind_min] += m_max
                    xis[ind_max] -= xi_min
                    # mass of ind_min changed -> update kernel index:
                    ind_kernel[ind_min] = \
                        compute_kernel_index(masses[ind_min], m_kernel_low_log,
                                             bin_factor_m_log, no_kernel_bins)
            cnt += 1
collision_step_Long_Bott_kernel_grid_m = \
    njit()(collision_step_Long_Bott_kernel_grid_m_np)

# given E_col grid for radii
# given velocities "vel"
# updates masses AND radii
# assume mass densities constant (but varying for diff SIPs) during coll step
def collision_step_Long_Bott_Ecol_grid_R_np(
        xis, masses, radii, vel, mass_densities,
        dt_over_dV, E_col_grid, no_kernel_bins,
        R_kernel_low_log, bin_factor_R_log, no_cols):
    no_SIPs = xis.shape[0]

    rnd = np.random.rand( (no_SIPs*(no_SIPs-1))//2 )
    
    ind_kernel = create_kernel_index_array(
                     radii, no_SIPs,
                     R_kernel_low_log, bin_factor_R_log, no_kernel_bins)
    
    # check each i-j combination for a possible collection event
    cnt = 0
    for i in range(0, no_SIPs-1):
        for j in range(i+1, no_SIPs):
            if xis[i] <= xis[j]:
                ind_min = i
                ind_max = j
            else:
                ind_min = j
                ind_max = i
            xi_min = xis[ind_min] # = nu_i in Unt
            xi_max = xis[ind_max] # = nu_j in Unt
            m_min = masses[ind_min] # = mu_i in Unt, not necc. the smaller mass
            m_max = masses[ind_max] # = mu_j in Unt, not necc. the larger mass

            p_crit = xi_max * dt_over_dV \
                     * compute_kernel_hydro(
                           radii[i], radii[j],
                           E_col_grid[ind_kernel[i], ind_kernel[j]],
                           abs(vel[i]-vel[j]))

            if p_crit > 1.0:
                # multiple collection
                xi_col = p_crit * xi_min
                masses[ind_min] = (xi_min*m_min + xi_col*m_max) / xi_min
                # mass changed: update radius
                radii[ind_min] =\
                    compute_radius_from_mass_jit(masses[ind_min],
                                                 mass_densities[ind_min])
                xis[ind_max] -= xi_col
                # rad of ind_min changed -> update kernel index:
                ind_kernel[ind_min] = \
                    compute_kernel_index(radii[ind_min], R_kernel_low_log,
                                         bin_factor_R_log, no_kernel_bins)
                no_cols[1] += 1
            elif p_crit > rnd[cnt]:
                no_cols[0] += 1
                xi_rel_dev = (xi_max-xi_min)/xi_max
                if xi_rel_dev < 1.0E-5:
                    print("xi_i approx xi_j, xi_rel_dev =",
                          xi_rel_dev,
                          " in collision")
                    xi_ges = xi_min + xi_max
                    masses[ind_min] = 2.0 * ( xi_min*m_min + xi_max*m_max ) \
                                      / xi_ges
                    masses[ind_max] = masses[ind_min]
                    radii[ind_min] =\
                        compute_radius_from_mass_jit(masses[ind_min],
                                                     mass_densities[ind_min])
                    radii[ind_max] = radii[ind_min]
                    xis[ind_max] = 0.5 * 0.7 * xi_ges
                    xis[ind_min] = 0.5 * xi_ges - xis[ind_max]
                    # radius of ind_min AND ind_max changed to same radii
                    # -> update kernel ind:
                    ind_kernel[ind_min] = \
                        compute_kernel_index(radii[ind_min], R_kernel_low_log,
                                             bin_factor_R_log, no_kernel_bins)
                    ind_kernel[ind_max] = ind_kernel[ind_min]
                else:
                    masses[ind_min] += m_max
                    radii[ind_min] =\
                        compute_radius_from_mass_jit(masses[ind_min],
                                                     mass_densities[ind_min])
                    xis[ind_max] -= xi_min
                    # rad of ind_min changed -> update kernel index:
                    ind_kernel[ind_min] = \
                        compute_kernel_index(radii[ind_min], R_kernel_low_log,
                                             bin_factor_R_log, no_kernel_bins)
            cnt += 1
collision_step_Long_Bott_Ecol_grid_R = \
    njit()(collision_step_Long_Bott_Ecol_grid_R_np)




#%% METHOD TEST

#OS = "LinuxDesk"
## OS = "MacOS"
#
#kernel_method = "kernel_grid_m"
##kernel_method = "Ecol_grid"
##kernel_method = "analytic"
#
##args = [1,0,0,0]
#args = [0,1,1,1]
## args = [1,1,1,1]
#
## simulate = True
## analyze = True
## plotting = True
## plot_moments_kappa_var = True
#
#simulate = bool(args[0])
#analyze = bool(args[1])
#plotting = bool(args[2])
#plot_moments_kappa_var = bool(args[3])
#
#dist = "expo"
#
#kappa = 20
#
## kappa = 800
#
##kappa_list=[400]
##kappa_list=[5]
##kappa_list=[5,10,20,40]
## kappa_list=[5,10,20,40,60,100,200]
## kappa_list=[5,10,20,40,60,100,200,400]
#kappa_list=[5,10,20,40,60,100,200,400]
## kappa_list=[5,10,20,40,60,100,200,400,600,800]
## kappa_list=[600]
## kappa_list=[800]
#
#eta = 1.0E-9
#
## no_sims = 163
## no_sims = 450
#no_sims = 500
## no_sims = 50
## no_sims = 250
#
## no_sims = 94
#
#start_seed = 3711
## start_seed = 4523
## start_seed = 4127
## start_seed = 4107
## start_seed = 4385
## start_seed = 3811
#
#seed = start_seed
#
#no_bins = 50
#
#gen_method = "SinSIP"
## kernel = "Golovin"
#kernel_name = "Long_Bott"
#
#bin_method = "auto_bin"
#
## dt = 1.0
#dt = 10.0
##dt = 20.0
##dt = 50.0
#dV = 1.0
## dt_save = 40.0
#dt_save = 300.0
## t_end = 200.0
#t_end = 3600.0
#
#mass_density = 1.0E3
#
#
#if OS == "MacOS":
#    sim_data_path = "/Users/bohrer/sim_data/"
#elif OS == "LinuxDesk":
#    sim_data_path = "/mnt/D/sim_data_my_kernel_grid_strict_thresh/"
##    sim_data_path = "/mnt/D/sim_data/"
#    
#ensemble_dir =\
#    sim_data_path + \
#    f"col_box_mod/ensembles/{dist}/{gen_method}/eta_{eta:.0e}/kappa_{kappa}/"
#
#masses = np.load(ensemble_dir + f"masses_seed_{seed}.npy")
#xis = np.load(ensemble_dir + f"xis_seed_{seed}.npy")
#
#masses*=1E18
#
#radii = compute_radius_from_mass_vec(masses, mass_density)
#mass_densities = np.ones_like(radii) * mass_density
## for testing
#vel = np.zeros_like(radii)
#for i in range (len(vel)):
#    vel[i] = kernel.compute_terminal_velocity_Beard(radii[i])
#
#m0 = np.copy(masses)
#xi0 = np.copy(xis)
#
#dt_over_dV = dt/dV
#
#np.random.seed(seed)
#
#no_cols = np.array( (0,0) )

#%%

def generate_E_col_grid_Long_Bott_np(R_low, R_high, no_bins_10,
                                     mass_density):

    bin_factor = 10**(1.0/no_bins_10)
    no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
    radius_grid = np.zeros( no_bins, dtype = np.float64 )
    
    radius_grid[0] = R_low
    for bin_n in range(1,no_bins):
        radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor
    
    # vel grid ONLY for testing...
#    vel_grid = np.zeros( no_bins, dtype = np.float64 )
#    for i in range(no_bins):
#        vel_grid[i] = kernel.compute_terminal_velocity_Beard(radius_grid[i])
    
    E_col_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    for j in range(0,no_bins):
        R_j = radius_grid[j]
        for i in range(j+1):
            E_col_grid[j,i] = compute_E_col_Long_Bott(radius_grid[i], R_j)
            E_col_grid[i,j] = E_col_grid[j,i]
    return E_col_grid, radius_grid
#    return E_col_grid, radius_grid, vel_grid
generate_E_col_grid_Long_Bott = \
    njit()(generate_E_col_grid_Long_Bott_np)

#R_low = 0.6
#R_high = 6E3
#no_bins_10 = 20
#
#E_col_grid, radius_grid = \
#    generate_E_col_grid_Long_Bott(R_low, R_high, no_bins_10, mass_density)
#
#no_kernel_bins = len(radius_grid)
#R_kernel_low = radius_grid[0]
#bin_factor_R = radius_grid[1] / radius_grid[0]
#R_kernel_low_log = math.log(R_kernel_low)
#bin_factor_R_log = math.log(bin_factor_R)
#
#no_steps = 360
#for step_n in range(no_steps):
#    collision_step_Long_Bott_Ecol_grid_R(
#        xis, masses, radii, vel, mass_densities,
#        dt_over_dV, E_col_grid, no_kernel_bins,
#        R_kernel_low_log, bin_factor_R_log, no_cols)
#    print(step_n, no_cols)
#
#print(masses-m0)
#print(xis-xi0)

#%% TESTING collision steps

#for i in range(4):
#    collision_step_Long_Bott_m(xis, masses, mass_density, dt_over_dV, no_cols)
##collision_step_Long_Bott_m(xis, masses, mass_density, dt_over_dV)
#
#print("kernel exact")
#print(no_cols)
#print(masses-m0)
#print(xis-xi0)
#
#kernel_method = "grid"
#if kernel_method == "grid":
#    mass_grid = np.load( sim_data_path + f"col_box_mod/results/{dist}/{kernel}/kernel_data/mass_grid_out.npy" )
#    mass_grid *= 1E18
#    kernel_grid = np.load( sim_data_path + f"col_box_mod/results/{dist}/{kernel}/kernel_data/kernel_grid.npy" )
#    no_kernel_bins = len(mass_grid)
#    m_kernel_low = mass_grid[0]
#    bin_factor_m = mass_grid[1] / mass_grid[0]
#    m_kernel_low_log = math.log(m_kernel_low)
#    bin_factor_m_log = math.log(bin_factor_m)
#
#masses = np.copy(m0)
#xis = np.copy(xi0)
#
#no_cols = np.array( (0,0) )
#
#for i in range(4):
#    collision_step_Long_Bott_kernel_grid_m(
#        xis, masses, dt_over_dV, kernel_grid, no_kernel_bins,
#        m_kernel_low_log, bin_factor_m_log, no_cols)
#
#print()
#print("kernel grid")
#print(no_cols)
#print(masses-m0)
#print(xis-xi0)
#
#print("done AON")