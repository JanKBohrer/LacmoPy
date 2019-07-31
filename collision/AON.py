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
#import kernel
from collision.kernel import compute_kernel_Long_Bott_m, compute_kernel_hydro, \
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
        R_kernel_low_log, bin_factor_R_log, no_cols ):
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

# given E_col grid for radii
# given velocities "vel"
# updates masses AND radii
# assume mass densities constant (but varying for diff SIPs) during coll step
def collision_step_Long_Bott_Ecol_grid_R_2D_np(
        xis, masses, radii, vel, mass_densities,
        dt_over_dV, E_col_grid, no_kernel_bins,
        R_kernel_low_log, bin_factor_R_log, no_cols ):
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
                           # abs difference of 2D vectors
                           math.sqrt( (vel[0,i] - vel[0,j])**2
                                      + (vel[1,i] - vel[1,j])**2 ) )

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
collision_step_Long_Bott_Ecol_grid_R_2D = \
    njit()(collision_step_Long_Bott_Ecol_grid_R_2D_np)