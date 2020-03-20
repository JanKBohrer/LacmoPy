#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinetic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

ALL-OR-NOTHING COLLISION ALGORITHM

the all-or-nothing collision algorithm is motivated by 
Shima et al. 2009, Q. J. R. Meteorol. Soc. 135: 1307–1320 and adapted from
Unterstrasser 2017, GMD 10: 1521–1548

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
import math
import numpy as np
from numba import njit

import collision.kernel as ker
import microphysics as mp

#%% INDEXING OF PARTICLES ON THE KERNEL GRID
# x is e.g. mass or radius
# depending on which log-grid is used
# x_kernel_low_log is the log of the minimum of the discrete kernel grid-range
@njit()
def compute_kernel_index(x, x_kernel_low_log, bin_factor_x_log,
                         no_kernel_bins):
    """Compute index of one mass or radius by binning in a 1D grid
    
    A logarithmically discretized grid is assumed, where 
    x_(n+1) = x_n * bin_factor_x => x_n = x_kernel_low * bin_factor_x^n.
    The grid is restricted by a minimum 'x_kernel_low' and a maximum given by
    the number of bins 'no_kernel_bins'.
    
    Parameters
    ----------
    x: float
        Mass or radius to be binned in the 1D grid
    x_kernel_low_log: float
        Nat. log of x_kernel_low (s.a.)
    bin_factor_x_log: float
        Nat. log of bin_factor_x (s.a.)
    no_kernel_bins: int
        number of kernel bins
    
    Returns
    -------
    ind: int
        index of the grid bin, in which x is distributed
    
    """
            
    ind = int( ( math.log(x) - x_kernel_low_log ) / bin_factor_x_log + 0.5 )
    if ind < 0 : ind = 0
    elif ind > no_kernel_bins - 1: ind = no_kernel_bins - 1
    return ind

# x is an array of masses or radii
# x_kernel_low_log is the log of the minimum of the discrete kernel grid-range
@njit()
def create_kernel_index_array(x, no_SIPs, x_kernel_low_log,
                              bin_factor_x_log, no_kernel_bins):
    """Compute indices of a mass or radius array by binning in a 1D grid
    
    A logarithmically discretized grid is assumed, where 
    x_(n+1) = x_n * bin_factor_x => x_n = x_kernel_low * bin_factor_x^n.
    The grid is restricted by a minimum 'x_kernel_low' and a maximum given by
    the number of bins 'no_kernel_bins'.
    
    Parameters
    ----------
    x: ndarray, dtype=float
        1D Mass or radius array to be binned in the 1D grid
    x_kernel_low_log: float
        Nat. log of x_kernel_low (s.a.)
    bin_factor_x_log: float
        Nat. log of bin_factor_x (s.a.)
    no_kernel_bins: int
        number of kernel bins
    
    Returns
    -------
    ind_kernel: ndarray, dtype=int
        array with indices of the grid bins, in which the values of x
        are distributed
    
    """
    
    ind_kernel = np.zeros( no_SIPs, dtype = np.int64 )
    for i in range(no_SIPs):
        ind_kernel[i] =\
            compute_kernel_index(x[i], x_kernel_low_log,
                                 bin_factor_x_log, no_kernel_bins)
    return ind_kernel

#%% BOX MODEL COLLISION ALGORITHMS
# only one mass-material for particles (e.g. water or solution),
# no solute/water differentiation    

def collision_step_Golovin_np(xis, masses, dt_over_dV, no_cols):
    """One collision step for a number of SIPs in dV, Golovin kernel, analytic
    
    Analytic computation of the kernel function for each collision pair.
    The collection pair (i-j) order is the same as in Unterstrasser 2017.
    Updates xis, masses, no_cols.
    
    Parameters
    ----------
    xis: ndarray, dtype=float
        1D array of SIP multiplicities (real numbers, non-integer)
    masses: ndarray, dtype=float
        1D array of SIP masses (unit = 1E-18 kg)
    dt_over_dV: float
        dt/dV, where: dt = collision time step,
        dV = volume, in which particles are well mixed and collide
    no_cols: ndarray, shape=(2,), type=int
        counts the collisions
        no_cols[0] = number of ordinary collision,
        no_cols[1] = number of multiple collision event
    
    """
    
    # check each i-j combination for a possible collection event
    no_SIPs = xis.shape[0]
    rnd = np.random.rand( (no_SIPs*(no_SIPs-1))//2 )
    cnt = 0
    
    for i in range(0,no_SIPs-1):
        for j in range(i+1, no_SIPs):
            if xis[i] <= xis[j]:
                ind_min = i
                ind_max = j
            else:
                ind_min = j
                ind_max = i
            xi_min = xis[ind_min] # = nu_i in Unterstrasser 2015
            xi_max = xis[ind_max] # = nu_j in Unt
            m_min = masses[ind_min] # = mu_i in Unt
            m_max = masses[ind_max] # = mu_j in Unt

            p_crit = xi_max \
                     * ker.compute_kernel_Golovin(m_min, m_max) \
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
            cnt += 1
collision_step_Golovin = njit()(collision_step_Golovin_np)

def collision_step_Long_Bott_m_np(xis, masses, mass_density, dt_over_dV,
                                  no_cols):
    """One collision step for a number of SIPs in dV, 'Long' kernel, analytic
    
    Analytic computation of the kernel function for each collision pair.
    Function is currently not in use.
    The collection pair (i-j) order is the same as in Unterstrasser 2017
    Updates xis, masses and no_cols.
    
    Parameters
    ----------
    xis: ndarray, dtype=float
        1D array of SIP multiplicities (real numbers, non-integer)
    masses: ndarray, dtype=float
        1D array of SIP masses (unit = 1E-18 kg)
    mass_density: float
        mass density the SIPs (kg/m^3). All SIP have the same density.
    dt_over_dV: float
        dt/dV, where: dt = collision time step,
        dV = volume, in which particles are well mixed and collide
    no_cols: ndarray, shape=(2,), type=int
        counts the collisions
        no_cols[0] = number of ordinary collisions,
        no_cols[1] = number of multiple collision events
    
    """    
    
    # check each i-j combination for a possible collection event
    no_SIPs = xis.shape[0]

    rnd = np.random.rand( (no_SIPs*(no_SIPs-1))//2 )
    cnt = 0
    for i in range(0,no_SIPs-1):
        for j in range(i+1, no_SIPs):
            if xis[i] <= xis[j]:
                ind_min = i
                ind_max = j
            else:
                ind_min = j
                ind_max = i
            xi_min = xis[ind_min] # = nu_i in Unterstrasser 2015
            xi_max = xis[ind_max] # = nu_j in Unt
            m_min = masses[ind_min] # = mu_i in Unt
            m_max = masses[ind_max] # = mu_j in Unt

            p_crit = xi_max \
                     * ker.compute_kernel_Long_Bott_m(m_min, m_max,
                                                      mass_density) \
                     * dt_over_dV

            if p_crit > 1.0:
                # multiple collection
                no_cols[1] += 1
                xi_col = p_crit * xi_min
                masses[ind_min] = (xi_min*m_min + xi_col*m_max) / xi_min
                xis[ind_max] -= xi_col
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
            cnt += 1
collision_step_Long_Bott_m = njit()(collision_step_Long_Bott_m_np)

def collision_step_Ecol_grid_R_np(
        xis, masses, radii, vel, mass_densities,
        dt_over_dV, E_col_grid, no_kernel_bins,
        R_kernel_low_log, bin_factor_R_log, no_cols):
    """One collision step for a number of SIPs in dV, discretized E_col(R1, R2)
    
    Uses a discretized collection efficiency
    based on a logarithmic radius grid.
    The collection pair (i-j) order is the same as in Unterstrasser 2017.
    Updates xis, masses, radii and no_cols.
    
    Parameters
    ----------
    xis: ndarray, dtype=float
        1D array of SIP multiplicities (real numbers, non-integer)
    masses: ndarray, dtype=float
        1D array of SIP masses (unit = 1E-18 kg)
    radii: ndarray, dtype=float
        1D array of SIP radii (unit = microns)
    vel: ndarray, dtype=float
        1D array of SIP velocites (unit = m/s)
        is kept stationary for one coll. step, but may vary from SIP to SIP
    mass_densities: ndarray, dtype=float
        mass densities of the SIPs (kg/m^3).
        is kept stationary for one coll. step, but may vary from SIP to SIP
    dt_over_dV: float
        dt/dV, where: dt = collision time step,
        dV = volume, in which particles are well mixed and collide
    E_col_grid: ndarray, shape=(no_kernel_bins,no_kernel_bins), type=float
        Discretized collection efficiency E_col(R1,R2) based on log. rad. grid
    no_kernel_bins: int
        number of bins used to discretize the collection efficiencies
    R_kernel_low_log: float
        nat. log of the lower radius boundary of the kernel discretization
    bin_factor_R_log: float
        nat. log of the radius bin factor => R_(n+1) = R_n * bin_factor_R_log
    no_cols: ndarray, shape=(2,), type=int
        counts the collisions
        no_cols[0] = number of ordinary collision,
        no_cols[1] = number of multiple collision event
    
    """    
    
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
                     * ker.compute_kernel_hydro(
                               radii[i], radii[j],
                               E_col_grid[ind_kernel[i], ind_kernel[j]],
                               abs(vel[i]-vel[j]))

            if p_crit > 1.0:
                # multiple collection
                xi_col = p_crit * xi_min
                masses[ind_min] = (xi_min*m_min + xi_col*m_max) / xi_min
                # mass changed: update radius
                radii[ind_min] =\
                    mp.compute_radius_from_mass(masses[ind_min],
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
                        mp.compute_radius_from_mass(masses[ind_min],
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
                        mp.compute_radius_from_mass(masses[ind_min],
                                                     mass_densities[ind_min])
                    xis[ind_max] -= xi_min
                    # rad of ind_min changed -> update kernel index:
                    ind_kernel[ind_min] = \
                        compute_kernel_index(radii[ind_min], R_kernel_low_log,
                                             bin_factor_R_log, no_kernel_bins)
            cnt += 1
collision_step_Ecol_grid_R = \
    njit()(collision_step_Ecol_grid_R_np)

def collision_step_kernel_grid_m_np(xis, masses, dt_over_dV,
                                    kernel_grid, no_kernel_bins,
                                    m_kernel_low_log, bin_factor_m_log,
                                    no_cols):
    """One collision step for a number of SIPs in dV, discretized K(m1, m2)

    Uses a discretized collection Kernel
    based on a logarithmic mass grid.
    The collection pair (i-j) order is the same as in Unterstrasser 2017.
    Updates xis, masses, radii and no_cols.
    
    Parameters
    ----------
    xis: ndarray, dtype=float
        1D array of SIP multiplicities (real numbers, non-integer)
    masses: ndarray, dtype=float
        1D array of SIP masses (unit = 1E-18 kg)
    dt_over_dV: float
        dt/dV, where: dt = collision time step,
        dV = volume, in which particles are well mixed and collide
    kernel_grid: ndarray, shape=(no_kernel_bins,no_kernel_bins), type=float
        Discretized coll. kernel K(m1,m2) based on log. mass grid
    no_kernel_bins: int
        number of bins used to discretize the collection efficiencies
    m_kernel_low_log: float
        nat. log of the lower mass boundary of the kernel discretization
    bin_factor_m_log: float
        nat. log of the mass bin factor => m_(n+1) = m_n * bin_factor_m_log
    no_cols: ndarray, shape=(2,), type=int
        counts the collisions
        no_cols[0] = number of ordinary collision,
        no_cols[1] = number of multiple collision event
    
    """  

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
            xi_min = xis[ind_min] # = nu_i in Unterstrasser 2015
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
collision_step_kernel_grid_m = \
    njit()(collision_step_kernel_grid_m_np)


#%% COLLISION ALGORITHMS FOR THE TWO DIMENSIONAL TEST CASE
# particles have water mass and one type of solute mass, which
# can either be NaCl or ammonium sulfate for all particles in the domain

def collision_step_Ecol_grid_R_2D_multicomp_np(
        xis, m_w, m_s, radii, vel, mass_densities,
        dt_over_dV, E_col_grid, no_kernel_bins,
        R_kernel_low_log, bin_factor_R_log, no_cols):
    """One collision step for a number of SIPs in dV, discretized E_col(R1, R2)
    
    Uses a discretized collection efficiency
    based on a logarithmic radius grid.
    The collection pair (i-j) order is the same as in Unterstrasser 2017.
    Updates xis, m_w, m_s, radii and no_cols.
    
    Parameters
    ----------
    xis: ndarray, dtype=float
        1D array of SIP multiplicities (real numbers, non-integer)
    m_w: ndarray, dtype=float
        1D array of SIP water-masses (unit = 1E-18 kg)
    m_s: ndarray, dtype=float
        1D array of SIP solute-masses, only one solute type (unit = 1E-18 kg)
    radii: ndarray, dtype=float
        1D array of SIP radii (unit = microns)
    vel: ndarray, dtype=float
        1D array of SIP velocites (unit = m/s)
        is kept stationary for one coll. step, but may vary from SIP to SIP
    mass_densities: ndarray, dtype=float
        mass densities of the SIPs (kg/m^3).
        is kept stationary for one coll. step, but may vary from SIP to SIP
    dt_over_dV: float
        dt/dV, where: dt = collision time step,
        dV = volume, in which particles are well mixed and collide
    E_col_grid: ndarray, shape=(no_kernel_bins,no_kernel_bins), type=float
        Discretized collection efficiency E_col(R1,R2) based on log. rad. grid
    no_kernel_bins: int
        number of bins used to discretize the collection efficiencies
    R_kernel_low_log: float
        nat. log of the lower radius boundary of the kernel discretization
    bin_factor_R_log: float
        nat. log of the radius bin factor => R_(n+1) = R_n * bin_factor_R_log
    no_cols: ndarray, shape=(2,), type=int
        counts the collisions
        no_cols[0] = number of ordinary collision,
        no_cols[1] = number of multiple collision event
    
    """    
    
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
            m_w_min = m_w[ind_min] # = mu_i in Unt, not necc. the smaller mass
            m_w_max = m_w[ind_max] # = mu_j in Unt, not necc. the larger mass
            m_s_min = m_s[ind_min] # = mu_i in Unt, not necc. the smaller mass
            m_s_max = m_s[ind_max] # = mu_j in Unt, not necc. the larger mass
            

            p_crit = xi_max * dt_over_dV \
                     * ker.compute_kernel_hydro(
                           radii[i], radii[j],
                           E_col_grid[ind_kernel[i], ind_kernel[j]],
                           # abs difference of 2D vectors
                           math.sqrt( (vel[0,i] - vel[0,j])**2
                                      + (vel[1,i] - vel[1,j])**2 ) )

            if p_crit > 1.0:
                # multiple collection
                xi_col = p_crit * xi_min # p_crit = xi_col / xi_min
                m_w[ind_min] = m_w_min + p_crit * m_w_max
                m_s[ind_min] = m_s_min + p_crit * m_s_max
                # mass changed: update radius
                # IMPORTANT: need to but (m_w + m_s) in paranthesis for jit
                radii[ind_min] =\
                    mp.compute_radius_from_mass((m_w[ind_min] + m_s[ind_min]),
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
                    m_w[ind_min] = 2.0 * ( xi_min*m_w_min + xi_max*m_w_max ) \
                                      / xi_ges
                    m_s[ind_min] = 2.0 * ( xi_min*m_s_min + xi_max*m_s_max ) \
                                      / xi_ges
                    m_w[ind_max] = m_w[ind_min]
                    m_s[ind_max] = m_s[ind_min]
                    # IMPORTANT: need to put (m_w + m_s) in paranthesis for jit
                    radii[ind_min] =\
                        mp.compute_radius_from_mass(
                                (m_w[ind_min] + m_s[ind_min]),
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
                    m_w[ind_min] += m_w_max
                    m_s[ind_min] += m_s_max
                    radii[ind_min] =\
                        mp.compute_radius_from_mass(
                                (m_w[ind_min] + m_s[ind_min]),
                                mass_densities[ind_min])
                    xis[ind_max] -= xi_min
                    # rad of ind_min changed -> update kernel index:
                    ind_kernel[ind_min] = \
                        compute_kernel_index(radii[ind_min], R_kernel_low_log,
                                             bin_factor_R_log, no_kernel_bins)
            cnt += 1
collision_step_Ecol_grid_R_2D_multicomp = \
    njit()(collision_step_Ecol_grid_R_2D_multicomp_np)

def collision_step_Ecol_const_2D_multicomp_np(
        xis, m_w, m_s, radii, vel, mass_densities,
        dt_over_dV, E_col_grid, no_cols):
    """One collision step for a number of SIPs in dV, discretized E_col(R1, R2)
    
    Uses the same constant collection efficiency
    'E_col_grid' (float, no array) for all collisions.
    The collection pair (i-j) order is the same as in Unterstrasser 2017.
    Updates xis, m_w, m_s, radii and no_cols.
    
    Parameters
    ----------
    xis: ndarray, dtype=float
        1D array of SIP multiplicities (real numbers, non-integer)
    m_w: ndarray, dtype=float
        1D array of SIP water-masses (unit = 1E-18 kg)
    m_s: ndarray, dtype=float
        1D array of SIP solute-masses, only one solute type (unit = 1E-18 kg)
    radii: ndarray, dtype=float
        1D array of SIP radii (unit = microns)
    vel: ndarray, dtype=float
        1D array of SIP velocites (unit = m/s)
        is kept stationary for one coll. step, but may vary from SIP to SIP
    mass_densities: ndarray, dtype=float
        mass densities of the SIPs (kg/m^3).
        is kept stationary for one coll. step, but may vary from SIP to SIP
    dt_over_dV: float
        dt/dV, where: dt = collision time step,
        dV = volume, in which particles are well mixed and collide
    E_col_grid: float
        constant collection efficiency E_col for all collisions
    no_cols: ndarray, shape=(2,), type=int
        counts the collisions
        no_cols[0] = number of ordinary collision,
        no_cols[1] = number of multiple collision event
    
    """     
    
    no_SIPs = xis.shape[0]

    rnd = np.random.rand( (no_SIPs*(no_SIPs-1))//2 )
    
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
            m_w_min = m_w[ind_min] # = mu_i in Unt, not necc. the smaller mass
            m_w_max = m_w[ind_max] # = mu_j in Unt, not necc. the larger mass
            m_s_min = m_s[ind_min] # = mu_i in Unt, not necc. the smaller mass
            m_s_max = m_s[ind_max] # = mu_j in Unt, not necc. the larger mass

            p_crit = xi_max * dt_over_dV \
                     * ker.compute_kernel_hydro(
                           radii[i], radii[j],
                           E_col_grid,
                           # abs difference of 2D vectors
                           math.sqrt( (vel[0,i] - vel[0,j])**2
                                      + (vel[1,i] - vel[1,j])**2 ) )

            if p_crit > 1.0:
                # multiple collection
                xi_col = p_crit * xi_min # p_crit = xi_col / xi_min
                m_w[ind_min] = m_w_min + p_crit * m_w_max
                m_s[ind_min] = m_s_min + p_crit * m_s_max
                # mass changed: update radius
                # IMPORTANT: need to put (m_w + m_s) in paranthesis for jit
                radii[ind_min] =\
                    mp.compute_radius_from_mass((m_w[ind_min] + m_s[ind_min]),
                                                 mass_densities[ind_min])
                xis[ind_max] -= xi_col
                # rad of ind_min changed -> update kernel index:
                no_cols[1] += 1
            elif p_crit > rnd[cnt]:
                no_cols[0] += 1
                xi_rel_dev = (xi_max-xi_min)/xi_max
                if xi_rel_dev < 1.0E-5:
                    print("xi_i approx xi_j, xi_rel_dev =",
                          xi_rel_dev,
                          " in collision")
                    xi_ges = xi_min + xi_max
                    m_w[ind_min] = 2.0 * ( xi_min*m_w_min + xi_max*m_w_max ) \
                                      / xi_ges
                    m_s[ind_min] = 2.0 * ( xi_min*m_s_min + xi_max*m_s_max ) \
                                      / xi_ges
                    m_w[ind_max] = m_w[ind_min]
                    m_s[ind_max] = m_s[ind_min]
                    # IMPORTANT: need to put (m_w + m_s) in paranthesis for jit
                    radii[ind_min] =\
                        mp.compute_radius_from_mass(
                                (m_w[ind_min] + m_s[ind_min]),
                                mass_densities[ind_min])
                    radii[ind_max] = radii[ind_min]
                    xis[ind_max] = 0.5 * 0.7 * xi_ges
                    xis[ind_min] = 0.5 * xi_ges - xis[ind_max]
                    # radius of ind_min AND ind_max changed to same radii
                else:
                    m_w[ind_min] += m_w_max
                    m_s[ind_min] += m_s_max
                    # IMPORTANT: need to put (m_w + m_s) in paranthesis for jit
                    radii[ind_min] =\
                        mp.compute_radius_from_mass(
                                (m_w[ind_min] + m_s[ind_min]),
                                mass_densities[ind_min])
                    xis[ind_max] -= xi_min
            cnt += 1
collision_step_Ecol_const_2D_multicomp = \
    njit()(collision_step_Ecol_const_2D_multicomp_np)

### EXECUTE COLLISION STEP IN ALL CELLS OF THE 2D GRID
def collision_step_Ecol_grid_R_all_cells_2D_multicomp_np(
        xis, m_w, m_s, vel, grid_temperature, cells, no_cells,
        dt_over_dV, E_col_grid, no_kernel_bins,
        R_kernel_low_log, bin_factor_R_log, no_cols, solute_type):
    """One collision step in all cells of the 2D spatial grid
    
    Uses a discretized collection efficiency E_col(R1,R2)
    OR a constant collection efficiency
    based on a logarithmic radius grid.
    Each spatial grid cell is independent regarding the collision step.
    The collection pair (i-j) order is the same as in Unterstrasser 2017
    Updates xis, m_w, m_s, and no_cols.
    Does not take radii as input. Does not update or return radii.
    Calculations in all algorithms are always based on m_w and m_s.
    No need to keep radii updated at all times.
    Keeps velocities and mass densities fixed for one collision timestep.
    The np-method is 2 times faster than the jitted version in this case.
    1000 collision steps for 75 x 75 cells take 3240 s = 54 min    
    
    Parameters
    ----------
    xis: ndarray, dtype=float
        1D array of SIP multiplicities (real numbers, non-integer)
    m_w: ndarray, dtype=float
        1D array of SIP water-masses (unit = 1E-18 kg)
    m_s: ndarray, dtype=float
        1D array of SIP solute-masses, only one solute type (unit = 1E-18 kg)
    grid_temperature: ndarray, shape=(no_cells_x,no_cells_z), dtype=float
        Ambient temperature in each grid cell
    vel: ndarray, dtype=float
        1D array of SIP velocites (unit = m/s)
        is kept stationary for one coll. step, but may vary from SIP to SIP
    mass_densities: ndarray, dtype=float
        mass densities of the SIPs (kg/m^3).
        is kept stationary for one coll. step, but may vary from SIP to SIP
    dt_over_dV: float
        dt/dV, where: dt = collision time step,
        dV = volume, in which particles are well mixed and collide
    E_col_grid: ndarray or float
        Discretized collection efficiency E_col(R1,R2) based on log. rad. grid
        OR constant coll. eff. for all particles
    no_kernel_bins: int
        number of bins used to discretize the collection efficiencies
    R_kernel_low_log: float
        nat. log of the lower radius boundary of the kernel discretization
    bin_factor_R_log: float
        nat. log of the radius bin factor => R_(n+1) = R_n * bin_factor_R_log
    no_cols: ndarray, shape=(2,), type=int
        counts the collisions
        no_cols[0] = number of ordinary collision,
        no_cols[1] = number of multiple collision event
    solute_type: str
        Name of the solute material.
        Either 'AS' (ammonium sulfate) or 'NaCl'
    
    """    
    
    for i in range(no_cells[0]):
        mask_i = (cells[0] == i)
        for j in range(no_cells[1]):
            mask_ij = np.logical_and(mask_i , (cells[1] == j))
            
            # for given cell:
            xi_cell = xis[mask_ij]
            m_w_cell = m_w[mask_ij]
            m_s_cell = m_s[mask_ij]
            T_p_cell = grid_temperature[i,j]
            
            if solute_type == "AS":
                R_p_cell, w_s_cell, rho_p_cell =\
                    mp.compute_R_p_w_s_rho_p_AS(m_w_cell, m_s_cell, T_p_cell)
            elif solute_type == "NaCl":
                R_p_cell, w_s_cell, rho_p_cell =\
                    mp.compute_R_p_w_s_rho_p_NaCl(m_w_cell, m_s_cell, T_p_cell)
            
            vels = vel[:,mask_ij]
            
            if isinstance(E_col_grid, float):
                collision_step_Ecol_const_2D_multicomp(
                        xi_cell, m_w_cell, m_s_cell, R_p_cell, vels,
                        rho_p_cell,                        
                        dt_over_dV, E_col_grid,
                        no_cols)                
            else:
                collision_step_Ecol_grid_R_2D_multicomp(
                        xi_cell, m_w_cell, m_s_cell, R_p_cell, vels,
                        rho_p_cell,
                        dt_over_dV, E_col_grid, no_kernel_bins,
                        R_kernel_low_log, bin_factor_R_log, no_cols)            
            
            xis[mask_ij] = xi_cell
            m_w[mask_ij] = m_w_cell   
            m_s[mask_ij] = m_s_cell   
#collision_step_Ecol_grid_R_all_cells_2D_multicomp = \
#    njit()(collision_step_Ecol_grid_R_all_cells_2D_multicomp_np)
