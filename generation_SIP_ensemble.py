#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

GENERATION OF SIP ENSEMBLES

for initialization, the "SingleSIP" method is applied, as proposed by
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

from microphysics import compute_mass_from_radius
from distributions import pdf_expo, pdf_lognormal

#%% GENERATION OF SINGLE SIP ENSEMBLES FOR CERTAIN DISTRIBUTIONS

# generate SIP ensemble from pure PDFs
# return masses and weights 
# -> In a second step multiply weights by 
# total number of real particles in cell "nrpc" to get multiplicities
# note that the multiplicities are not integers but floats
# exponential distribution:
# f = 1/m_avg * exp(-m/m_avg)
def gen_mass_ensemble_weights_SinSIP_expo_np(
        m_mean, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low=1.0E6,
        seed=3711, setseed=True):
    if setseed: np.random.seed(seed)
    m_low = compute_mass_from_radius(r_critmin, mass_density) # in 1E-18 kg
    m_mean_inv = 1.0 / m_mean
    
    bin_factor = 10**(1.0/kappa)
    m_high = m_low * m_high_over_m_low
    m_left = m_low

    l_max = int(kappa * np.log10(m_high_over_m_low))
    rnd = np.random.rand( l_max )
    
    if weak_threshold:
        rnd2 = np.random.rand( l_max )

    weights = np.zeros(l_max, dtype = np.float64)
    masses = np.zeros(l_max, dtype = np.float64)
    bins = np.zeros(l_max+1, dtype = np.float64)
    bins[0] = m_left

    bin_n = 0
    while m_left < m_high:
        m_right = m_left * bin_factor
        bin_width = m_right - m_left
        mu = m_left + rnd[bin_n] * bin_width
        weights[bin_n] = pdf_expo(mu, m_mean_inv) * bin_width
        masses[bin_n] = mu
        m_left = m_right

        bin_n += 1
        bins[bin_n] = m_left

    weight_max = weights.max()
    weight_critmin = weight_max * eta

    valid_ids = np.full(l_max, True)
    for bin_n in range(l_max):
        if weights[bin_n] < weight_critmin:
            if weak_threshold:
                if rnd2[bin_n] < weights[bin_n] / weight_critmin:
                    weights[bin_n] = weight_critmin
                else: valid_ids[bin_n] = False
            else: valid_ids[bin_n] = False
    weights = weights[valid_ids]
    masses = masses[valid_ids]
    
    return masses, weights, m_low, bins
gen_mass_ensemble_weights_SinSIP_expo =\
    njit()(gen_mass_ensemble_weights_SinSIP_expo_np)   

# generate SIP ensemble from pure PDFs
# return masses and weights 
# -> In a second step multiply weights by total number of real particles
# per cell and mode "no_rpcm" to get multiplicities
# note that the multiplicities are not integers but floats
# lognormal distribution for mode 'k':
# f = \frac{DNC_k}{\sqrt{2 \pi} \, \ln (\sigma_k) m}
# \exp [ -(\frac{\ln (m / \mu_k)}{\sqrt{2} \, \ln (\sigma_k)} )^2 ]
def gen_mass_ensemble_weights_SinSIP_lognormal_np(
        mu_m_log, sigma_m_log,
        mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low,
        seed, setseed=True):
    if setseed: np.random.seed(seed)
    m_low = compute_mass_from_radius(r_critmin, mass_density) # in 1E-18 kg
    
    bin_factor = 10**(1.0/kappa)
    m_left = m_low

    l_max = int( math.ceil( kappa * np.log10(m_high_over_m_low)))
    rnd = np.random.rand( l_max )
    
    if weak_threshold:
        rnd2 = np.random.rand( l_max )

    weights = np.zeros(l_max, dtype = np.float64)
    masses = np.zeros(l_max, dtype = np.float64)
    bins = np.zeros(l_max+1, dtype = np.float64)
    bins[0] = m_left

    bin_n = 0
    
    for bin_n in range(l_max):
        m_right = m_left * bin_factor
        bin_width = m_right - m_left
        mu = m_left + rnd[bin_n] * bin_width
        weights[bin_n] = pdf_lognormal(mu, mu_m_log, sigma_m_log) * bin_width
        masses[bin_n] = mu
        bins[bin_n+1] = m_right
        m_left = m_right
        
    weight_max = weights.max()
    weight_critmin = weight_max * eta

    valid_ids = np.full(l_max, True)
    for bin_n in range(l_max):
        if weights[bin_n] < weight_critmin:
            if weak_threshold:
                if rnd2[bin_n] < weights[bin_n] / weight_critmin:
                    weights[bin_n] = weight_critmin
                else: valid_ids[bin_n] = False
            else: valid_ids[bin_n] = False
    weights = weights[valid_ids]
    masses = masses[valid_ids]
    
    return masses, weights, m_low, bins
gen_mass_ensemble_weights_SinSIP_lognormal =\
    njit()(gen_mass_ensemble_weights_SinSIP_lognormal_np)   

#%% CREATE ENSEMBLE MASSES AND WEIGHTS IN EACH CELL of a given z-level (j-lvl)
# at the z-level, the mass density and thereby the number of real particles per
# cell is known
# create SIPs for each cell of the level, then
# lump all in one array and assign cell - x values for indexing
# if no_modes = 1: monomodal, mu_m_log, sigma_m_log = scalars
# if no_modes > 1: multimodal, mu_m_log = [mu0, mu1, ...] same for sigm & kappa
# mass density is only for conversion r_critmin -> m_critmin
# for lognorm weak threshold: kappa = 3.5 -> no_SIPs_avg = 20.2
def gen_mass_ensemble_weights_SinSIP_lognormal_z_lvl(no_modes,
        mu_m_log, sigma_m_log, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low, seed, no_cells_x, no_rpcm, setseed = True):
    
    if setseed:
        np.random.seed(seed)
    
    # numpy arrays not used, since nr of SIPs is not equal in each cell
    masses_lvl = []
    xis_lvl = []
    cells_x_lvl = []
    
    no_spc_lvl = np.zeros(no_cells_x, dtype = np.int64)
    
    if no_modes > 1:
        modes_lvl = []
    
    for i in range(no_cells_x):
        if no_modes == 1:
            masses, weights, m_low, bins = \
                gen_mass_ensemble_weights_SinSIP_lognormal(
                    mu_m_log, sigma_m_log, mass_density,
                    dV, kappa, eta, weak_threshold, r_critmin,
                    m_high_over_m_low, seed, setseed=False)
            no_sp_cell = len(masses)
            no_spc_lvl[i] += no_sp_cell
            masses_lvl.append(masses)
            xis_lvl.append(weights*no_rpcm)
            cells_x_lvl.append( i*np.ones(no_sp_cell, dtype=np.int64) )
        elif no_modes > 1:
            for mode_n in range(no_modes):
                masses, weights, m_low, bins = \
                    gen_mass_ensemble_weights_SinSIP_lognormal(
                        mu_m_log[mode_n], sigma_m_log[mode_n], mass_density,
                        dV, kappa[mode_n], eta, weak_threshold,
                        r_critmin[mode_n],
                        m_high_over_m_low, seed, setseed=False)
                no_sp_cell = len(masses)
                no_spc_lvl[i] += no_sp_cell
                masses_lvl.append(masses)
                xis_lvl.append(weights * no_rpcm[mode_n])
                cells_x_lvl.append( i*np.ones(no_sp_cell, dtype=np.int64) )
                modes_lvl.append(mode_n*np.ones(no_sp_cell, dtype=np.int64))
    
    masses_lvl = np.concatenate(masses_lvl)
    xis_lvl = np.concatenate(xis_lvl)
    cells_x_lvl = np.concatenate(cells_x_lvl)
    if no_modes > 1:
        modes_lvl = np.concatenate(modes_lvl)
    else: modes_lvl = np.zeros_like(cells_x_lvl)
                    
    return masses_lvl, xis_lvl, cells_x_lvl, modes_lvl, no_spc_lvl

# expo distr. not finished for z-lvl
# see above, now for expo dist
# at the z-level, the mass density and thereby the number of real particles per
# cell is known
# create SIPs for each cell of the level, then
# lump all in one array and assign cell - x values for indexing
# if no_modes = 1: monomodal, mu_m_log, sigma_m_log = scalars
# if no_modes > 1: multimodal, mu_m_log = [mu0, mu1, ...] same for sigm & kappa
# mass density is only for conversion r_critmin -> m_critmin
# for lognorm weak threshold: kappa = 3.5 -> no_SIPs_avg = 20.2
# expo is monomodal ONLY!
# IN WORK: UNFINISHED!
#def gen_mass_ensemble_weights_SinSIP_expo_z_lvl(
#        m_mean, mass_density,
#        dV, kappa, eta, weak_threshold, r_critmin,
#        m_high_over_m_low, seed, no_cells_x, no_rpcm, setseed = True):
#    
#    if setseed:
#        np.random.seed(seed)
#    
#    # numpy arrays not used, since nr of SIPs is not equal in each cell
#    masses_lvl = []
#    xis_lvl = []
#    cells_x_lvl = []
#    
#    no_spc_lvl = np.zeros(no_cells_x, dtype = np.int64)
#    
#    for i in range(no_cells_x):
#        # NOT FINISHED
#        masses, weights, m_low, bins = \
#            gen_mass_ensemble_weights_SinSIP_expo(
#                    m_mean, mass_density,
#                    dV, kappa, eta, weak_threshold, r_critmin,
#                    m_high_over_m_low,
#                    seed, setseed=False)        
#        no_sp_cell = len(masses)
#        no_spc_lvl[i] += no_sp_cell
#        masses_lvl.append(masses)
#        xis_lvl.append(weights*no_rpcm)
#        cells_x_lvl.append( i*np.ones(no_sp_cell, dtype=np.int64) )
#    
#    masses_lvl = np.concatenate(masses_lvl)
#    xis_lvl = np.concatenate(xis_lvl)
#    cells_x_lvl = np.concatenate(cells_x_lvl)
#                    
#    return masses_lvl, xis_lvl, cells_x_lvl, no_spc_lvl
    
#%% CREATE ENSEMBLE MASSES AND WEIGHTS IN EACH CELL
    
# this function is currently not in use. particles are gen. for each z-lvl
# because of layering process with saturation adjustment.
# this function could be used for a box model with connected grid cells.    
# njit not possible because of lists
def gen_mass_ensemble_weights_SinSIP_lognormal_grid(
        mu_m_log, sigma_m_log, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low, seed, no_cells):
    
    mass_grid_ji = []
    weights_grid_ji = []
    
    np.random.seed(seed)
    
    no_sp_placed_ji = np.zeros( no_cells, dtype = np.int64 )
    
    for j in range(no_cells[0]):
        mg_ = []
        wg_ = []
        for i in range(no_cells[1]):
            masses, weights, m_low, bins = \
                gen_mass_ensemble_weights_SinSIP_lognormal_np(
                    mu_m_log, sigma_m_log, mass_density,
                    dV, kappa, eta, weak_threshold, r_critmin,
                    m_high_over_m_low, seed, setseed=False)
            mg_.append(masses)
            wg_.append(weights)
            no_sp_placed_ji[j,i] = len(masses)
        mass_grid_ji.append(mg_)
        weights_grid_ji.append(wg_)
    return mass_grid_ji, weights_grid_ji, no_sp_placed_ji

# expo distr. not finished for grid
# no_cells = [no_c_x, no_c_z]
#def gen_mass_ensemble_weights_SinSIP_expo_grid_np(
#        m_mean, mass_density,
#        dV, kappa, eta, weak_threshold, r_critmin,
#        m_high_over_m_low, seed, no_cells, no_rpcm):
#    
#    np.random.seed(seed)
#    
#    masses = []
#    cells_x = []
#    cells_z = []
#    xis = []
#    
#    for j in range(no_cells[1]):
#        masses_lvl, xis_lvl, cells_x_lvl, no_spc_lvl = \
#            gen_mass_ensemble_weights_SinSIP_expo_z_lvl(
#                m_mean, mass_density,
#                dV, kappa, eta, weak_threshold, r_critmin,
#                m_high_over_m_low, seed, no_cells[0], no_rpcm, setseed = False)
#        masses.append(masses_lvl)
#        cells_x.append(cells_x_lvl)
#        cells_z.append(np.ones_like(cells_x_lvl) * j)
#        xis.append(xis_lvl)
#    
#    masses = np.concatenate(masses)
#    xis = np.concatenate(xis)
#    cells = np.array( (np.concatenate(cells_x), np.concatenate(cells_z)) )
#    
#    return masses, xis, cells