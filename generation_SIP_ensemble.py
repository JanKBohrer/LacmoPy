#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

GENERATION OF SIP ENSEMBLES

for initialization, the 'SingleSIP' method is applied, as proposed by
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

#%% GENERATION OF SIP ENSEMBLES FOR CERTAIN DISTRIBUTIONS

def gen_mass_ensemble_weights_SinSIP_expo_np(
        m_mean, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low=1.0E6,
        seed=3711, setseed=True):
    """Generate weights for a particle ensemble from expo. mass distribution
    
    Adapted from 'SingleSIP'-method by Unterstrasser 2017.
    Exponential probability density function:
    f(m) = 1/m_mean * exp(-m / m_mean).
    This function generates a 1D mass-array and a corresponding weight-array,
    with a weight for each mass. The sum of weights adds up to a value close
    to one. Due to restrictions of the allowed relative minimum weight, the
    sum of weights will not exactly be one, but fluctuate around one.
    To obtain the corresponding array of SIP-multiplicities,
    one can multiply the array of weights by the number of
    real particles in the current cell or simulation domain.
    In contrast to the super-droplet approach by
    Shima et al. 2009, Q. J. R. Meteorol. Soc. 135: 1307–1320,
    the multiplicities will not be integers, but floats, which are also
    allowed to assume values smaller than one.
    
    Parameters
    ----------
    m_mean: float
        Expected value of the exponential PDF (1E-18 kg).
    mass_density: float
        Particle mass density (kg/m^3)
    dV: float
        Volume, in which the particles are generated (m^3).
    kappa: float
        'kappa' parameter, which defines the number of SIPs per simulation
        box, as defined in Unterstrasser 2017, SingleSIP method.
    eta: float
        'eta' parameter, which relatively defines a lower border
        for the initial allowed multiplicity size,
        as defined in Unterstrasser 2017, SingleSIP method
    weak_threshold: bool
        Sets weak (True) or fix (False) threshold during SIP generation,
        as defined in Unterstrasser 2017, SingleSIP method
    r_critmin: float
        Defines the smallest allowed particle radius and thereby the
        smallest allowed mass of the SIP-ensemble (microns).
    m_high_over_m_low: float, optional
        Ratio of largest to smallest allowed mass of the SIP-ensemble. The
        smallest mass is defined by the smallest radius r_critmin.
    seed: int, optional
        Seed of the Numpy random number generator.
    setseed: bool, optional
        'True' to initialize the Numpy random number generator with 'seed'.
    
    Returns
    -------
    masses: ndarray, dtype=float
        1D array of generated simulation-particle masses. (1E-18 kg)
    weights: ndarray, dtype=float
        1D array of generated weights. To obtain the corresponding array
        of SIP-multiplicities, one can multiply the array of weights by the
        number of real particles in the current cell or simulation domain.
    m_low: float
        Smallest allowed particle mass, corr. to 'r_critmin' (1E-18 kg).
    bins: ndarray, dtype=float
        1D array holding the borders of the logarithmic mass bins. One mass
        value is chosen randomly per mass bin.

    """
    
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

def gen_mass_ensemble_weights_SinSIP_lognormal_np(
        mu_m_log, sigma_m_log,
        mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low,
        seed, setseed=True):
    """Generate weights for a particle ensemble from lognorm. mass distrib.
    
    Adapted from 'SingleSIP'-method by Unterstrasser 2017.
    Lognormal probability density function:
    f(m) = \exp( -0.5 ( ( \ln( m ) - mu_m_log ) / sigma_m_log )^2 )
           / ( m * two_pi_sqrt * sigma_m_log )
    This function generates a 1D mass-array and a corresponding weight-array,
    with a weight for each mass. The sum of weights adds up to a value close
    to one. Due to restrictions of the allowed relative minimum weight, the
    sum of weights will not exactly be one, but fluctuate around one.
    To obtain the corresponding array of SIP-multiplicities,
    one can multiply the array of weights by the number of
    real particles in the current cell or simulation domain.
    In contrast to the super-droplet approach by
    Shima et al. 2009, Q. J. R. Meteorol. Soc. 135: 1307–1320,
    the multiplicities will not be integers, but floats, which are also
    allowed to assume values smaller than one.
    
    Parameters
    ----------
    mu_m_log: float
        Log. nat. of the geometric expected value
        of the lognormal PDF [ln(1E-18 kg)].
    sigma_m_log: float
        Log. nat. of the geometric standard deviation of the lognormal PDF
    mass_density: float
        Particle mass density (kg/m^3)
    dV: float
        Volume, in which the particles are generated (m^3).
    kappa: float
        'kappa' parameter, which defines the number of SIPs per simulation
        box, as defined in Unterstrasser 2017, SingleSIP method.
    eta: float
        'eta' parameter, which relatively defines a lower border
        for the initial allowed multiplicity size,
        as defined in Unterstrasser 2017, SingleSIP method
    weak_threshold: bool
        Sets weak (True) or fix (False) threshold during SIP generation,
        as defined in Unterstrasser 2017, SingleSIP method
    r_critmin: float
        Defines the smallest allowed particle radius and thereby the
        smallest allowed mass of the SIP-ensemble (microns).
    m_high_over_m_low: float, optional
        Ratio of largest to smallest allowed mass of the SIP-ensemble. The
        smallest mass is defined by the smallest radius r_critmin.
    seed: int, optional
        Seed of the Numpy random number generator.
    setseed: bool, optional
        'True' to initialize the Numpy random number generator with 'seed'.
    
    Returns
    -------
    masses: ndarray, dtype=float
        1D array of generated simulation-particle masses. (1E-18 kg)
    weights: ndarray, dtype=float
        1D array of generated weights. To obtain the corresponding array
        of SIP-multiplicities, one can multiply the array of weights by the
        number of real particles in the current cell or simulation domain.
    m_low: float
        Smallest allowed particle mass, corr. to 'r_critmin' (1E-18 kg).
    bins: ndarray, dtype=float
        1D array holding the borders of the logarithmic mass bins. One mass
        value is chosen randomly per mass bin.

    """
    
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

def gen_mass_ensemble_SinSIP_lognormal_z_lvl(no_modes,
        mu_m_log, sigma_m_log, mass_density,
        dV, kappa, eta, weak_threshold, r_critmin,
        m_high_over_m_low, seed, no_cells_x, no_rpcm, setseed = True):
    """Generate simulation-particle ensembles in cells at the same height
    
    Adapted from 'SingleSIP'-method by Unterstrasser 2017.
    Multimodal lognormal probability density function:
    f(m) = \sum_k \exp( -0.5 ( ( \ln( m ) - mu_m_log_k ) / sigma_m_log_k )^2 )
           / ( m * two_pi_sqrt * sigma_m_log_k )
    Given a 2D spatial grid. At the vertical z-level of grid cells, the mass
    density and thereby the number of real particles per cell is known.
    In each cell of the z-level, this function generates a 1D mass-array
    and a corresponding weight-array with a weight for each mass.
    The sum of weights in each cell adds up to a value close to one. Due to
    restrictions of the allowed relative minimum weight, the sum of weights
    will not exactly be one, but fluctuate around one. To obtain the
    corresponding array of SIP-multiplicities, the array of weights is
    multiplied by the number of real particles in the current cell.
    In contrast to the super-droplet approach by
    Shima et al. 2009, Q. J. R. Meteorol. Soc. 135: 1307–1320,
    the multiplicities will not be integers, but floats, which are also
    allowed to assume values smaller than one.
    
    Parameters
    ----------
    no_modes: int
        Number of modes of the lognormal PDF. If no_modes==1, then the
        parameters below must be provided as single floats.
        If no_modes>1, then the parameters below must be provided as ndarrays.
    mu_m_log: float or ndarray, dtype=float
        1D array holding the log. nat. of the geometric expected values
        of the lognormal PDF for each mode [ln(1E-18 kg)].
        If no_modes==1: Provide single float
    sigma_m_log: float or ndarray, dtype=float
        1D array holding the log. nat. of the geometric standard deviation
        of the lognormal PDF for each mode.
        If no_modes==1: Provide single float
    mass_density: float
        Particle mass density (kg/m^3)
    dV: float
        Volume, in which the particles are generated (m^3).
    kappa: float or ndarray, dtype=float
        1D array holding for each mode the 'kappa' parameter,
        which defines the number of SIPs per simulation box,
        as defined in Unterstrasser 2017, SingleSIP method.
        For lognorm weak threshold: kappa = 3.5 => no_SIPs_avg approx. 20.2
        If no_modes==1: Provide single float
    eta: float
        'eta' parameter, which relatively defines a lower border
        for the initial allowed multiplicity size,
        as defined in Unterstrasser 2017, SingleSIP method
    weak_threshold: bool
        Sets weak (True) or fix (False) threshold during SIP generation,
        as defined in Unterstrasser 2017, SingleSIP method
    r_critmin: float or ndarray, dtype=float
        1D array, holding for each mode the smallest allowed particle radius
        and thereby the smallest allowed mass of the SIP-ensemble (microns).
        If no_modes==1: Provide single float
    m_high_over_m_low: float
        Ratio of largest to smallest allowed mass of the SIP-ensemble. The
        smallest mass is defined by the smallest radius r_critmin.
    seed: int
        Seed of the Numpy random number generator.
    no_cells_x: int
        Number of cells in horizontal direction of the 2D spatial grid.
    no_rpcm: ndarray
        Number of real particles per cell and mode:
        no_rpcm[0] = Number of real particles per cell in mode 0 etc.
    setseed: bool, optional
        'True' to initialize the Numpy random number generator with 'seed'.
    
    Returns
    -------
    masses_lvl: ndarray, dtype=float
        1D array holding generated simulation-particle masses of the entire
        z-level (1E-18 kg). The masses are assigned to cells by 'cells_x_lvl'.
    xis_lvl: ndarray, dtype=float
        1D array holding generated multiplicities of the entire z-level.
        The 'xis' are assigned to cells by 'cells_x_lvl'.
    cells_x_lvl: ndarray, dtype=int
        1D array providing the horizontal cell-index for 'masses' and 'xis'.
    modes_lvl: ndarray, dtype=int
        1D array providing the mode number for 'masses' and 'xis'.
    no_spc_lvl: ndarray, dtype=int
        1D array holding the number of generated simulation-particles
        in each cell of the z-level.

    """    
    
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
#def gen_mass_ensemble_weights_SinSIP_lognormal_grid(
#        mu_m_log, sigma_m_log, mass_density,
#        dV, kappa, eta, weak_threshold, r_critmin,
#        m_high_over_m_low, seed, no_cells):
#    
#    mass_grid_ji = []
#    weights_grid_ji = []
#    
#    np.random.seed(seed)
#    
#    no_sp_placed_ji = np.zeros( no_cells, dtype = np.int64 )
#    
#    for j in range(no_cells[0]):
#        mg_ = []
#        wg_ = []
#        for i in range(no_cells[1]):
#            masses, weights, m_low, bins = \
#                gen_mass_ensemble_weights_SinSIP_lognormal_np(
#                    mu_m_log, sigma_m_log, mass_density,
#                    dV, kappa, eta, weak_threshold, r_critmin,
#                    m_high_over_m_low, seed, setseed=False)
#            mg_.append(masses)
#            wg_.append(weights)
#            no_sp_placed_ji[j,i] = len(masses)
#        mass_grid_ji.append(mg_)
#        weights_grid_ji.append(wg_)
#    return mass_grid_ji, weights_grid_ji, no_sp_placed_ji

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