#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

COLLISION KERNELS

"Golovin": Analytic sum-off-mass kernel by
Golovin 1963, Izv. Geophys. Ser 5, 482

"Long_Bott": Analytic kernel from Long 1974, J. Atm. Sci. 31, 1040,
modified by Bott 1998, J. Atm. Sci. 55, 2284

"Hall_Bott": Tabulated kernel from Hall 1980, J. Atm. Sci. 37, 2486,
modified by Seeßelberg 1996, Atm. Research. 40, 33 with extensions from
Davis 1972, J. Atm. Sci. 29, 911; Jonas 1972, Q. J. R. Meteorol. Soc. 98, 681
and Lin 1975, J. Atm. Sci. 22, 1065 as used in Bott 1998, J. Atm. Sci. 55, 2284

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS
import numpy as np
from numba import njit, vectorize
import math

import constants as c
from grid import bilinear_weight
from microphysics import compute_radius_from_mass
from material_properties import compute_viscosity_air
from material_properties import compute_surface_tension_water

#%% PERMUTATION

@njit()
def generate_permutation(N):
    """Generates a random permutation of the integer set {0,1,2,...,N-1}
    
    The method is described in
    Shima 2009: Q. J. R. Meteorol. Soc. 135: 1307–1320
    
    Parameters
    ----------
    N: int
        Number of elements of the set {0,1,2,...,N-1}
    
    Returns
    -------
    permutation: ndarray, dtype=int
        1D array holding the permutation
    
    """
    
    permutation = np.zeros(N, dtype=np.int64)
    for n_next in range(1, N):
        q = np.random.randint(0,n_next+1)
        if q==n_next:
            permutation[n_next] = n_next
        else:
            permutation[n_next] = permutation[q]
            permutation[q] = n_next
    return permutation

#%% TERMINAL VELOCITY AS FUNCTION OF RADIUS

@njit()
def compute_polynom(par, x):
    """Computes the polynom p(x)

    p(x) = par[0]*x^(N-1) + par[1]*x^(N-2) + ... + par[N]

    Parameters
    ----------
    par: ndarray, dtype=float
        1D array with the polynom parameters. par[n], n = 0, ..., N.
        The order of the polynom is N-1
    x: float
        Argument of the polynom function
    
    Returns
    -------
    res: float
        Polynom evaluated at x
    
    """
    
    res = par[0] * x + par[1]
    for a in par[2:]:
        res = res * x + a
    return res

# constants used in compute_terminal_velocity_Beard
p1_Beard = (-0.318657e1, 0.992696, -0.153193e-2,
          -0.987059e-3,-0.578878e-3,0.855176e-4,-0.327815e-5)[::-1]
p2_Beard = (-0.500015e1,0.523778e1,-0.204914e1,0.475294,-0.542819e-1,
           0.238449e-2)[::-1]
one_sixth = 1.0/6.0
v_max_Beard = 9.04929248
@njit()
def compute_terminal_velocity_Beard(R):
    """Terminal velocity of falling cloud droplets depending on radius
    
    Parametrization by Beard 1976: J. Atm. Sci. 33: 851.
    Used in Bott 1998 (s.a.) and Unterstrasser 2017, GMD 10: 1521–1548
    Material constants are taken from Bott 1998

    Parameters
    ----------
    R: float
        Cloud droplet radius (microns)
    
    Returns
    -------
    v: float
        Sedimentation velocity (m/s)
    
    """
    
    rho_w = 1.0E3 # density of droplets (water) (kg/m^3)
    rho_a = 1.225 # density of air (kg/m^3)
    viscosity_air_NTP = 1.818E-5 # (kg/(m*s))
    sigma_w_NTP = 73.0E-3 # surface tension water-air (N/m)
    # R in mu = 1E-6 m
    R_0 = 10.0
    R_1 = 535.0
    R_max = 3500.0
    drho = rho_w-rho_a
    if R < R_0:
        l0 = 6.62E-2 # mu
        # this is converted for radius instead of diameter
        # i.e. my_C1 = 4*C1_Beard
        C1 = drho*c.earth_gravity / (4.5*viscosity_air_NTP)
        # Bott uses factor 1.257 in data from Unterstrasser, but in his paper 
        # factor is 1.255
        C_sc = 1.0 + 1.257 * l0 / R
        # C_sc = 1.0 + 1.255 * l0 / R
        v = C1 * C_sc * R * R * 1.0E-12
    elif R < R_1:
        N_Da = 32.0E-18 * R*R*R * rho_a * drho \
               * c.earth_gravity / (3.0 * viscosity_air_NTP*viscosity_air_NTP)
        Y = np.log(N_Da)
        Y = compute_polynom(p1_Beard,Y)
        l0 = 6.62E-2 # mu
        # Bott uses factor 1.257 in data from Unterstrasser, but in his paper 
        # factor is 1.255
        C_sc = 1.0 + 1.257 * l0 / R
        # C_sc = 1.0 + 1.255 * l0 / R
        v = viscosity_air_NTP * C_sc * np.exp(Y)\
            / (rho_a * R * 2.0E-6)
    elif R < R_max:
        N_Bo = 16.0E-12 * R*R * drho * c.earth_gravity / (3.0 * sigma_w_NTP)
        N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
                 * rho_a * rho_a 
                 / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
        Y = np.log(N_Bo * N_P16)
        Y = compute_polynom(p2_Beard,Y)
        v = viscosity_air_NTP * N_P16 * np.exp(Y)\
            / (rho_a * R * 2.0E-6)
    else: v = v_max_Beard
    return v

@vectorize("float64(float64)")
def compute_terminal_velocity_Beard_vec(R):
    """Terminal velocity of falling cloud droplets depending on radius
    
    Vectorized form using Numba package
    Parametrization by Beard 1976: J. Atm. Sci. 33: 851.
    Used in Bott 1998 (s.a.) and Unterstrasser 2017, GMD 10: 1521–1548
    Material constants are taken from Bott 1998

    Parameters
    ----------
    R: float
        Cloud droplet radius (microns)
    
    Returns
    -------
    v: float
        Sedimentation velocity (m/s)
    
    """
    
    return compute_terminal_velocity_Beard(R)

@njit()
def update_velocity_Beard(vel, R):
    """Updates terminal velocities of falling cloud droplets dep. on radius
    
    Parametrization by Beard 1976: J. Atm. Sci. 33: 851.
    Used in Bott 1998 (s.a.) and Unterstrasser 2017, GMD 10: 1521–1548
    Material constants are taken from Bott 1998

    Parameters
    ----------
    vel: ndarray, dtype=float
        1D array of velocities (m/s). gets updated
    R: ndarray, dtype=float
        1D array of cloud droplet radii (microns)
    
    """
    
    for i in range(vel.shape[0]):
        vel[i] = compute_terminal_velocity_Beard(R[i])

# constants used in compute_terminal_velocity_Beard_my_mat_const
viscosity_air_NTP = compute_viscosity_air(293.15)
sigma_w_NTP = compute_surface_tension_water(293.15)
v_max_Beard_my_mat_const = 9.11498
@njit()
def compute_terminal_velocity_Beard_my_mat_const(R):
    """Terminal velocity of falling cloud droplets depending on radius
    
    Parametrization by Beard 1976: J. Atm. Sci. 33: 851.
    Used in Bott 1998 (s.a.) and Unterstrasser 2017, GMD 10: 1521–1548
    Material constants are taken from 'constants.py' (see sources therein)

    Parameters
    ----------
    R: float
        Cloud droplet radius (microns)
    
    Returns
    -------
    v: float
        Sedimentation velocity (m/s)
    
    """
    
    # R in mu = 1E-6 m
    R_0 = 9.5
    R_1 = 535.0
    R_max = 3500.0
    drho = c.mass_density_water_liquid_NTP - c.mass_density_air_dry_NTP
    if R < R_0:
        l0 = 6.62E-2 # mu
        # this is converted for radius instead of diameter
        # i.e. my_C1 = 4*C1_Beard
        C1 = drho*c.earth_gravity / (4.5*viscosity_air_NTP)
        C_sc = 1.0 + 1.255 * l0 / R
        v = C1 * C_sc * R * R * 1.0E-12
    elif R < R_1:
        N_Da = 32.0E-18 * R*R*R * c.mass_density_air_dry_NTP * drho \
               * c.earth_gravity / (3.0 * viscosity_air_NTP*viscosity_air_NTP)
        Y = np.log(N_Da)
        Y = compute_polynom(p1_Beard,Y)
        l0 = 6.62E-2 # mu
        C_sc = 1.0 + 1.255 * l0 / R
        v = viscosity_air_NTP * C_sc * np.exp(Y)\
            / (c.mass_density_air_dry_NTP * R * 2.0E-6)
    elif R < R_max:
        N_Bo = 16.0E-12 * R*R * drho * c.earth_gravity / (3.0 * sigma_w_NTP)
        N_P16 = (sigma_w_NTP * sigma_w_NTP * sigma_w_NTP 
                 * c.mass_density_air_dry_NTP * c.mass_density_air_dry_NTP 
                 / (viscosity_air_NTP**4 * drho * c.earth_gravity))**one_sixth
        Y = np.log(N_Bo * N_P16)
        Y = compute_polynom(p2_Beard,Y)
        v = viscosity_air_NTP * N_P16 * np.exp(Y)\
            / (c.mass_density_air_dry_NTP * R * 2.0E-6)
    else: v = v_max_Beard_my_mat_const
    return v

#%% COLLECTION EFFICIENCIES

@njit()
def linear_weight(i, weight, f):
    """Linear interpolation on a one-dimensional grid (weight based)
    
    Parameters
    ----------
    i: int
        Cell index
    weight: float
        Relative coordinate in cell i. Resides in interval [0,1)
    f: ndarray, dtype=float
        1D discrete grid f[i]
    
    Returns
    -------
        Linear interpolation between two grid points
    
    """    
    
    return f[i+1]*weight + f[i]*(1.0-weight)

# Data for the original efficiency tables of Hall 1980
# in Hall_E_col:
# row =  collector drop radius (radius of larger drop) in mu
# column = ratio of R_small/R_large "collector ratio"
fp_load = "collision/kernel_data/"
Hall_E_col = np.load(fp_load + "Hall/Hall_collision_efficiency.npy")
Hall_R_col = np.load(fp_load + "Hall/Hall_collector_radius.npy")
Hall_R_col_ratio = np.load(fp_load + "Hall/Hall_radius_ratio.npy")
@njit()
def compute_E_col_Hall(R_i, R_j):
    """Interpolates the collision efficiency for a pair of two droplets
    
    Coll. efficiency interpolated from original table of Hall 1980
    
    Parameters
    ----------
    R_i: float
        Radius of first droplet (microns)
    R_j: float
        Radius of second droplet (microns)
    
    Returns
    -------
        Interpolated collision efficiency
    
    """
    
    if R_i <= 0.0 or R_j <= 0.0:
        return 0.0
    if R_i < R_j:
        R_col = R_j
        R_ratio = R_i/R_j
    else:
        R_col = R_i
        R_ratio = R_j/R_i
    if R_col > 300.0:
        return 1.0
    else:
        # ind_R is for index of R_collection,
        # which indicates the row of Hall_E_col
        ind_R = int(R_col/10.0)
        ind_ratio = int(R_ratio/0.05)
        if ind_R == Hall_R_col.shape[0]-1:
            if ind_ratio == Hall_R_col_ratio.shape[0]-1:
                return 1.0
            else:
                weight = (R_ratio - ind_ratio * 0.05) / 0.05
                return linear_weight(ind_ratio, weight, Hall_E_col[ind_R])
        elif ind_ratio == Hall_R_col_ratio.shape[0]-1:
            weight = (R_col - ind_R * 10.0) / 10.0
            return linear_weight(ind_R, weight, Hall_E_col[:,ind_ratio])
        else:
            weight_1 = (R_col - ind_R * 10.0) / 10.0
            weight_2 = (R_ratio - ind_ratio * 0.05) / 0.05
            return bilinear_weight(ind_R, ind_ratio,
                                   weight_1, weight_2, Hall_E_col)

@njit()
def compute_E_col_Long_Bott(R_i, R_j):
    """Computes the collision efficiency for a pair of two droplets
    
    Collision efficiency parametrization by Long 1974, modified version
    by Bott 1998, used in Unterstrasser 2017  
    
    Parameters
    ----------
    R_i: float
        Radius of first droplet (microns)
    R_j: float
        Radius of second droplet (microns)
    
    Returns
    -------
        Collision efficiency
    
    """
    
    if R_j >= R_i:
        R_max = R_j
        R_min = R_i
    else:
        R_max = R_i
        R_min = R_j
        
    if R_max <= 50:
        return 4.5E-4 * R_max * R_max \
                * ( 1.0 - 3.0 / ( max(3.0, R_min) + 1.0E-2) )
    else: return 1.0    

#%% COLLECTION KERNELS
    
# input:
# R1, R2 (in mu)
# collection effic. E_col (-)
# absolute difference in terminal velocity (>0) in m/s
# output: Hydrodynamic collection kernel in m^3/s
@njit()
def compute_kernel_hydro(R_1, R_2, E_col, dv):
    """Computes the hydrodynamic collision kernel for a pair of two droplets
    
    Parameters
    ----------
    R_i: float
        Radius of first droplet (microns)
    R_j: float
        Radius of second droplet (microns)
    E_col: float
        Collision efficiency
    dv: float
        Absolute difference in the droplet velocities of drop 1 and drop 2
    
    Returns
    -------
        Collision kernel (m^3/s)
    
    """
    
    return math.pi * (R_1 + R_2) * (R_1 + R_2) * E_col * dv * 1.0E-12

@njit()
def compute_kernel_Long_Bott_R(R_i, R_j):
    """Computes the modified 'Long' collision kernel for a droplet pair
    
    Analytic kernel from Long 1974, J. Atm. Sci. 31, 1040,
    modified by Bott 1998, J. Atm. Sci. 55, 2284    
    
    Parameters
    ----------
    R_i: float
        Radius of first droplet (microns)
    R_j: float
        Radius of second droplet (microns)
    
    Returns
    -------
        Collision kernel (m^3/s)
    
    """
    
    E_col = compute_E_col_Long_Bott(R_i, R_j)
    dv = abs(compute_terminal_velocity_Beard(R_i)\
             - compute_terminal_velocity_Beard(R_j))
    return compute_kernel_hydro(R_i, R_j, E_col, dv)

@njit()
def compute_kernel_Long_Bott_m(m_i, m_j, mass_density):
    """Computes the modified 'Long' collision kernel for a droplet pair
    
    Analytic kernel from Long 1974, J. Atm. Sci. 31, 1040,
    modified by Bott 1998, J. Atm. Sci. 55, 2284    
    
    Parameters
    ----------
    m_i: float
        Mass of first droplet (1E-18 kg = 1 fg)
    m_j: float
        Mass of second droplet (1E-18 kg = 1 fg)
    
    Returns
    -------
        Collision kernel (m^3/s)
    
    """
    
    R_i = compute_radius_from_mass(m_i, mass_density)
    R_j = compute_radius_from_mass(m_j, mass_density)
    return compute_kernel_Long_Bott_R(R_i, R_j)

@njit()
def compute_kernel_Golovin(m_i, m_j):
    """Computes the analytic 'Golovin' collision kernel for a droplet pair
    
    Analytic kernel from Golovin 1963, Izv. Geophys. Ser 5, 482
    Proportionality constant as in Unterstrasser 2017
    
    Parameters
    ----------
    m_i: float
        Mass of first droplet (1E-18 kg = 1 fg)
    m_j: float
        Mass of second droplet (1E-18 kg = 1 fg)
    
    Returns
    -------
        Collision kernel (m^3/s)
    
    """    
    
    return (m_i + m_j) * 1.5E-18
    
#%% GENERATION (DISCRETIZATION) OF COL. EFFICIENCY GRIDS
    
@njit()
def interpol_bilin(i,j,p,q,f):
    """Bilinear interpolation on a two-dimensional grid (weight based)
    
    Parameters
    ----------
    i: int
        Cell index first dimension
    j: int
        Cell index second dimension
    p: float
        Relative coordinate in cell [i,j] (first dimension)
        Resides in interval [0,1)
    q: float
        Relative coordinate in cell [i,j] (2nd dimension)
        Resides in interval [0,1)
    f: ndarray, dtype=float
        2D discrete grid f[i,j]
    
    Returns
    -------
        Bilinear interpolation between four grid points
    
    """
    
    return (1.-p)*(1.-q)*f[i,j] + (1.-p)*q*f[i,j+1]\
           + p*(1.-q)*f[i+1,j] + p*q*f[i+1,j+1]

def generate_E_col_grid_R_Long_Bott_np(R_low, R_high, no_bins_10,
                                       radius_grid_in=None):
    """Generation of a discretized collision efficiency grid (radius based)
    
    When no radius grid is provided, the radius grid is generated
    between R_low and R_high with logarithmic spacing
    based on the number of bins per decade multiplication.
    R_low and R_high are both included in the radius_grid range interval
    but R_high itself might NOT be a value of the radius_grid
    (which is def by no_bins_10 and R_low).
    
    Collision efficiency parametrization by Long 1974, modified version
    by Bott 1998, used in Unterstrasser 2017.
    
    Parameters
    ----------
    R_low: float
        Smallest radius of the discretized grid (microns). Must be > 0
    R_high: float
        Largest radius of the discretized grid (microns)
    no_bins_10: int
        Number of bins per decade multiplication on logarithmic scale
    radius_grid_in: ndarray, dtype=float, optional
        1D radius grid. The collision eff. is evaluated at these points.
        When not provided, a logarithmic is automatically generated.
    
    Returns
    -------
    E_col_grid: ndarray, dtype=float
        1D array of collision efficiencies corresponding to the radius grid        
    radius_grid: ndarray, dtype=float
        1D array of radii (microns), where the coll. eff. where evaluated
    
    """
    
    if radius_grid_in is None:
        bin_factor = 10**(1.0/no_bins_10)
        no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
        radius_grid = np.zeros( no_bins, dtype = np.float64 )
        
        radius_grid[0] = R_low
        for bin_n in range(1,no_bins):
            radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor   
    else:
        radius_grid = radius_grid_in
        no_bins = radius_grid_in.shape[0]
    
    Ecol_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    for j in range(0,no_bins):
        R_j = radius_grid[j]
        for i in range(j+1):
            Ecol_grid[j,i] = compute_E_col_Long_Bott(radius_grid[i], R_j)
            Ecol_grid[i,j] = Ecol_grid[j,i]    

    return Ecol_grid, radius_grid
generate_E_col_grid_R_Long_Bott = \
    njit()(generate_E_col_grid_R_Long_Bott_np)

# Data for the collision efficiency table used by Bott 1998
# based on Hall 1980 and Seeßelberg 1996 (s.a.)
# a typo was corrected for one single entry
# of the original E_col table of Bott 1998
Hall_Bott_E_col_raw_corr =\
    np.load("collision/kernel_data/Hall/Hall_Bott_E_col_table_raw_corr.npy")
Hall_Bott_R_col_raw =\
    np.load("collision/kernel_data/Hall/Hall_Bott_R_col_table_raw.npy")
Hall_Bott_R_ratio_raw =\
    np.load("collision/kernel_data/Hall/Hall_Bott_R_ratio_table_raw.npy") 
def generate_E_col_grid_R_Hall_Bott_np(R_low, R_high, no_bins_10,
                                       radius_grid_in=None):
    """Generation of a discretized collision efficiency grid (radius based)
    
    When no radius grid is provided, the radius grid is generated
    between R_low and R_high with logarithmic spacing
    based on the number of bins per decade multiplication.
    R_low and R_high are both included in the radius_grid range interval
    but R_high itself might NOT be a value of the radius_grid
    (which is def by no_bins_10 and R_low).
    
    Collision efficiency table by Hall 1980, modified version
    by Seeßelberg 1996, used in Bott 1998 and Unterstrasser 2017.
    
    Parameters
    ----------
    R_low: float
        Smallest radius of the discretized grid (microns). Must be > 0
    R_high: float
        Largest radius of the discretized grid (microns)
    no_bins_10: int
        Number of bins per decade multiplication on logarithmic scale
    radius_grid_in: ndarray, dtype=float, optional
        1D radius grid. The collision eff. is evaluated at these points.
        When not provided, a logarithmic is automatically generated.
    
    Returns
    -------
    E_col_grid: ndarray, dtype=float
        1D array of collision efficiencies corresponding to the radius grid        
    radius_grid: ndarray, dtype=float
        1D array of radii (microns), where the coll. eff. where evaluated
    
    """
    
    no_R_table = Hall_Bott_R_col_raw.shape[0]
    no_rat_table = Hall_Bott_R_ratio_raw.shape[0]
    
    if radius_grid_in is None:
        bin_factor = 10**(1.0/no_bins_10)
        no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
        radius_grid = np.zeros( no_bins, dtype = np.float64 )
        
        radius_grid[0] = R_low
        for bin_n in range(1,no_bins):
            radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor   
    else:
        radius_grid = radius_grid_in
        no_bins = radius_grid_in.shape[0]
    
    Ecol_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    # R = larger radius
    # r = smaller radius
    for ind_R_out, R_col in enumerate(radius_grid):
        # get index of coll. radius lower boundary (floor)
        if R_col <= Hall_Bott_R_col_raw[0]:
            ind_R_table = -1
        elif R_col > Hall_Bott_R_col_raw[-1]:
            ind_R_table = no_R_table - 1 # = 14
        else:
            for ind_R_table_ in range(0,no_R_table-1):
                # drops with R = 300 mu (exact) go to another category
                if (Hall_Bott_R_col_raw[ind_R_table_] < R_col) \
                   and (R_col <= Hall_Bott_R_col_raw[ind_R_table_+1]):
                    ind_R_table = ind_R_table_
                    break
        for ind_r_out in range(ind_R_out+1):
            ratio = radius_grid[ind_r_out] / R_col
            # get index of radius ratio lower boundary (floor)
            # index of ratio should be at least one smaller
            # than the max. possible index. The case ratio = 1 is also covered
            # by the bilinear interpolation
            if ratio >= 1.0: ind_ratio = no_rat_table-2
            else:
                for ind_ratio_ in range(no_rat_table-1):
                    if (Hall_Bott_R_ratio_raw[ind_ratio_] <= ratio) \
                       and (ratio < Hall_Bott_R_ratio_raw[ind_ratio_+1]):
                        ind_ratio = ind_ratio_
                        break
            # bilinear interpolation:
            # linear interpol in ratio, if R >= R_max
            if ind_R_table == no_R_table - 1:
                q = ( ratio - Hall_Bott_R_ratio_raw[ind_ratio] ) \
                    / ( Hall_Bott_R_ratio_raw[ind_ratio+1] 
                        - Hall_Bott_R_ratio_raw[ind_ratio] )
                E_col = (1.-q)*Hall_Bott_E_col_raw_corr[ind_R_table,ind_ratio]\
                        + q * Hall_Bott_E_col_raw_corr[ind_R_table,ind_ratio+1]
                # note that Bott 1998 cuts the Efficiency at a maximum of 1.0
                # for the largest collector radii                
                if E_col <= 1.0:
                    Ecol_grid[ind_R_out, ind_r_out] = E_col
                else:    
                    Ecol_grid[ind_R_out, ind_r_out] = 1.0
            elif ind_R_table == -1:
                q = ( ratio - Hall_Bott_R_ratio_raw[ind_ratio] ) \
                    / ( Hall_Bott_R_ratio_raw[ind_ratio+1] 
                        - Hall_Bott_R_ratio_raw[ind_ratio] )
                Ecol_grid[ind_R_out, ind_r_out] =\
                        (1.-q) * Hall_Bott_E_col_raw_corr[0,ind_ratio]\
                        + q * Hall_Bott_E_col_raw_corr[0,ind_ratio+1] 
            else:
                p = ( R_col - Hall_Bott_R_col_raw[ind_R_table] ) \
                    / ( Hall_Bott_R_col_raw[ind_R_table+1] 
                        - Hall_Bott_R_col_raw[ind_R_table] )                
                q = ( ratio - Hall_Bott_R_ratio_raw[ind_ratio] ) \
                    / ( Hall_Bott_R_ratio_raw[ind_ratio+1] 
                        - Hall_Bott_R_ratio_raw[ind_ratio] )                
                Ecol_grid[ind_R_out, ind_r_out] = \
                    interpol_bilin(ind_R_table, ind_ratio, p, q,
                                   Hall_Bott_E_col_raw_corr)
            Ecol_grid[ind_r_out, ind_R_out] =\
                Ecol_grid[ind_R_out, ind_r_out]
    return Ecol_grid, radius_grid
generate_E_col_grid_R_Hall_Bott = \
    njit()(generate_E_col_grid_R_Hall_Bott_np)

def generate_E_col_grid_R_np(R_low, R_high, no_bins_10, kernel_name):
    """Generation of a discretized collision efficiency grid (radius based)
    
    The radius grid is generated
    between R_low and R_high with logarithmic spacing
    based on the number of bins per decade multiplication.
    R_low and R_high are both included in the radius_grid range interval
    but R_high itself might NOT be a value of the radius_grid
    (which is def by no_bins_10 and R_low).
    
    This function wraps the two generation methods for
    'Hall_Bott' and 'Long_Bott' kernels.
    
    Parameters
    ----------
    R_low: float
        Smallest radius of the discretized grid (microns). Must be > 0
    R_high: float
        Largest radius of the discretized grid (microns)
    no_bins_10: int
        Number of bins per decade multiplication on logarithmic scale
    kernel_name: str
        Choose parametrization method for the collision efficiency.
        Either 'Hall_Bott' or 'Long_Bott'.
    
    Returns
    -------
    E_col_grid: ndarray, dtype=float
        1D array of collision efficiencies corresponding to the radius grid        
    radius_grid: ndarray, dtype=float
        1D array of radii (microns), where the coll. eff. where evaluated
    
    """
    
    if kernel_name == "Hall_Bott":
        E_col_grid, radius_grid = \
            generate_E_col_grid_R_Hall_Bott(R_low, R_high, no_bins_10, None)
    
    elif kernel_name == "Long_Bott":
        E_col_grid, radius_grid = \
            generate_E_col_grid_R_Long_Bott(R_low, R_high, no_bins_10, None)
        
    return E_col_grid, radius_grid
generate_E_col_grid_R = \
    njit()(generate_E_col_grid_R_np)
    
def generate_and_save_E_col_grid_R(R_low, R_high, no_bins_10, kernel_name,
                                   save_dir):
    """Generation of a discretized collision efficiency grid (radius based)
    
    The radius grid is generated
    between R_low and R_high with logarithmic spacing
    based on the number of bins per decade multiplication.
    R_low and R_high are both included in the radius_grid range interval
    but R_high itself might NOT be a value of the radius_grid
    (which is def by no_bins_10 and R_low).    
    
    This function wraps the two generation methods for
    'Hall_Bott' and 'Long_Bott' kernels and writes the grids to hard disc
    in .npy data format.
    
    Parameters
    ----------
    R_low: float
        Smallest radius of the discretized grid (microns). Must be > 0
    R_high: float
        Largest radius of the discretized grid (microns)
    no_bins_10: int
        Number of bins per decade multiplication on logarithmic scale
    kernel_name: str
        Choose parametrization method for the collision efficiency.
        Either 'Hall_Bott' or 'Long_Bott'.
    save_dir: str
        Path to the directory, where the grid data shall be written.
        Provide in form '/path/to/directory/'
    
    Returns
    -------
    E_col_grid: ndarray, dtype=float
        1D array of collision efficiencies corresponding to the radius grid        
    radius_grid: ndarray, dtype=float
        1D array of radii (microns), where the coll. eff. where evaluated
    
    """
    
    E_col_grid, radius_grid = \
        generate_E_col_grid_R_np(R_low, R_high, no_bins_10, kernel_name)
    
    np.save(save_dir + "radius_grid_out.npy", radius_grid)
    np.save(save_dir + "E_col_grid.npy", E_col_grid)
    
    return E_col_grid, radius_grid

#%% GENERATION (DISCRETIZATION) OF COL. KERNEL GRIDS

def generate_kernel_grid_Long_Bott_np(R_low, R_high, no_bins_10,
                                      mass_density):
    """Generation of a discretized collision kernel grid (radius based)
    
    The radius grid is generated
    between R_low and R_high with logarithmic spacing
    based on the number of bins per decade multiplication.
    R_low and R_high are both included in the radius_grid range interval
    but R_high itself might NOT be a value of the radius_grid
    (which is def by no_bins_10 and R_low).
    
    Collision efficiency parametrization by Long 1974, modified version
    by Bott 1998, used in Unterstrasser 2017.
    Velocity parameterization (radius based) by Beard 1976 (s.a.)
    
    Parameters
    ----------
    R_low: float
        Smallest radius of the discretized grid (microns). Must be > 0
    R_high: float
        Largest radius of the discretized grid (microns)
    no_bins_10: int
        Number of bins per decade multiplication on logarithmic scale
    mass_density: float
        Droplet mass density (kg/m^3). The same density is assumed for all
        droplets.
    
    Returns
    -------
    kernel_grid: ndarray, dtype=float
        1D array of collision kernel values corresponding to the radius grid
        unit m^3/s
    vel_grid: ndarray, dtype=float
        1D array of velocity values corresponding to the radius grid (m/s)
    mass_grid: ndarray, dtype=float
        1D array of masses (1E-18 kg) corresponding to the radius grid
    radius_grid: ndarray, dtype=float
        1D array of radii (microns), where the kernel was evaluated
    
    """
    
    bin_factor = 10**(1.0/no_bins_10)
    
    no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
    
    radius_grid = np.zeros( no_bins, dtype = np.float64 )
    radius_grid[0] = R_low
    for bin_n in range(1,no_bins):
        radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor
    
    # generate velocity grid first at pos. of radius_grid
    vel_grid = compute_terminal_velocity_Beard_vec(radius_grid)
    
    kernel_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    for j in range(1,no_bins):
        R_j = radius_grid[j]
        v_j = vel_grid[j]
        for i in range(j):
            R_i = radius_grid[i]
            ### Kernel "LONG" provided by Bott (via Unterstrasser)
            if R_j <= 50.0:
                E_col = 4.5E-4 * R_j * R_j \
                        * ( 1.0 - 3.0 / ( max(3.0, R_i) + 1.0E-2 ) )
            else: E_col = 1.0
            
            kernel_grid[j,i] = 1.0E-12 * math.pi * (R_i + R_j) * (R_i + R_j) \
                               * E_col * abs(v_j - vel_grid[i]) 
            kernel_grid[i,j] = kernel_grid[j,i]

    c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
    mass_grid = c_radius_to_mass * radius_grid**3

    return kernel_grid, vel_grid, mass_grid, radius_grid
generate_kernel_grid_Long_Bott = njit()(generate_kernel_grid_Long_Bott_np)

def generate_kernel_grid_Hall_Bott_np(R_low, R_high, no_bins_10,
                                      mass_density):
    """Generation of a discretized collision kernel grid (radius based)
    
    The radius grid is generated
    between R_low and R_high with logarithmic spacing
    based on the number of bins per decade multiplication.
    R_low and R_high are both included in the radius_grid range interval
    but R_high itself might NOT be a value of the radius_grid
    (which is def by no_bins_10 and R_low).
    
    Collision efficiency table by Hall 1980, modified version
    by Seeßelberg 1996, used in Bott 1998 and Unterstrasser 2017.
    
    Parameters
    ----------
    R_low: float
        Smallest radius of the discretized grid (microns). Must be > 0
    R_high: float
        Largest radius of the discretized grid (microns)
    no_bins_10: int
        Number of bins per decade multiplication on logarithmic scale
    mass_density: float
        Droplet mass density (kg/m^3). The same density is assumed for all
        droplets.
    
    Returns
    -------
    kernel_grid: ndarray, dtype=float
        1D array of collision kernel values corresponding to the radius grid
        unit m^3/s
    vel_grid: ndarray, dtype=float
        1D array of velocity values corresponding to the radius grid (m/s)
    mass_grid: ndarray, dtype=float
        1D array of masses (1E-18 kg) corresponding to the radius grid
    radius_grid: ndarray, dtype=float
        1D array of radii (microns), where the kernel was evaluated
    
    """
    
    bin_factor = 10**(1.0/no_bins_10)
    
    no_bins = int(math.ceil( no_bins_10 * math.log10(R_high/R_low) ) ) + 1
    
    radius_grid = np.zeros( no_bins, dtype = np.float64 )
    radius_grid[0] = R_low
    for bin_n in range(1,no_bins):
        radius_grid[bin_n] = radius_grid[bin_n-1] * bin_factor
    
    # generate velocity grid first at pos. of radius_grid
    vel_grid = compute_terminal_velocity_Beard_vec(radius_grid)
    
    kernel_grid = np.zeros( (no_bins, no_bins), dtype = np.float64 )
    
    Ecol_grid, _ = generate_E_col_grid_R_Hall_Bott(R_low, R_high, no_bins_10,
                                                   radius_grid_in=radius_grid)
    
    for j in range(1,no_bins):
        R_j = radius_grid[j]
        v_j = vel_grid[j]
        for i in range(j):
            R_i = radius_grid[i]
            ### Kernel "Hall" provided by Bott (via Unterstrasser)
            E_col = Ecol_grid[j,i]            
            
            kernel_grid[j,i] = 1.0E-12 * math.pi * (R_i + R_j) * (R_i + R_j) \
                               * E_col * abs(v_j - vel_grid[i]) 
            kernel_grid[i,j] = kernel_grid[j,i]

    c_radius_to_mass = 4.0E-18 * math.pi * mass_density / 3.0
    mass_grid = c_radius_to_mass * radius_grid**3

    return kernel_grid, vel_grid, mass_grid, radius_grid
generate_kernel_grid_Hall_Bott = njit()(generate_kernel_grid_Hall_Bott_np)

def generate_and_save_kernel_grid(R_low, R_high, no_bins_10,
                                  mass_density, kernel_name, save_dir):
    """Generation of a discretized collision kernel grid (radius based)
    
    The radius grid is generated
    between R_low and R_high with logarithmic spacing
    based on the number of bins per decade multiplication.
    R_low and R_high are both included in the radius_grid range interval
    but R_high itself might NOT be a value of the radius_grid
    (which is def by no_bins_10 and R_low).

    This is a wrapper for the two generation methods
    'Hall_Bott' and 'Long_Bott' (s.a.).
    
    Parameters
    ----------
    R_low: float
        Smallest radius of the discretized grid (microns). Must be > 0
    R_high: float
        Largest radius of the discretized grid (microns)
    no_bins_10: int
        Number of bins per decade multiplication on logarithmic scale
    mass_density: float
        Droplet mass density (kg/m^3). The same density is assumed for all
        droplets.
    kernel_name: str
        Choose parametrization method for the collision efficiency.
        Either 'Hall_Bott' or 'Long_Bott'.
    save_dir: str
        Path to the directory, where the grid data shall be written.
        Provide in form '/path/to/directory/'        
    
    Returns
    -------
    kernel_grid: ndarray, dtype=float
        1D array of collision kernel values corresponding to the radius grid
        unit m^3/s
    vel_grid: ndarray, dtype=float
        1D array of velocity values corresponding to the radius grid (m/s)
    mass_grid: ndarray, dtype=float
        1D array of masses (1E-18 kg) corresponding to the radius grid
    radius_grid: ndarray, dtype=float
        1D array of radii (microns), where the kernel was evaluated
    
    """
    
    if kernel_name == "Long_Bott":
        kernel_grid, vel_grid, mass_grid, radius_grid = \
            generate_kernel_grid_Long_Bott(R_low, R_high, no_bins_10,
                                           mass_density)
    if kernel_name == "Hall_Bott":
        kernel_grid, vel_grid, mass_grid, radius_grid = \
            generate_kernel_grid_Hall_Bott(R_low, R_high, no_bins_10,
                                           mass_density)
    
    np.save(save_dir + "radius_grid_out.npy", radius_grid)
    np.save(save_dir + "mass_grid_out.npy", mass_grid)
    np.save(save_dir + "kernel_grid.npy", kernel_grid)
    np.save(save_dir + "velocity_grid.npy", vel_grid)        
    
    return kernel_grid, vel_grid, mass_grid, radius_grid
