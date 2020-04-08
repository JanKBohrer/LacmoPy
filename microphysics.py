#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

MICROPHYSICS:
PARTICLE FORCES, CONDENSATIONAL MASS GROWTH, SATURATATION ADJUSTMENT

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

#%% MODULE IMPORTS

import math
import numpy as np
from scipy.optimize import fminbound
from scipy.optimize import brentq
from numba import njit, vectorize

import constants as c
import material_properties as mat
par_sol_dens_NaCl = mat.par_sol_dens_NaCl
par_rho_AS = mat.par_rho_AS
par_wat_act_AS = mat.par_wat_act_AS
par_wat_act_NaCl = mat.par_wat_act_NaCl
par_sigma_AS = mat.par_sigma_AS
par_sigma_NaCl = mat.par_sigma_NaCl
                       
from algebra import compute_polynom

#%% CONVERSIONS

@njit()
def compute_mass_from_radius(radius, density):
    """Compute mass of a rigid sphere, given radius and density
    
    Parameters
    ----------
    radius: float
        Sphere radius (microns)
    density: float
        Mass density (kg/m^3)
    
    Returns
    -------
        float
        Mass (1E-18 kg)
    
    """
    
    return c.four_pi_over_three * density * radius * radius * radius 

@vectorize("float64(float64,float64)")
def compute_mass_from_radius_vec(radius, density):
    """Compute mass of a rigid sphere, given radius and density
    
    Vectorized version with Numba
    
    Parameters
    ----------
    radius: float
        Sphere radius (microns)
    density: float
        Mass density (kg/m^3)
    
    Returns
    -------
        float
        Mass (1E-18 kg)
    
    """
    
    return c.four_pi_over_three * density * radius * radius * radius 

@njit()
def compute_radius_from_mass(mass, density):
    """Compute radius of a rigid sphere, given mass and density
    
    Parameters
    ----------
    mass: float
        Sphere mass (1E-18 kg)
    density: float
        Mass density (kg/m^3)
    
    Returns
    -------
        float
        Radius (microns)
    
    """
    
    return ( c.four_pi_over_three_inv * mass / density ) ** (c.one_third)

@vectorize("float64(float64,float64)")
def compute_radius_from_mass_vec(mass, density):
    """Compute radius of a rigid sphere, given mass and density
    
    Vectorized version with Numba
    
    Parameters
    ----------
    mass: float
        Sphere mass (1E-18 kg)
    density: float
        Mass density (kg/m^3)
    
    Returns
    -------
        float
        Radius (microns)
    
    """
    
    return ( c.four_pi_over_three_inv * mass / density ) ** (c.one_third)

@njit()
def compute_mass_fraction_from_molality(molality, molecular_weight):
    """Compute mass fraction from molality for single solute
    
    Molality = n_solute / m_solvent (mol/kg)
    Mass fraction = m_solute / m_solution
    For single solute = m_solute / (m_solute + m_solvent)
    
    Vectorized version with Numba
    
    Parameters
    ----------
    molality: float
        Molality = n_solute / m_solvent (mol/kg)
    molecular_weight: float
        Molecular weight (kg/mol)
    
    Returns
    -------
        float
        Mass fraction m_solute / m_solution (no unit)
    
    """    
    
    return 1.0 / ( 1.0 + 1.0 / (molality * molecular_weight) )

@njit()
def compute_molality_from_mass_fraction(mass_fraction, molecular_weight):
    """Compute mass fraction from molality for single solute
    
    Molality = n_solute / m_solvent (mol/kg)
    Mass fraction = m_solute / m_solution
    For single solute = m_solute / (m_solute + m_solvent)
    
    Vectorized version with Numba
    
    Parameters
    ----------
    mass_fraction: float
        Mass fraction m_solute / m_solution (no unit)
    molecular_weight: float
        Molecular weight (kg/mol)

    Returns
    -------
    molality: float
        Molality = n_solute / m_solvent (mol/kg)
    
    """  
    
    return mass_fraction / ( (1. - mass_fraction) * molecular_weight )

### FOR CONVENIENCE: RADIUS, MASS FRACTION AND DENSITY 
        
@njit()
def compute_R_p_w_s_rho_p_NaCl(m_w, m_s, T_p):
    """Compute particle radius, mass fraction and density for sodium chloride
    
    Parameters
    ----------
    m_w: float
        Particle water mass (1E-18 kg)
    m_s: float
        Particle solute mass (1E-18 kg)
    T_p: float
        Particle temperature (K)

    Returns
    -------
    R_p: float
        Particle radius (microns)
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    rho_p: float
        Mass density (kg/m^3)
    
    """
    
    m_p = m_w + m_s
    w_s = m_s / m_p
    rho_p = mat.compute_density_NaCl_solution(w_s, T_p)
    return compute_radius_from_mass(m_p, rho_p), w_s, rho_p
    
@njit()
def compute_R_p_w_s_rho_p_AS(m_w, m_s, T_p):
    """Compute particle radius, mass fraction and density for ammon. sulfate
    
    Parameters
    ----------
    m_w: float
        Particle water mass (1E-18 kg)
    m_s: float
        Particle solute mass (1E-18 kg)
    T_p: float
        Particle temperature (K)

    Returns
    -------
    R_p: float
        Particle radius (microns)
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    rho_p: float
        Mass density (kg/m^3)
    
    """
    
    m_p = m_w + m_s
    w_s = m_s / m_p
    rho_p = mat.compute_density_AS_solution(w_s, T_p)
    return compute_radius_from_mass(m_p, rho_p), w_s, rho_p

@njit()
def compute_R_p_w_s_rho_p(m_w, m_s, T_p, solute_type):
    """Compute particle radius, mass fraction and density for single solute
    
    Parameters
    ----------
    m_w: float
        Particle water mass (1E-18 kg)
    m_s: float
        Particle solute mass (1E-18 kg)
    T_p: float
        Particle temperature (K)
    solute_type: str
        Solute material
        Either 'AS' (ammonium sulfate) or 'NaCl' (sodium chloride)

    Returns
    -------
    R_p: float
        Particle radius (microns)
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    rho_p: float
        Mass density (kg/m^3)
    
    """
    
    if solute_type == 'AS':
        return compute_R_p_w_s_rho_p_AS(m_w, m_s, T_p)
    elif solute_type == 'NaCl':
        return compute_R_p_w_s_rho_p_NaCl(m_w, m_s, T_p)

def compute_particle_radius_from_ws_T_ms_NaCl(mass_fraction_solute,
                                              temperature, dry_mass):
    """Compute particle radius, given mass fraction, temperature and dry mass
    
    Parameters
    ----------
    mass_fraction_solute: float
        Mass fraction of solute m_solute / m_solution (no unit)
    temperature: float
        Particle temperature (K)
    dry_mass: float
        Particle dry mass (1E-18 kg)
    
    Returns
    -------
        float
        Radius (microns)
    
    """
    
    Vp = dry_mass\
        / ( mass_fraction_solute \
           * mat.compute_density_NaCl_solution(mass_fraction_solute,
                                               temperature) )
    return (c.four_pi_over_three_inv * Vp)**(c.one_third)

#%% FORCES
           
@njit()
def compute_particle_reynolds_number(radius, velocity_dev, fluid_density,
                                     fluid_viscosity):
    """Compute particle Reynolds number
    
    As given in
    Crowe et al. 2011: Multiphase Flows with Droplets and Particles,
    CRC Press, 2nd Edition (2011)
    
    Parameters
    ----------
    radius: float
        Particle radius (microns)
    velocity_dev: float
        Absolute difference of the velocities of particle and
        surrounding fluid: |v_p - u_fluid| (m/s)
    fluid_density: float
        Fluid mass density (kg/m^3)
    fluid_viscosity: float
        Dynamic viscosity of the fluid (Pa s)
    
    Returns
    -------
        float
        Particle Reynolds number (-)
    
    """
    
    return 2.0E-6 * fluid_density * radius * velocity_dev / fluid_viscosity    

#%% EQUILIBRIUM SATURATION - KELVIN RAOULT

@njit()
def compute_kelvin_argument(R_p, T_p, rho_p, sigma_p):
    """Compute the argument of the exponential in the Kelvin term 
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation: 
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    R_p: float
        Particle radius (microns)
    T_p: float
        Particle temperature (K)
    rho_p: float
        Particle density (kg/m^3)
    sigma_p: float
        Particle surface tension (N/m)
    
    Returns
    -------
        float
        Argument of the exponential in the Kelvin-Term (s.a.) (no unit)
    
    """
    
    return 2.0E6 * sigma_p \
           / ( c.specific_gas_constant_water_vapor * T_p * rho_p * R_p )

@njit()
def compute_kelvin_term(R_p, T_p, rho_p, sigma_p):
    """Compute the Kelvin term in the equilibrium saturation
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    R_p: float
        Particle radius (microns)
    T_p: float
        Particle temperature (K)
    rho_p: float
        Particle density (kg/m^3)
    sigma_p: float
        Particle surface tension (N/m)
    
    Returns
    -------
        float
        Kelvin-Term of the Köhler-equation (s.a.) (no unit)
    
    """
    
    return np.exp(compute_kelvin_argument(R_p, T_p, rho_p, sigma_p))

@njit()
def compute_kelvin_term_mf(mass_fraction_solute,
                           temperature,
                           mass_solute,
                           mass_density_particle,
                           surface_tension ):
    """Compute the Kelvin term in the equil. saturation (from mass fract.)
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    mass_fraction_solute: float
        Mass fraction of solute m_solute / m_solution (no unit)
    temperature: float
        Particle temperature (K)
    mass_solute: float
        Particle solute mass (1E-18 kg)
    mass_density_particle: float
        Particle mass densit (kg/m^3)
    rho_p: float
        Particle density (kg/m^3)
    surface_tension: float
        Particle surface tension (N/m)
    
    Returns
    -------
        float
        Kelvin-Term of the Köhler-equation (s.a.) (no unit)
    
    """
    
    return np.exp( 2.0 * surface_tension * 1.0E6\
                   * (mass_fraction_solute / mass_solute)**(c.one_third)
                   / ( c.specific_gas_constant_water_vapor
                       * temperature
                       * mass_density_particle**(0.66666667)
                       * c.volume_to_radius) )

@vectorize(
    "float64(float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_NaCl(w_s, R_p, T_p, rho_p, sigma_p):
    """Compute the Köhler equilibrium saturation for sodium chloride
    
    Vectorized with Numba
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p)
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    R_p: float
        Particle radius (microns)
    T_p: float
        Particle temperature (K)
    rho_p: float
        Particle density (kg/m^3)
    sigma_p: float
        Particle surface tension (N/m)
    
    Returns
    -------
        float
        Equilibrium saturation by Köhler-equation (s.a.) (no unit)
    
    """
    
    return mat.compute_water_activity_NaCl(w_s)\
           * compute_kelvin_term(R_p, T_p, rho_p, sigma_p)

@vectorize(
    "float64(float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_AS(w_s, R_p, T_p, rho_p, sigma_p):
    """Compute the Köhler equilibrium saturation for ammonium sulfate
    
    Vectorized with Numba
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    R_p: float
        Particle radius (microns)
    T_p: float
        Particle temperature (K)
    rho_p: float
        Particle density (kg/m^3)
    sigma_p: float
        Particle surface tension (N/m)
    
    Returns
    -------
        float
        Equilibrium saturation by Köhler-equation (s.a.) (no unit)
    
    """
    
    return mat.compute_water_activity_AS(w_s)\
           * compute_kelvin_term(R_p, T_p, rho_p, sigma_p)

# from mass fraction mf
@njit()
def compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_s):
    """Compute the Köhler equilibrium saturation for sodium chloride
    
    Using the 'mass_fraction' version of 'compute_kelvin_term_mf'
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    T_p: float
        Particle temperature (K)
    m_s: float
        Particle solute mass (1E-18 kg)
    
    Returns
    -------
        float
        Equilibrium saturation by Köhler-equation (s.a.) (no unit)
    
    """
    
    rho_p = mat.compute_density_NaCl_solution(w_s, T_p)
    sigma_p = mat.compute_surface_tension_NaCl(w_s, T_p)
    return mat.compute_water_activity_NaCl(w_s) \
           * compute_kelvin_term_mf(w_s, T_p, m_s, rho_p, sigma_p)

# from mass fraction mf
@njit()
def compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s):
    """Compute the Köhler equilibrium saturation for ammonium sulfate
    
    Using the 'mass_fraction' version of 'compute_kelvin_term_mf'
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    T_p: float
        Particle temperature (K)
    m_s: float
        Particle solute mass (1E-18 kg)
    
    Returns
    -------
        float
        Equilibrium saturation by Köhler-equation (s.a.) (no unit)
    
    """
    
    rho_p = mat.compute_density_AS_solution(w_s, T_p)
    sigma_p = mat.compute_surface_tension_AS(w_s, T_p)
    return mat.compute_water_activity_AS(w_s) \
           * compute_kelvin_term_mf(w_s, T_p, m_s, rho_p, sigma_p)

#%% INITIAL MASS FRACTION

### INITIAL MASS FRACTION SODIUM CHLORIDE
@njit()
def compute_equilibrium_saturation_negative_NaCl_mf(w_s, T_p, m_s):
    """Compute negative Köhler equilibrium saturation for sodium chloride
    
    This wrapper function is used to find the maximum of the
    equil. sat. wrt. w_s.
    Using the 'mass_fraction' version of 'compute_kelvin_term_mf'
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    T_p: float
        Particle temperature (K)
    m_s: float
        Particle solute mass (1E-18 kg)
    
    Returns
    -------
        float
        Negative equilibrium saturation by Köhler-equation (s.a.) (no unit)
    
    """
    
    return -compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_s)

def compute_equilibrium_saturation_minus_S_amb_NaCl_mf(w_s, T_p, m_s, S_amb):
    """Compute equilibrium saturation minus ambient sat. (sodium chloride)
    
    This wrapper function is used to find the root of S_eq-S_amb wrt. w_s.
    Using the 'mass_fraction' version of 'compute_kelvin_term_mf'.
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    T_p: float
        Particle temperature (K)
    m_s: float
        Particle solute mass (1E-18 kg)
    S_amb: float
        Ambient saturation S (no unit)
    
    Returns
    -------
        float
        Difference of equilibrium saturation and ambient saturation
        by Köhler-equation (s.a.) (no unit)
    
    """
    
    return -S_amb + compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_s)

@vectorize( "float64(float64,float64,float64)", forceobj=True )
def compute_initial_mass_fraction_solute_m_s_NaCl(m_s, S_amb, T_amb):
    """Compute initial particle mass fraction in equilibrium with moist air
    
    Solute material = sodium chloride
    
    The particle solute mass fraction w_s is chosen such that the
    equilibrium saturation S_eq(w_s) is equal to the ambient saturation.
    If the ambient saturation is larger than the maximum of S_eq(w_s),
    then w_s is set to the value corresponding to the maximum of S_eq
    (which is the activation mass fraction).
    The mass fraction w_s is implemented with an upper bound w_s_max,
    which marks the interval, where the parametrization of the
    surface tension is valid.
    
    1. S_min = S_eq (w_s_max, m_s)
    2. if S_a <= S_min : w_s = w_s_max
    3. else (S_a > S_min): S_act, w_s_act = max( S(w_s, m_s) )
    4a. w_s_act = 1.00001 * w_s_act (numerical stability ->
    want to be on branch of high w_s <-> low R_p for cont. fct. S(w_s) )
    4b. S_act = S(w_s_act)   ( < S_act_real! )
    5. if S_a > S_act : w_s_init = w_s_act
    6. else (S_a <= S_act) : calc w_s_init from S( w_s_init ) - S_a = 0
    Check for convergence at every stage
    For sodium chloride: fix an upper bound for w_s: w_s_max = 0.45
    w_s cannot get larger than that.
    The border is chosen, because the approximation of sigma_NaCl(w_s)
    is only given for 0 < w_s < 0.45
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    m_s: float
        Particle solute mass (1E-18 kg)
    S_amb: float
        Ambient saturation S (no unit)
    T_amb: float
        Ambient temperature (K)
    
    Returns
    -------
    w_s_init: float
        Initial particle solute mass fraction, 
        Difference of equilibrium saturation and ambient saturation
        by Köhler-equation (s.a.) (no unit)
    
    """
    
    # 1.
    S_min = compute_equilibrium_saturation_NaCl_mf(mat.w_s_max_NaCl,
                                                    T_amb, m_s)
    # 2.
    if S_amb <= S_min:
        w_s_init = mat.w_s_max_NaCl
    else:
        # 3.
        w_s_act, S_act, flag, nofc  = \
            fminbound(compute_equilibrium_saturation_negative_NaCl_mf,
                      x1=1E-8, x2=mat.w_s_max_NaCl,
                      args=(T_amb, m_s),
                      xtol = 1.0E-12, full_output=True )
        # 4.
        # increase w_s_act slightly to avoid numerical problems
        # in solving with brentq() below
        if flag == 0:
            w_s_act *= 1.000001
        # set w_s_act (i.e. the min bound for brentq() solve below )
        # to deliqu. mass fraction if fminbound does not converge
        else:
            w_s_act = mat.compute_solubility_NaCl(T_amb)
        S_act = compute_equilibrium_saturation_NaCl_mf(w_s_act,
                                                       T_amb,
                                                       m_s)
        # 5.
        if S_amb > S_act:
            w_s_init = w_s_act
        else:
            # 6.
            solve_result = \
                brentq(
                    compute_equilibrium_saturation_minus_S_amb_NaCl_mf,
                    w_s_act,
                    mat.w_s_max_NaCl,
                    (T_amb, m_s, S_amb),
                    xtol = 1e-15,
                    full_output=True)
            if solve_result[1].converged:
                w_s_init = solve_result[0]
            else:
                w_s_init = w_s_act        
    
    return w_s_init

### INITIAL MASS FRACTION AMMONIUM SULFATE
@njit()
def compute_equilibrium_saturation_negative_AS_mf(w_s, T_p, m_s):
    """Compute negative Köhler equilibrium saturation for ammonium sulfate
    
    This wrapper function is used to find the maximum of the
    equil. sat. wrt. w_s.
    Using the 'mass_fraction' version of 'compute_kelvin_term_mf'
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    T_p: float
        Particle temperature (K)
    m_s: float
        Particle solute mass (1E-18 kg)
    
    Returns
    -------
        float
        Negative equilibrium saturation by Köhler-equation (s.a.) (no unit)
    
    """
    
    return -compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s)

def compute_equilibrium_saturation_minus_S_amb_AS_mf(w_s, T_p, m_s, S_amb):
    """Compute equilibrium saturation minus ambient sat. (ammon. sulfate)
    
    This wrapper function is used to find the root of S_eq-S_amb wrt. w_s.
    Using the 'mass_fraction' version of 'compute_kelvin_term_mf'.
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    T_p: float
        Particle temperature (K)
    m_s: float
        Particle solute mass (1E-18 kg)
    S_amb: float
        Ambient saturation S (no unit)
    
    Returns
    -------
        float
        Difference of equilibrium saturation and ambient saturation
        by Köhler-equation (s.a.) (no unit)
    
    """
                                                     
    return -S_amb \
           + compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s)
                 
@vectorize( "float64(float64,float64,float64)", forceobj=True )
def compute_initial_mass_fraction_solute_m_s_AS(m_s, S_amb, T_amb):
    """Compute initial particle mass fraction in equilibrium with moist air
    
    Solute material = ammonium sulfate
    
    The particle solute mass fraction w_s is chosen such that the
    equilibrium saturation S_eq(w_s) is equal to the ambient saturation.
    If the ambient saturation is larger than the maximum of S_eq(w_s),
    then w_s is set to the value corresponding to the maximum of S_eq
    (which is the activation mass fraction).
    The mass fraction w_s is implemented with an upper bound w_s_max,
    which marks the interval, where the parametrization of the
    surface tension is valid.
    
    1. S_min = S_eq (w_s_max, m_s)
    2. if S_a <= S_min : w_s = w_s_max
    3. else (S_a > S_min): S_act, w_s_act = max( S(w_s, m_s) )
    4a. w_s_act = 1.00001 * w_s_act (numerical stability ->
    want to be on branch of high w_s <-> low R_p for cont. fct. S(w_s) )
    4b. S_act = S(w_s_act)   ( < S_act_real! )
    5. if S_a > S_act : w_s_init = w_s_act
    6. else (S_a <= S_act) : calc w_s_init from S( w_s_init ) - S_a = 0
    Check for convergence at every stage
    For ammonium sulfate: fix an upper bound for w_s: w_s_max = 0.78
    w_s cannot get larger than that.
    The border is chosen, because the approximation of sigma_AS(w_s)
    is only given for 0 < w_s < 0.78
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p )
    
    Parameters
    ----------
    m_s: float
        Particle solute mass (1E-18 kg)
    S_amb: float
        Ambient saturation S (no unit)
    T_amb: float
        Ambient temperature (K)
    
    Returns
    -------
    w_s_init: float
        Initial particle solute mass fraction, 
        Difference of equilibrium saturation and ambient saturation
        by Köhler-equation (s.a.) (no unit)
    
    """
    
    # 1.
    S_effl = compute_equilibrium_saturation_AS_mf(mat.w_s_max_AS,
                                                  T_amb, m_s)
    # 2.
    # np.where(S_amb <= S_effl, w_s_init = w_s_effl,)
    if S_amb <= S_effl:
        w_s_init = mat.w_s_max_AS
    else:
        # 3.
        w_s_act, S_act, flag, nofc  = \
            fminbound(compute_equilibrium_saturation_negative_AS_mf,
                      x1=1E-8, x2=mat.w_s_max_AS,
                      args=(T_amb, m_s),
                      xtol = 1.0E-12, full_output=True )
        # 4.
        # increase w_s_act slightly to avoid numerical problems
        # in solving with brentq() below
        if flag == 0:
            w_s_act *= 1.000001
        # set w_s_act (i.e. the min bound for brentq() solve below )
        # to deliqu. mass fraction if fminbound does not converge
        else:
            w_s_act = mat.compute_solubility_AS(T_amb)
        # update S_act to S_act* < S_act (right branch of S_eq vs w_s curve)
        S_act = compute_equilibrium_saturation_AS_mf(w_s_act,
                                                     T_amb, m_s)
        # 5.
        if S_amb > S_act:
            w_s_init = w_s_act
        else:
            # 6.
            solve_result = \
                brentq(
                    compute_equilibrium_saturation_minus_S_amb_AS_mf,
                    w_s_act,
                    mat.w_s_max_AS,
                    (T_amb, m_s, S_amb),
                    xtol = 1e-15,
                    full_output=True)
            if solve_result[1].converged:
                w_s_init = solve_result[0]
            else:
                w_s_init = w_s_act        
    return w_s_init

#%% CONDENSATION MASS RATE (= "gamma")
    
### linearization of the size correction functions of Fukuta 1970
accommodation_coeff = 1.0
# adiabatic index = 1.4 = 7/5 -> 1/1.4 = 5/7 = 0.7142857142857143
adiabatic_index_inv = 0.7142857142857143
T_alpha_0 = 289. # K
c_alpha_1 = 1.0E6 * math.sqrt(2.0 * np.pi * c.specific_gas_constant_air_dry 
                              * T_alpha_0 )\
                    / ( accommodation_coeff
                        * ( c.specific_heat_capacity_air_dry_NTP
                            * adiabatic_index_inv\
                            + 0.5 * c.specific_gas_constant_air_dry ) )
c_alpha_2 = 0.5E6\
            * math.sqrt(2.0 * np.pi * c.specific_gas_constant_air_dry
                        / T_alpha_0 )\
              / ( accommodation_coeff
                 * (c.specific_heat_capacity_air_dry_NTP * adiabatic_index_inv\
                    + 0.5 * c.specific_gas_constant_air_dry) )

@vectorize("float64(float64,float64,float64)")
def compute_l_alpha_lin(T_amb, p_amb, K):
    """Heat-term size correction for the condensation mass rate (linearized)
    
    Particle radius size correction (Knudsen flow) for the heat-term
    in the condensation mass rate equation by
    Fukuta and Walter 1970, J. Atmos. Sci. 27: 1160
    
    We use accommodation_coefficient = 1.0 after recommendation in
    Pruppacher 1997, adiabatic_index = 7/5 and
    c_v_air = c_v_air_dry_NTP (normal temperature and pressure)
    
    Parameters
    ----------
    T_amb: float
        Ambient temperature (K)
    p_amb: float
        Ambient pressure (Pa)
    K: float
        Thermal conductivity of air (W/(m K))
    
    Returns
    -------
        float
        Linearized particle radius size correction for the
        heat-term of the condensation mass rate (microns)
    
    """
    
    return ( c_alpha_1 + c_alpha_2 * (T_amb - T_alpha_0) ) * K / p_amb

### linearization of the size correction functions of Fukuta 1970
condensation_coeff = 0.0415
c_beta_1 = 1.0E6 * math.sqrt( 2.0 * np.pi * c.molar_mass_water\
                              / ( c.universal_gas_constant * T_alpha_0 ) )\
                   / condensation_coeff
c_beta_2 = -0.5 / T_alpha_0
@vectorize("float64(float64,float64)")
def compute_l_beta_lin(T_amb, D_v):
    """Diffusion-term size correction for the condens. mass rate (linearized)
    
    Particle radius size correction (Knudsen flow) for the diffusion-term
    in the condensation mass rate equation by
    Fukuta and Walter 1970, J. Atmos. Sci. 27: 1160
    
    We use condensation_coefficient = 0.0415, after recommendation in
    Pruppacher 1997 
    
    Parameters
    ----------
    T_amb: float
        Ambient temperature (K)
    D_v: float
        Diffusion coefficent of water vapor in air (m^2/s)
    
    Returns
    -------
        float
        Linearized particle radius size correction for the
        diffusion-term of the condensation mass rate (microns)
    
    """
    
    return c_beta_1 * ( 1.0 + c_beta_2 * (T_amb - T_alpha_0) ) * D_v

@vectorize(
"float64(\
float64, float64, float64, float64, float64, float64, float64, float64)")
def compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v):
    """Compute denominator of the condensation mass rate
    
    Condensation mass rate equation by
    Fukuta and Walter 1970, J. Atmos. Sci. 27: 1160
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p)
    
    Parameters
    ----------
    R_p: float
        Particle radius (microns)
    S_eq: float
        Equilibrium saturation (s.a.) (no unit)
    T_amb: float
        Ambient temperature (K)
    p_amb: float
        Ambient pressure (Pa)
    e_s_amb: float
        Ambient saturation pressure (vapor-liquid)
    L_v: float
        Heat of vaporization (J/kg)
    K: float
        Thermal conductivity of air (W/(m K))
    D_v: float
        Diffusion coefficent of water vapor in air (m^2/s)
    
    Returns
    -------
        float
        Denominator of the condensation mass rate ((m^2 * s) / kg ) (SI)
    
    """
    
    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb)
    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
    l_alpha = compute_l_alpha_lin(T_amb, p_amb, K)
    l_beta = compute_l_beta_lin(T_amb, D_v)
    return 1.0E-6 * ( c1 * S_eq * (R_p + l_alpha) + c2 * (R_p + l_beta) )

# MASS RATE NACL
# in fg/s = 1.0E-18 kg/s
# function was compared to compute_mass_rate_NaCl_vH:
# for small m_w, there are deviations up to 10 %, as expected
# due to different paramentrizations of water activity.
# for large m_w, the fcts converge           
@vectorize(
"float64(\
float64, float64, float64, float64, float64, float64,\
float64, float64, float64, float64, float64, float64)")
def compute_mass_rate_NaCl(w_s, R_p, T_p, rho_p,
                           T_amb, p_amb, S_amb, e_s_amb,
                           L_v, K, D_v, sigma_p):
    """Compute the condensation mass rate (sodium chloride)
    
    Condensation mass rate equation by
    Fukuta and Walter 1970, J. Atmos. Sci. 27: 1160
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p)
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    R_p: float
        Particle radius (microns)
    T_p: float
        Particle temperature (K)
    rho_p: float
        Mass density (kg/m^3)
    T_amb: float
        Ambient temperature (K)
    p_amb: float
        Ambient pressure (Pa)
    S_amb: float
        Ambient saturation S (no unit)
    e_s_amb: float
        Ambient saturation pressure (vapor-liquid)
    L_v: float
        Heat of vaporization (J/kg)
    K: float
        Thermal conductivity of air (W/(m K))
    D_v: float
        Diffusion coefficent of water vapor in air (m^2/s)
    sigma_p: float
        Particle surface tension (N/m)
    
    Returns
    -------
        float
        Condensation mass rate (s.a.) (1E-18 kg / s)
    
    """
    
    S_eq = compute_equilibrium_saturation_NaCl(w_s, R_p,
                                          T_p, rho_p, sigma_p)
    return 4.0E6 * np.pi * R_p * R_p * (S_amb - S_eq)\
           / compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v) 

# MASS RATE AMMON SULF
# in fg/s = 1.0E-18 kg/s
@vectorize(
"float64(\
float64, float64, float64, float64, float64, float64,\
float64, float64, float64, float64, float64, float64)")
def compute_mass_rate_AS(w_s, R_p, T_p, rho_p,
                         T_amb, p_amb, S_amb, e_s_amb,
                         L_v, K, D_v, sigma_p):
    """Compute the condensation mass rate (ammonium sulfate)
    
    Condensation mass rate equation by
    Fukuta and Walter 1970, J. Atmos. Sci. 27: 1160
    
    Saturation S = e / e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p)
    
    Parameters
    ----------
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    R_p: float
        Particle radius (microns)
    T_p: float
        Particle temperature (K)
    rho_p: float
        Mass density (kg/m^3)
    T_amb: float
        Ambient temperature (K)
    p_amb: float
        Ambient pressure (Pa)
    S_amb: float
        Ambient saturation S (no unit)
    e_s_amb: float
        Ambient saturation pressure (vapor-liquid)
    L_v: float
        Heat of vaporization (J/kg)
    K: float
        Thermal conductivity of air (W/(m K))
    D_v: float
        Diffusion coefficent of water vapor in air (m^2/s)
    sigma_p: float
        Particle surface tension (N/m)
    
    Returns
    -------
        float
        Condensation mass rate (s.a.) (1E-18 kg / s)
    
    """
    
    S_eq = compute_equilibrium_saturation_AS(w_s, R_p,
                                          T_p, rho_p, sigma_p)
    return 4.0E6 * np.pi * R_p * R_p * (S_amb - S_eq)\
           / compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v)  

par_wat_act_deriv_NaCl = np.copy(par_wat_act_NaCl[:-1]) \
                       * np.arange(1,len(par_wat_act_NaCl))[::-1]
def compute_mass_rate_and_derivative_NaCl_np(m_w, m_s, w_s, R_p, T_p, rho_p,
                                             T_amb, p_amb, S_amb, e_s_amb,
                                             L_v, K, D_v, sigma_p):
    """Compute the condensation mass rate and its mass derivative
    
    Material = sodium chloride
    
    Condensation mass rate equation by
    Fukuta and Walter 1970, J. Atmos. Sci. 27: 1160
    
    Let gamma be the mass rate. The function returns gamma
    and d gamma / d m_w (derivative by water mass)
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p)
    
    Parameters
    ----------
    m_w: float
        Particle water mass (1E-18 kg)
    m_s: float
        Particle solute mass (1E-18 kg)
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    R_p: float
        Particle radius (microns)
    T_p: float
        Particle temperature (K)
    rho_p: float
        Mass density (kg/m^3)
    T_amb: float
        Ambient temperature (K)
    p_amb: float
        Ambient pressure (Pa)
    S_amb: float
        Ambient saturation S (no unit)
    e_s_amb: float
        Ambient saturation pressure (vapor-liquid)
    L_v: float
        Heat of vaporization (J/kg)
    K: float
        Thermal conductivity of air (W/(m K))
    D_v: float
        Diffusion coefficent of water vapor in air (m^2/s)
    sigma_p: float
        Particle surface tension (N/m)
    
    Returns
    -------
    gamma: float
        Condensation mass rate (s.a.) (1E-18 kg / s)
    dgamma_dm: float
        Derivative of the mass rate by the water mass (1/s) (SI)
    
    """
    
    # thermal size correction in SI
    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
    # diffusive size correction in SI
    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
       
    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
    # different to the AS function due to different density parametrization
    drho_dm_over_rho = -w_s * m_p_inv_SI / rho_p\
                       * ( par_sol_dens_NaCl[1] \
                           + 2.0 * par_sol_dens_NaCl[3] * w_s \
                           + par_sol_dens_NaCl[4] * T_p)

    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = 1.0E-6 * dR_p_dm_over_R_p * R_p
    
    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_p) # in SI - no unit
    kelvin_term = np.exp(eps_k)
    
    # no unit
    a_w = mat.compute_water_activity_NaCl(w_s)
    
    
    # in 1/kg
    da_w_dm = -m_p_inv_SI * w_s * compute_polynom(par_wat_act_deriv_NaCl, w_s)
    
    dsigma_dm = -par_sigma_NaCl * mat.compute_surface_tension_water(T_p) \
                * m_p_inv_SI * ( w_s / ( (1.-w_s)*(1.-w_s) ) )
            
    S_eq = a_w * kelvin_term    
    
    dSeq_dm = da_w_dm * kelvin_term \
              + S_eq * eps_k * ( dsigma_dm / sigma_p
                                 - drho_dm_over_rho - dR_p_dm_over_R_p )
    
    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb )
    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
    
    # in SI : m^2 s / kg
    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
    
    f1f3 = 4.0 * np.pi * R_p * R_p * f3 # in 1E-12
    
    # set l_alpha l_beta constant, i.e. neglect their change with m_p here
    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1 + dR_p_dm*c2
    S_eq = S_amb - S_eq
    
    return 1.0E6 * f1f3 * S_eq,\
           1.0E-12 * f1f3\
           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
compute_mass_rate_and_derivative_NaCl =\
njit()(compute_mass_rate_and_derivative_NaCl_np)

par_rho_deriv_AS = np.copy(par_rho_AS[:-1]) \
                       * np.arange(1,len(par_rho_AS))[::-1]
par_wat_act_deriv_AS = np.copy(par_wat_act_AS[:-1]) \
                       * np.arange(1,len(par_wat_act_AS))[::-1]
def compute_mass_rate_and_derivative_AS_np(m_w, m_s, w_s, R_p, T_p, rho_p,
                                           T_amb, p_amb, S_amb, e_s_amb,
                                           L_v, K, D_v, sigma_p):
    """Compute the condensation mass rate and its mass derivative
    
    Material = ammonium sulfate
    
    Condensation mass rate equation by
    Fukuta and Walter 1970, J. Atmos. Sci. 27: 1160
    
    Let gamma be the mass rate. The function returns gamma
    and d gamma / d m_w (derivative by water mass)
    
    Saturation S = e/e_s(T)
    e = ambient water vapor pressure
    e_s = saturation vapor pressure over a flat water surface
    Köhler-equation:
    Equilibrium saturation over a curved solution surface =
    S_eq = e_equi / e_s(T) = a_w * f_Kelvin
    e_equi = equilibrium vapor pressure over a curved solution surface
    a_w = water activity
    f_Kelvin = exp(kelvin_argument)
    kelvin_argument = (2 sigma_p) / (R_v * T_p * rho_p * R_p)
    
    Parameters
    ----------
    m_w: float
        Particle water mass (1E-18 kg)
    m_s: float
        Particle solute mass (1E-18 kg)
    w_s: float
        Particle solute mass fraction m_solute / m_solution (no unit)
    R_p: float
        Particle radius (microns)
    T_p: float
        Particle temperature (K)
    rho_p: float
        Mass density (kg/m^3)
    T_amb: float
        Ambient temperature (K)
    p_amb: float
        Ambient pressure (Pa)
    S_amb: float
        Ambient saturation S (no unit)
    e_s_amb: float
        Ambient saturation pressure (vapor-liquid)
    L_v: float
        Heat of vaporization (J/kg)
    K: float
        Thermal conductivity of air (W/(m K))
    D_v: float
        Diffusion coefficent of water vapor in air (m^2/s)
    sigma_p: float
        Particle surface tension (N/m)
    
    Returns
    -------
    gamma: float
        Condensation mass rate (s.a.) (1E-18 kg / s)
    dgamma_dm: float
        Derivative of the mass rate by the water mass (1/s) (SI)
    
    """
    
    # thermal size correction in SI
    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
    # diffusive size correction in SI
    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
       
    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
    # different to the NaCl function due to different density parametrization
    drho_dm_over_rho = -mat.compute_density_water(T_p) * m_p_inv_SI / rho_p\
                       * w_s * compute_polynom(par_rho_deriv_AS, w_s)

    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = 1.0E-6 * dR_p_dm_over_R_p * R_p
    
    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_p) # in SI - no unit
    kelvin_term = np.exp(eps_k)

    # no unit
    a_w = mat.compute_water_activity_AS(w_s)
    
    # in 1/kg
    da_w_dm = -m_p_inv_SI * w_s * compute_polynom(par_wat_act_deriv_AS, w_s)
    
    dsigma_dm = -par_sigma_AS * mat.compute_surface_tension_water(T_p) \
                * m_p_inv_SI * ( w_s / ( (1.-w_s)*(1.-w_s) ) )
            
    S_eq = a_w * kelvin_term
    
    dSeq_dm = da_w_dm * kelvin_term \
              + S_eq * eps_k * ( dsigma_dm / sigma_p
                                 - drho_dm_over_rho - dR_p_dm_over_R_p )
    
    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb )
    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
    # in SI : m^2 s / kg
    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
    
    f1f3 = 4.0 * np.pi * R_p * R_p * f3 # in 1E-12
    # set l_alpha l_beta constant, i.e. neglect their change with m_p here
    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1 + dR_p_dm*c2
    # use name S_eq = f2
    S_eq = S_amb - S_eq
    # NOTE: here S_eq = f2 = S_amb - S_eq
    return 1.0E6 * f1f3 * S_eq,\
           1.0E-12 * f1f3\
           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
compute_mass_rate_and_derivative_AS =\
    njit()(compute_mass_rate_and_derivative_AS_np)

#%% water activity calculation for NaCl with parametrization of the vant Hoff
#   factor (currently not in use, we use the direct polynomial form from above)
#   this might get interesting, when considering more than one solute species
#@njit()
#def compute_water_activity_NaCl_vH_mf(mass_fraction_solute, vant_Hoff_):
#    return (1. - mass_fraction_solute)\
#         / ( 1. - ( 1. - mat.molar_mass_ratio_w_NaCl * vant_Hoff_ )
#                 * mass_fraction_solute )
#
#@vectorize(
#"float64(float64, float64, float64, float64, float64, float64, float64)")
#def compute_equilibrium_saturation_NaCl_vH(m_w, m_s, w_s, R_p,
#                                        T_p, rho_p, sigma_w):
#    return mat.compute_water_activity_NaCl_vH(m_w, m_s, w_s)\
#           * compute_kelvin_term(R_p, rho_p, T_p, sigma_w)
#
#@njit()
#def compute_equilibrium_saturation_NaCl_from_vH_mf(mass_fraction_solute,
#                                                   temperature,
#                                                   vant_Hoff_,
#                                                   mass_solute,
#                                                   mass_density_particle,
#                                                   surface_tension_):
#    return compute_water_activity_NaCl_vH_mf(mass_fraction_solute,vant_Hoff_)\
#           * compute_kelvin_term_mf(mass_fraction_solute,
#                                    temperature,
#                                    mass_solute,
#                                    mass_density_particle,
#                                    surface_tension_)
#           
#@njit()
#def compute_equilibrium_saturation_NaCl_vH_mf(mass_fraction_solute,
#                                              temperature,
#                                              mass_solute):
#    return compute_equilibrium_saturation_NaCl_from_vH_mf(
#               mass_fraction_solute,
#               temperature,
#               mat.compute_vant_Hoff_factor_NaCl( mass_fraction_solute ),
#               mass_solute,
#               mat.compute_density_NaCl_solution(mass_fraction_solute,
#                                                 temperature),
#               mat.compute_surface_tension_water(temperature))

#def compute_equilibrium_saturation_negative_NaCl_vH_mf(mass_fraction_solute,
#                                           temperature,
#                                           mass_solute):
#    return -compute_equilibrium_saturation_NaCl_from_vH_mf(
#               mass_fraction_solute,
#               temperature,
#               mat.compute_vant_Hoff_factor_NaCl( mass_fraction_solute ),
#               mass_solute,
#               mat.compute_density_NaCl_solution(mass_fraction_solute,
#                                             temperature),
#               mat.compute_surface_tension_water(temperature))
#
#def compute_equilibrium_saturation_minus_S_amb_NaCl_vH_mf(
#        mass_fraction_solute, temperature,
#        mass_solute, ambient_saturation):
#    return -ambient_saturation\
#           + compute_equilibrium_saturation_NaCl_from_vH_mf(
#                 mass_fraction_solute,
#                 temperature,
#                 mat.compute_vant_Hoff_factor_NaCl( mass_fraction_solute ),
#                 mass_solute,
#                 mat.compute_density_NaCl_solution(mass_fraction_solute,
#                                                 temperature),
#                 mat.compute_surface_tension_water(temperature))
## input:
## m_s
## S_amb
## T_amb
## 0. calculate efflorescence mass fraction
## 1. S_effl = S_eq (w_s_effl, m_s)
## 2. if S_a <= S_effl : w_s = w_s_effl
## 3. else (S_a > S_effl): S_act, w_s_act = max( S(w_s, m_s) )
## 4a. w_s_act = 1.00001 * w_s_act (numerical stability ->
## want to be on branch of high w_s <-> low R_p for cont. fct. S(w_s) )
## 4b. S_act = S(w_s_act)   ( < S_act_real! )
## 5. if S_a > S_act : w_s_init = w_s_act
## 6. else (S_a <= S_act) : calc w_s_init from S( w_s_init ) - S_a = 0
## check for convergence at every stage... if not converged
#@vectorize( "float64(float64,float64,float64)", forceobj=True )
#def compute_initial_mass_fraction_solute_m_s_NaCl_vH(m_s,
#                                                  ambient_saturation,
#                                                  ambient_temperature,
#                                                  ):
#    #0.
#    w_s_effl =\
#        mat.compute_efflorescence_mass_fraction_NaCl(ambient_temperature)
#    # 1.
#    S_effl = compute_equilibrium_saturation_NaCl_mf(w_s_effl,
#                                                    ambient_temperature, m_s)
#    # 2.
#    if ambient_saturation <= S_effl:
#        w_s_init = w_s_effl
#    else:
#        # 3.
#        w_s_act, S_act, flag, nofc  = \
#            fminbound(compute_equilibrium_saturation_negative_NaCl_vH_mf,
#                      x1=1E-8, x2=w_s_effl, args=(ambient_temperature, m_s),
#                      xtol = 1.0E-12, full_output=True )
#        # 4.
#        # increase w_s_act slightly to avoid numerical problems
#        # in solving with brentq() below
#        if flag == 0:
#            w_s_act *= 1.000001
#        # set w_s_act (i.e. the min bound for brentq() solve below )
#        # to deliqu. mass fraction if fminbound does not converge
#        else:
#            w_s_act = mat.compute_solubility_NaCl(ambient_temperature)
#        # update S_act to S_act* < S_act (right branch of S_eq vs w_s curve)
#        S_act = compute_equilibrium_saturation_NaCl_mf(w_s_act,
#                                                   ambient_temperature, m_s)
#        # 5.
#        if ambient_saturation > S_act:
#            w_s_init = w_s_act
#        else:
#            # 6.
#            solve_result = \
#                brentq(
#                    compute_equilibrium_saturation_minus_S_amb_NaCl_vH_mf,
#                    w_s_act,
#                    w_s_effl,
#                    (ambient_temperature, m_s, ambient_saturation),
#                    xtol = 1e-15,
#                    full_output=True)
#            if solve_result[1].converged:
#                w_s_init = solve_result[0]
#            else:
#                w_s_init = w_s_act        
#    
#    return w_s_init

### MASS RATE NACL VERSION WITH VANT HOFF FACTOR (UNUSED)
## the linearization of l_alpha, l_beta has small effects
## for small radii, but the coefficients are somewhat arbitrary anyways.
## in fg/s = 1.0E-18 kg/s
#@vectorize(
#"float64(\
#float64, float64, float64, float64, float64, float64, float64, float64,\
#float64, float64, float64, float64, float64, float64)")
#def compute_mass_rate_NaCl_vH(m_w, m_s, w_s, R_p, T_p, rho_p,
#                      T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w):
#    S_eq = compute_equilibrium_saturation_NaCl_vH (m_w, m_s, w_s, R_p,
#                                          T_p, rho_p, sigma_w)
#    return 4.0E6 * np.pi * R_p * R_p * (S_amb - S_eq)\
#           / compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K,D_v)

#def compute_mass_rate_derivative_NaCl_vH_np(
#        m_w, m_s, w_s, R_p, T_p, rho_p, T_amb, p_amb, S_amb, e_s_amb,
#        L_v, K, D_v, sigma_w):
#    R_p_SI = 1.0E-6 * R_p # in SI: meter   
#    
#    # thermal size correction in SI
#    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
#    # diffusive size correction in SI
#    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
#       
#    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
#    drho_dm_over_rho = -w_s * m_p_inv_SI / rho_p\
#                       * (par_sol_dens_NaCl[1]
#                          + 2.0 * par_sol_dens_NaCl[3] * w_s\
#                          + par_sol_dens_NaCl[4] * T_p )
#
#    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
#    dR_p_dm = dR_p_dm_over_R_p * R_p_SI
#    
#    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_w) # in SI - no unit
#    
#    vH = mat.compute_vant_Hoff_factor_NaCl(w_s)
#    dvH_dws = mat.compute_dvH_dws_NaCl(w_s)
#    
#    # dont convert masses here 
#    h1_inv = 1.0 / (m_w + m_s * mat.molar_mass_ratio_w_NaCl * vH) 
#        
#    S_eq = m_w * h1_inv * np.exp(eps_k)
#    
#    dSeq_dm =\
#        S_eq * (1.0E18 / m_w - eps_k * ( dR_p_dm_over_R_p + drho_dm_over_rho )\
#                - (1 - mat.molar_mass_ratio_w_NaCl * dvH_dws * w_s * w_s)\
#                  * h1_inv * 1.0E18)
#    
#    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb)
#    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
#    # in SI : m^2 s / kg
#    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
#    
#    f1f3 = 4.0 * np.pi * R_p_SI * R_p_SI * f3 # SI
#    
#    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1\
#             + dR_p_dm * c2
#    return f1f3 * ( ( S_amb - S_eq )\
#                    * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
#compute_mass_rate_derivative_NaCl_vH =\
#njit()(compute_mass_rate_derivative_NaCl_vH_np)
##compute_mass_rate_derivative_NaCl_par =\
##njit(parallel = True)(compute_mass_rate_derivative_NaCl_np)
#
## return mass rate in fg/s and mass rate deriv in SI: 1/s
#def compute_mass_rate_and_derivative_NaCl_vH_np(m_w, m_s, w_s, R_p, T_p, rho_p,
#                                        T_amb, p_amb, S_amb, e_s_amb,
#                                        L_v, K, D_v, sigma_p):
#    
#    # thermal size correction in SI
#    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
#    # diffusive size correction in SI
#    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
#       
#    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
#    # dont use piecewise for now to avoid discontinuity in density...
#    drho_dm_over_rho = -w_s * m_p_inv_SI / rho_p\
#                       * (par_sol_dens_NaCl[1]
#                          + 2.0 * par_sol_dens_NaCl[3] * w_s\
#                          + par_sol_dens_NaCl[4] * T_p )
#
#    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
#    dR_p_dm = 1.0E-6 * dR_p_dm_over_R_p * R_p
#    
#    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_p) # in SI - no unit
#    
#    vH = mat.compute_vant_Hoff_factor_NaCl(w_s)
#    dvH_dws = mat.compute_dvH_dws_NaCl(w_s)
#    # dont convert masses here
#    h1_inv = 1.0 / (m_w + m_s * mat.molar_mass_ratio_w_NaCl * vH) 
#        
#    S_eq = m_w * h1_inv * np.exp(eps_k)
#    
#    dSeq_dm =\
#        S_eq * (1.0E18 / m_w - eps_k * ( dR_p_dm_over_R_p + drho_dm_over_rho )\
#                - (1 - mat.molar_mass_ratio_w_NaCl * dvH_dws * w_s * w_s)\
#                  * h1_inv * 1.0E18)
#    
#    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb)
#    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
#    # in SI : m^2 s / kg
#    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
#    
#    f1f3 = 4.0 * np.pi * R_p * R_p * f3 # in 1E-12
#    # set l_alpha l_beta constant, i.e. neglect their change with m_p here
#    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1 + dR_p_dm*c2
#    # use name S_eq = f2
#    S_eq = S_amb - S_eq
#    # NOTE: here S_eq = f2 = S_amb - S_eq
#    return 1.0E6 * f1f3 * S_eq,\
#           1.0E-12 * f1f3\
#           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
#compute_mass_rate_and_derivative_NaCl_vH =\
#    njit()(compute_mass_rate_and_derivative_NaCl_vH_np)
##compute_mass_rate_and_derivative_NaCl_par =\
##njit(parallel = True)(compute_mass_rate_and_derivative_NaCl_np)
   
    