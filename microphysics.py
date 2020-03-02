#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinetic framework
(Test Case 1 ICMW 2012)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

microphysics module

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
import materialproperties as mat

par_sol_dens_NaCl = mat.par_sol_dens_NaCl
par_rho_AS = mat.par_rho_AS

par_wat_act_AS = mat.par_wat_act_AS
par_wat_act_NaCl = mat.par_wat_act_NaCl

par_sigma_AS = mat.par_sigma_AS
par_sigma_NaCl = mat.par_sigma_NaCl
                       
from algebra import compute_polynom

#%% CONVERSIONS

# compute mass in femto gram = 10^-18 kg 
# from radius in microns (10^-6 m)
# and density in kg/m^3
@vectorize("float64(float64,float64)")
def compute_mass_from_radius_vec(radius_, density_):
    return c.pi_times_4_over_3 * density_ * radius_ * radius_ * radius_ 

@njit()
def compute_mass_from_radius(radius_, density_):
    return c.pi_times_4_over_3 * density_ * radius_ * radius_ * radius_ 

# compute radius in microns
# mass in 10^-18 kg, density in kg/m^3, radius in micro meter
# vectorize for mass array
@vectorize("float64(float64,float64)")
def compute_radius_from_mass_vec(mass_, density_):
    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)

@njit()
def compute_radius_from_mass(mass_, density_):
    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)

# molality and molec. weight have to be in inverse units, e.g.
# mol/kg and kg/mol 
@njit()
def compute_mass_fraction_from_molality(molality_, molecular_weight_):
    return 1.0 / ( 1.0 + 1.0 / (molality_ * molecular_weight_) )

# mass_frac in [-] (not percent!!)
# mol weight in kg/mol
# result in mol/kg
@njit()
def compute_molality_from_mass_fraction(mass_fraction_, molecular_weight_):
    return mass_fraction_ / ( (1. - mass_fraction_) * molecular_weight_ )

### FOR CONVENIENCE: RADIUS, MASS FRACTION AND DENSITY 
        
@njit()
def compute_R_p_w_s_rho_p_NaCl(m_w, m_s, T_p):
    m_p = m_w + m_s
    w_s = m_s / m_p
    rho_p = mat.compute_density_NaCl_solution(w_s, T_p)
    return compute_radius_from_mass(m_p, rho_p), w_s, rho_p
    
@njit()
def compute_R_p_w_s_rho_p_AS(m_w, m_s, T_p):
    m_p = m_w + m_s
    w_s = m_s / m_p
    rho_p = mat.compute_density_AS_solution(w_s, T_p)
    return compute_radius_from_mass(m_p, rho_p), w_s, rho_p

@njit()
def compute_R_p_w_s_rho_p(m_w, m_s, T_p, solute_type):
    if solute_type == "AS":
        return compute_R_p_w_s_rho_p_AS(m_w, m_s, T_p)
    elif solute_type == "NaCl":
        return compute_R_p_w_s_rho_p_NaCl(m_w, m_s, T_p)

def compute_particle_radius_from_ws_T_ms_NaCl( mass_fraction_solute_,
                                         temperature_, dry_mass_):
    Vp = dry_mass_\
        / ( mass_fraction_solute_ \
           * mat.compute_density_NaCl_solution(mass_fraction_solute_,
                                               temperature_) )
    return (c.pi_times_4_over_3_inv * Vp)**(c.one_third)

#%% FORCES
           
# Particle Reynolds number as given in Sommerfeld 2008
# radius in mu (1E-6 m)
# velocity dev = |u_f-v_p| in m/s
# density in kg/m^3
# viscosity in N s/m^2
@njit()
def compute_particle_reynolds_number(radius_, velocity_dev_, fluid_density_,
                                     fluid_viscosity_ ):
    return 2.0E-6 * fluid_density_ * radius_ * velocity_dev_ / fluid_viscosity_    


#%% EQUILIBRIUM SATURATION - KELVIN RAOULT

# R_p in mu
# result without unit -> conversion from mu 
@njit()
def compute_kelvin_argument(R_p, T_p, rho_p, sigma_p):
    return 2.0E6 * sigma_p \
           / ( c.specific_gas_constant_water_vapor * T_p * rho_p * R_p )

@njit()
def compute_kelvin_term(R_p, T_p, rho_p, sigma_w):
    return np.exp(compute_kelvin_argument(R_p, T_p, rho_p, sigma_w))

@njit()
def compute_kelvin_term_mf(mass_fraction_solute_,
                        temperature_,
                        mass_solute_,
                        mass_density_particle_,
                        surface_tension_ ):
    return np.exp( 2.0 * surface_tension_ * 1.0E6\
                   * (mass_fraction_solute_ / mass_solute_)**(c.one_third)
                   / ( c.specific_gas_constant_water_vapor
                       * temperature_
                       * mass_density_particle_**(0.66666667)
                       * c.const_volume_to_radius) )

@vectorize(
    "float64(float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_NaCl(w_s, R_p, T_p, rho_p, sigma_w):
    return mat.compute_water_activity_NaCl(w_s)\
           * compute_kelvin_term(R_p, T_p, rho_p, sigma_w)

@vectorize(
    "float64(float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_AS(w_s, R_p, T_p, rho_p, sigma_w):
    return mat.compute_water_activity_AS(w_s)\
           * compute_kelvin_term(R_p, T_p, rho_p, sigma_w)

# from mass fraction mf
@njit()
def compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_s):
    rho_p = mat.compute_density_NaCl_solution(w_s, T_p)
    sigma_p = mat.compute_surface_tension_NaCl(w_s, T_p)
    return mat.compute_water_activity_NaCl(w_s) \
           * compute_kelvin_term_mf(w_s, T_p, m_s, rho_p, sigma_p)

# from mass fraction mf
@njit()
def compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s):
    rho_p = mat.compute_density_AS_solution(w_s, T_p)
    sigma_p = mat.compute_surface_tension_AS(w_s, T_p)
    return mat.compute_water_activity_AS(w_s) \
           * compute_kelvin_term_mf(w_s, T_p, m_s, rho_p, sigma_p)
           
### water activity calculation for NaCl with parametrization of the vant Hoff
#   factor (currently not in use, we use the direct polynomial form from above)
#   this might get interesting, when considering more than one solute species
@njit()
def compute_water_activity_NaCl_vH_mf(mass_fraction_solute_, vant_Hoff_):
    return (1. - mass_fraction_solute_)\
         / ( 1. - ( 1. - mat.molar_mass_ratio_w_NaCl * vant_Hoff_ )
                 * mass_fraction_solute_ )

@vectorize(
"float64(float64, float64, float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_NaCl_vH(m_w, m_s, w_s, R_p,
                                        T_p, rho_p, sigma_w):
    return mat.compute_water_activity_NaCl_vH(m_w, m_s, w_s)\
           * compute_kelvin_term(R_p, rho_p, T_p, sigma_w)

@njit()
def compute_equilibrium_saturation_NaCl_from_vH_mf(mass_fraction_solute_,
                                                   temperature_,
                                                   vant_Hoff_,
                                                   mass_solute_,
                                                   mass_density_particle_,
                                                   surface_tension_):
    return   compute_water_activity_NaCl_vH_mf(mass_fraction_solute_, vant_Hoff_) \
           * compute_kelvin_term_mf(mass_fraction_solute_,
                                    temperature_,
                                    mass_solute_,
                                    mass_density_particle_,
                                    surface_tension_)
           
@njit()
def compute_equilibrium_saturation_NaCl_vH_mf(mass_fraction_solute_,
                                              temperature_,
                                              mass_solute_):
    return compute_equilibrium_saturation_NaCl_from_vH_mf(
               mass_fraction_solute_,
               temperature_,
               mat.compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
               mass_solute_,
               mat.compute_density_NaCl_solution(mass_fraction_solute_,
                                                 temperature_),
               mat.compute_surface_tension_water(temperature_))

#%% INITIAL MASS FRACTION

### INITIAL MASS FRACTION SODIUM CHLORIDE
@njit()
def compute_equilibrium_saturation_negative_NaCl_mf(w_s, T_p, m_s):
    return -compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_s)

def compute_equilibrium_saturation_minus_S_amb_NaCl_mf(w_s, T_p, m_s, S_amb):
    return -S_amb + compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_s)

# input:
# m_s
# S_amb
# T_amb
# 1. S_effl = S_eq (w_s_effl, m_s)
# 2. if S_a <= S_effl : w_s = w_s_effl
# 3. else (S_a > S_effl): S_act, w_s_act = max( S(w_s, m_s) )
# 4a. w_s_act = 1.00001 * w_s_act (numerical stability ->
# want to be on branch of high w_s <-> low R_p for cont. fct. S(w_s) )
# 4b. S_act = S(w_s_act)   ( < S_act_real! )
# 5. if S_a > S_act : w_s_init = w_s_act
# 6. else (S_a <= S_act) : calc w_s_init from S( w_s_init ) - S_a = 0
# check for convergence at every stage... if not converged
# for sodium chloride: fix a maximum border for w_s: w_s_max = 0.45
# w_s can not get larger than that.
# the border is chosen, because the approximation of sigma_NaCl(w_s)
# is only given for 0 < w_s < 0.45
@vectorize( "float64(float64,float64,float64)", forceobj=True )
def compute_initial_mass_fraction_solute_m_s_NaCl(m_s,
                                                  ambient_saturation_,
                                                  ambient_temperature_,
                                                  ):
    # 1.
    S_effl = compute_equilibrium_saturation_NaCl_mf(mat.w_s_max_NaCl,
                                                  ambient_temperature_, m_s)
    # 2.
    if ambient_saturation_ <= S_effl:
        w_s_init = mat.w_s_max_NaCl
    else:
        # 3.
        w_s_act, S_act, flag, nofc  = \
            fminbound(compute_equilibrium_saturation_negative_NaCl_mf,
                      x1=1E-8, x2=mat.w_s_max_NaCl,
                      args=(ambient_temperature_, m_s),
                      xtol = 1.0E-12, full_output=True )
        # 4.
        # increase w_s_act slightly to avoid numerical problems
        # in solving with brentq() below
        if flag == 0:
            w_s_act *= 1.000001
        # set w_s_act (i.e. the min bound for brentq() solve below )
        # to deliqu. mass fraction if fminbound does not converge
        else:
            w_s_act = mat.compute_solubility_NaCl(ambient_temperature_)
        S_act = compute_equilibrium_saturation_NaCl_mf(w_s_act,
                                                   ambient_temperature_, m_s)
        # 5.
        if ambient_saturation_ > S_act:
            w_s_init = w_s_act
        else:
            # 6.
            solve_result = \
                brentq(
                    compute_equilibrium_saturation_minus_S_amb_NaCl_mf,
                    w_s_act,
                    mat.w_s_max_NaCl,
                    (ambient_temperature_, m_s, ambient_saturation_),
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
    return -compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s)

def compute_equilibrium_saturation_minus_S_amb_AS_mf(w_s, T_p, m_s, S_amb):
                                                     
    return -S_amb \
           + compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s)
                 
# input:
# m_s
# S_amb
# T_amb
# 1. S_effl = S_eq (w_s_effl, m_s)
# 2. if S_a <= S_effl : w_s = w_s_effl
# 3. else (S_a > S_effl): S_act, w_s_act = max( S(w_s, m_s) )
# 4a. w_s_act = 1.00001 * w_s_act (numerical stability ->
# want to be on branch of high w_s <-> low R_p for cont. fct. S(w_s) )
# 4b. S_act = S(w_s_act)   ( < S_act_real! )
# 5. if S_a > S_act : w_s_init = w_s_act
# 6. else (S_a <= S_act) : calc w_s_init from S( w_s_init ) - S_a = 0
# check for convergence at every stage... if not converged
# for ammonium sulfate: fix a maximum border for w_s: w_s_max = 0.78
# w_s can not get larger than that.
# the border is chosen, because the approximation of sigma_AS(w_s)
# is only given for 0 < w_s < 0.78
@vectorize( "float64(float64,float64,float64)", forceobj=True )
def compute_initial_mass_fraction_solute_m_s_AS(m_s,
                                                  ambient_saturation_,
                                                  ambient_temperature_,
                                                  ):
                                                  # opt = 'None'
    # 1.
    S_effl = compute_equilibrium_saturation_AS_mf(mat.w_s_max_AS,
                                                  ambient_temperature_, m_s)
    # 2.
    # np.where(ambient_saturation_ <= S_effl, w_s_init = w_s_effl,)
    if ambient_saturation_ <= S_effl:
        w_s_init = mat.w_s_max_AS
    else:
        # 3.
        w_s_act, S_act, flag, nofc  = \
            fminbound(compute_equilibrium_saturation_negative_AS_mf,
                      x1=1E-8, x2=mat.w_s_max_AS,
                      args=(ambient_temperature_, m_s),
                      xtol = 1.0E-12, full_output=True )
        # 4.
        # increase w_s_act slightly to avoid numerical problems
        # in solving with brentq() below
        if flag == 0:
            w_s_act *= 1.000001
        # set w_s_act (i.e. the min bound for brentq() solve below )
        # to deliqu. mass fraction if fminbound does not converge
        else:
            w_s_act = mat.compute_solubility_AS(ambient_temperature_)
        # update S_act to S_act* < S_act (right branch of S_eq vs w_s curve)
        S_act = compute_equilibrium_saturation_AS_mf(w_s_act,
                                                   ambient_temperature_, m_s)
        # 5.
        if ambient_saturation_ > S_act:
            w_s_init = w_s_act
        else:
            # 6.
            solve_result = \
                brentq(
                    compute_equilibrium_saturation_minus_S_amb_AS_mf,
                    w_s_act,
                    mat.w_s_max_AS,
                    (ambient_temperature_, m_s, ambient_saturation_),
                    xtol = 1e-15,
                    full_output=True)
            if solve_result[1].converged:
                w_s_init = solve_result[0]
            else:
                w_s_init = w_s_act        
    return w_s_init

#%% INITAL MASS FRACTION USING THE VANT HOFF FACTOR VERSION FOR NACL (UNUSED)

#def compute_equilibrium_saturation_negative_NaCl_vH_mf(mass_fraction_solute_,
#                                           temperature_,
#                                           mass_solute_):
#    return -compute_equilibrium_saturation_NaCl_from_vH_mf(
#               mass_fraction_solute_,
#               temperature_,
#               mat.compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
#               mass_solute_,
#               mat.compute_density_NaCl_solution(mass_fraction_solute_,
#                                             temperature_),
#               mat.compute_surface_tension_water(temperature_))
#
#def compute_equilibrium_saturation_minus_S_amb_NaCl_vH_mf(mass_fraction_solute_,
#                                           temperature_,
#                                           mass_solute_, ambient_saturation_):
#    return -ambient_saturation_ + compute_equilibrium_saturation_NaCl_from_vH_mf(
#               mass_fraction_solute_,
#               temperature_,
#               mat.compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
#               mass_solute_,
#               mat.compute_density_NaCl_solution(mass_fraction_solute_,
#                                                 temperature_),
#               mat.compute_surface_tension_water(temperature_))
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
#                                                  ambient_saturation_,
#                                                  ambient_temperature_,
#                                                  ):
#    #0.
#    w_s_effl =\
#        mat.compute_efflorescence_mass_fraction_NaCl(ambient_temperature_)
#    # 1.
#    S_effl = compute_equilibrium_saturation_NaCl_mf(w_s_effl,
#                                                    ambient_temperature_, m_s)
#    # 2.
#    if ambient_saturation_ <= S_effl:
#        w_s_init = w_s_effl
#    else:
#        # 3.
#        w_s_act, S_act, flag, nofc  = \
#            fminbound(compute_equilibrium_saturation_negative_NaCl_vH_mf,
#                      x1=1E-8, x2=w_s_effl, args=(ambient_temperature_, m_s),
#                      xtol = 1.0E-12, full_output=True )
#        # 4.
#        # increase w_s_act slightly to avoid numerical problems
#        # in solving with brentq() below
#        if flag == 0:
#            w_s_act *= 1.000001
#        # set w_s_act (i.e. the min bound for brentq() solve below )
#        # to deliqu. mass fraction if fminbound does not converge
#        else:
#            w_s_act = mat.compute_solubility_NaCl(ambient_temperature_)
#        # update S_act to S_act* < S_act (right branch of S_eq vs w_s curve)
#        S_act = compute_equilibrium_saturation_NaCl_mf(w_s_act,
#                                                   ambient_temperature_, m_s)
#        # 5.
#        if ambient_saturation_ > S_act:
#            w_s_init = w_s_act
#        else:
#            # 6.
#            solve_result = \
#                brentq(
#                    compute_equilibrium_saturation_minus_S_amb_NaCl_vH_mf,
#                    w_s_act,
#                    w_s_effl,
#                    (ambient_temperature_, m_s, ambient_saturation_),
#                    xtol = 1e-15,
#                    full_output=True)
#            if solve_result[1].converged:
#                w_s_init = solve_result[0]
#            else:
#                w_s_init = w_s_act        
#    
#    return w_s_init


#%% CONDENSATION MASS RATE (= "gamma")
    
# Size corrections Fukuta (both in mu!)
# size corrections in droplet growth equation of Fukuta 1970
# used in Szumowski 1998 
# we use adiabatic index = 1.4 = 7/5 -> 1/1.4 = 5/7 = 0.7142857142857143
# also accommodation coeff = 1.0
# and c_v_air = c_v_air_dry_NTP

accommodation_coeff = 1.0
adiabatic_index_inv = 0.7142857142857143

### linearization of the size correction functions of Fukuta 1970
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
# in mu
@vectorize("float64(float64,float64,float64)")
def compute_l_alpha_lin(T_amb, p_amb, K):
    return ( c_alpha_1 + c_alpha_2 * (T_amb - T_alpha_0) ) * K / p_amb

condensation_coeff = 0.0415
c_beta_1 = 1.0E6 * math.sqrt( 2.0 * np.pi * c.molar_mass_water\
                              / ( c.universal_gas_constant * T_alpha_0 ) )\
                   / condensation_coeff
c_beta_2 = -0.5 / T_alpha_0
# in mu
@vectorize("float64(float64,float64)")
def compute_l_beta_lin(T_amb, D_v):
    return c_beta_1 * ( 1.0 + c_beta_2 * (T_amb - T_alpha_0) ) * D_v

# in SI
@vectorize(
"float64(\
float64, float64, float64, float64, float64, float64, float64, float64)")
def compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v  ):
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
                           T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p):
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
                         T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_p):
    S_eq = compute_equilibrium_saturation_AS(w_s, R_p,
                                          T_p, rho_p, sigma_p)
    return 4.0E6 * np.pi * R_p * R_p * (S_amb - S_eq)\
           / compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v)  

#### MASS RATE NACL VERSION WITH VANT HOFF FACTOR (UNUSED)
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
#           / compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v)


# return mass rate in fg/s and mass rate deriv in SI: 1/s
# function was compared to compute_mass_rate_NaCl and yields same results
# (1E-15 rel err.)
# analytic derivative was tested versus numerical derivative:
# rel error is < 1E-8 for sufficiently small increments
par_wat_act_deriv_NaCl = np.copy(par_wat_act_NaCl[:-1]) \
                       * np.arange(1,len(par_wat_act_NaCl))[::-1]
def compute_mass_rate_and_derivative_NaCl_np(m_w, m_s, w_s, R_p, T_p, rho_p,
                                             T_amb, p_amb, S_amb, e_s_amb,
                                             L_v, K, D_v, sigma_p):
    
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


# return mass rate in fg/s and mass rate deriv in SI: 1/s
par_rho_deriv_AS = np.copy(par_rho_AS[:-1]) \
                       * np.arange(1,len(par_rho_AS))[::-1]
par_wat_act_deriv_AS = np.copy(par_wat_act_AS[:-1]) \
                       * np.arange(1,len(par_wat_act_AS))[::-1]
def compute_mass_rate_and_derivative_AS_np(m_w, m_s, w_s, R_p, T_p, rho_p,
                                           T_amb, p_amb, S_amb, e_s_amb,
                                           L_v, K, D_v, sigma_p):
    
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

### MASS RATE OF NACL WITH VANT HOFF FACTOR (UNUSED)

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
#                       * (par_sol_dens_NaCl[1] + 2.0 * par_sol_dens_NaCl[3] * w_s\
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
#                       * (par_sol_dens_NaCl[1] + 2.0 * par_sol_dens_NaCl[3] * w_s\
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
#    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb )
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
   
    