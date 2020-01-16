#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Microphysics module
"""

#%% MODULE IMPORTS

import math
import numpy as np
from scipy.optimize import fminbound
from scipy.optimize import brentq
from numba import njit, vectorize

import constants as c
#import materialproperties
#molar_mass_ratio_w_NaCl = materialproperties.molar_mass_ratio_w_NaCl
from materialproperties import compute_density_water,\
                               compute_density_NaCl_solution,\
                               compute_density_AS_solution,\
                               compute_solubility_NaCl,\
                               compute_solubility_AS,\
                               compute_surface_tension_water,\
                               compute_surface_tension_NaCl,\
                               compute_surface_tension_AS,\
                               compute_water_activity_NaCl,\
                               compute_water_activity_AS,\
                               compute_vant_Hoff_factor_NaCl,\
                               compute_water_activity_NaCl_vH,\
                               compute_dvH_dws_NaCl,\
                               compute_efflorescence_mass_fraction_NaCl,\
                               molar_mass_ratio_w_NaCl,\
                               par_sol_dens_NaCl,\
                               par_rho_AS,\
                               par_wat_act_AS,\
                               par_wat_act_NaCl,\
                               par_sigma_AS,\
                               par_sigma_NaCl,\
                               w_s_max_AS,\
                               w_s_max_NaCl
                               
#from atmosphere import compute_specific_heat_capacity_air_moist,\
#                       compute_diffusion_constant,\
#                       compute_thermal_conductivity_air
                       
from algebra import compute_polynom

###############################################################################
#%% CONVERSIONS

# compute mass in femto gram = 10^-18 kg 
# from radius in microns 
# and density in kg/m^3
@vectorize("float64(float64,float64)")
def compute_mass_from_radius_vec(radius_, density_):
    return c.pi_times_4_over_3 * density_ * radius_ * radius_ * radius_ 

@njit()
# jitted version
def compute_mass_from_radius(radius_, density_):
    return c.pi_times_4_over_3 * density_ * radius_ * radius_ * radius_ 

# compute radius in microns
# mass in 10^-18 kg, density in kg/m^3, radius in micro meter
# vectorize for mass array
@vectorize("float64(float64,float64)")
def compute_radius_from_mass_vec(mass_, density_):
    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)

@njit()
# jitted for single mass value when function is used in another jitted function
def compute_radius_from_mass(mass_, density_):
    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)

#@vectorize("float64(float64,float64)", target="parallel")
#def compute_radius_from_mass_par(mass_, density_):
#    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)

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
    rho_p = compute_density_NaCl_solution(w_s, T_p)
    return compute_radius_from_mass(m_p, rho_p), w_s, rho_p
    
@njit()
def compute_R_p_w_s_rho_p_AS(m_w, m_s, T_p):
    m_p = m_w + m_s
    w_s = m_s / m_p
    rho_p = compute_density_AS_solution(w_s, T_p)
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
           * compute_density_NaCl_solution(mass_fraction_solute_, temperature_) )
    return (c.pi_times_4_over_3_inv * Vp)**(c.one_third)

###############################################################################
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
#def compute_particle_Reynolds_number(radius_, v_, u_, fluid_density_,
#                                       fluid_viscosity_ ):
#    return 2 * fluid_density_ *\
#    radius_ * ( deviation_magnitude_between_vectors(v_, u_) )
# / fluid_viscosity_    

# size corrections in droplet growth equation of Fukuta 1970
# used in Szumowski 1998 
# we use adiabatic index = 1.4 = 7/5 -> 1/1.4 = 5/7 = 0.7142857142857143
# also accommodation coeff = 1.0
# and c_v_air = c_v_air_dry_NTP

###############################################################################
#%% EQUILIBRIUM SATURATION - KELVIN RAOULT

# R_p in mu
# result without unit -> conversion from mu 
@vectorize("float64(float64,float64,float64,float64)")
def compute_kelvin_argument(R_p, T_p, rho_p, sigma_w):
    return 2.0E6 * sigma_w \
           / ( c.specific_gas_constant_water_vapor * T_p * rho_p * R_p )

#@vectorize(
#   "float64[::1](float64[::1], float64[::1], float64[::1], float64[::1])")
@vectorize( "float64(float64, float64, float64, float64)")
def compute_kelvin_term(R_p, T_p, rho_p, sigma_w):
    return np.exp(compute_kelvin_argument(R_p, T_p, rho_p, sigma_w))

#@vectorize( "float64(float64, float64, float64, float64)", target = "parallel")
#def compute_kelvin_term_par(R_p, T_p, rho_p, sigma_w):
#    return np.exp(compute_kelvin_argument(R_p, T_p, rho_p, sigma_w))

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
                       * c.const_volume_to_radius)
                 )

@vectorize(
    "float64(float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_NaCl(w_s, R_p, T_p, rho_p, sigma_w):
    return compute_water_activity_NaCl(w_s)\
           * compute_kelvin_term(R_p, T_p, rho_p, sigma_w)

@vectorize(
    "float64(float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_AS(w_s, R_p, T_p, rho_p, sigma_w):
    return compute_water_activity_AS(w_s)\
           * compute_kelvin_term(R_p, T_p, rho_p, sigma_w)
           
# @vectorize(
#   "float64(float64, float64, float64, float64, float64, float64, float64)",
#   target = "parallel")
# def compute_equilibrium_saturation_par(m_w, m_s, w_s, R_p, T_p, rho_p, sigma_w):
    # return compute_water_activity_NaCl(m_w, m_s, w_s)\
    #        * compute_kelvin_term(R_p, rho_p, T, sigma_w)


@njit()
def compute_water_activity_NaCl_vH_mf(mass_fraction_solute_, vant_Hoff_):
#    vant_Hoff =\
#        np.where(mass_fraction_solute_ < mf_cross,
#                 2.0,
#                 self.compute_vant_Hoff_factor_fit_init(mass_fraction_solute_))
#    return ( 1 - mass_fraction_solute_ )\
#           / ( 1 - (1 - molar_mass_ratio_w_NaCl\
#                       * compute_vant_Hoff_factor_NaCl(mass_fraction_solute_))\
#                   * mass_fraction_solute_ )
    return (1. - mass_fraction_solute_)\
         / ( 1. - ( 1. - molar_mass_ratio_w_NaCl * vant_Hoff_ )
                 * mass_fraction_solute_ )

@vectorize(
"float64(float64, float64, float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation_NaCl_vH(m_w, m_s, w_s, R_p,
                                        T_p, rho_p, sigma_w):
    return compute_water_activity_NaCl_vH(m_w, m_s, w_s)\
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

### IN WORK: remove vant Hoff factgor here, apply initial saturation/size
#### algorithm analog to AS
# NOTE that the effect of sigma in comparison to sigma_water
# on the kelvin term and the equilibrium saturation is very small
# for small R_s = 5 nm AND small R_p = 6 nm
# the deviation reaches 6 %
# but for larger R_s AND/OR larger R_p=10 nm, the deviation resides at < 1 %            
# this is why it is possible to use the surface tension of water
# for the calculations with NaCl
@njit()
def compute_equilibrium_saturation_NaCl_vH_mf(mass_fraction_solute_,
                                           temperature_,
                                           mass_solute_):
    return compute_equilibrium_saturation_NaCl_from_vH_mf(
               mass_fraction_solute_,
               temperature_,
               compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
               mass_solute_,
               compute_density_NaCl_solution(mass_fraction_solute_, temperature_),
               compute_surface_tension_water(temperature_))

@njit()
def compute_equilibrium_saturation_NaCl_mf(w_s, T_p, m_s):
    rho_p = compute_density_NaCl_solution(w_s, T_p)
    sigma_p = compute_surface_tension_NaCl(w_s, T_p)
    return compute_water_activity_NaCl(w_s) \
           * compute_kelvin_term_mf(w_s, T_p, m_s, rho_p, sigma_p)

@njit()
def compute_equilibrium_saturation_AS_mf(w_s, T_p, m_s):
    rho_p = compute_density_AS_solution(w_s, T_p)
    sigma_p = compute_surface_tension_AS(w_s, T_p)
    return compute_water_activity_AS(w_s) \
           * compute_kelvin_term_mf(w_s, T_p, m_s, rho_p, sigma_p)

#%% INITIAL MASS FRACTION NaCl

# for ammonium sulfate: fix a maximum border for w_s: w_s_max = 0.78
# w_s can not get larger than that.
# the border is chosen, because the approximation of sigma_AS(w_s)
# is only given for 0 < w_s < 0.78
           
@njit()
def compute_equilibrium_saturation_negative_NaCl_mf(w_s, T_p, m_s):
    return -compute_equilibrium_saturation_NaCl_mf(
                w_s, T_p, m_s)

def compute_equilibrium_saturation_minus_S_amb_NaCl_mf(w_s, T_p, m_s, 
                                                     ambient_saturation_):
    return -ambient_saturation_ \
           + compute_equilibrium_saturation_NaCl_mf(
                 w_s, T_p, m_s)

# this function was tested and yields the same results as the non-vectorized
# version. The old version had to be modified because inside vectorized
# function, you can not create another function via lambda: 
#@vectorize( "float64(float64,float64,float64)")
@vectorize( "float64(float64,float64,float64)", forceobj=True )
def compute_initial_mass_fraction_solute_m_s_NaCl(m_s,
                                                  ambient_saturation_,
                                                  ambient_temperature_,
                                                  ):
                                                  # opt = 'None'
    # 0.
#    m_s = compute_mass_from_radius(radius_dry_, c.mass_density_NaCl_dry)
#    w_s_effl = compute_efflorescence_mass_fraction_NaCl(ambient_temperature_)
    # 1.
    S_effl = compute_equilibrium_saturation_NaCl_mf(w_s_max_NaCl,
                                                  ambient_temperature_, m_s)
    # 2.
    # np.where(ambient_saturation_ <= S_effl, w_s_init = w_s_effl,)
    if ambient_saturation_ <= S_effl:
        w_s_init = w_s_max_NaCl
    else:
        # 3.
        w_s_act, S_act, flag, nofc  = \
            fminbound(compute_equilibrium_saturation_negative_NaCl_mf,
                      x1=1E-8, x2=w_s_max_NaCl, args=(ambient_temperature_, m_s),
                      xtol = 1.0E-12, full_output=True )
        # 4.
        # increase w_s_act slightly to avoid numerical problems
        # in solving with brentq() below
        if flag == 0:
            w_s_act *= 1.000001
        # set w_s_act (i.e. the min bound for brentq() solve below )
        # to deliqu. mass fraction if fminbound does not converge
        else:
            w_s_act = compute_solubility_NaCl(ambient_temperature_)
        # update S_act to S_act* < S_act (right branch of S_eq vs w_s curve)
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
                    # lambda w: compute_equilibrium_saturation_NaCl_mf(
                    #               w, ambient_temperature_, m_s)\
                    #           - ambient_saturation_,
                    w_s_act,
                    w_s_max_NaCl,
                    (ambient_temperature_, m_s, ambient_saturation_),
                    xtol = 1e-15,
                    full_output=True)
            if solve_result[1].converged:
                w_s_init = solve_result[0]
    #         solute_mass_fraction
    # = brentq(droplet.compute_equilibrium_saturation_mf_init,
    #            mf_max, mf_del, args = S_a)
            else:
                w_s_init = w_s_act        
    
    # if opt == 'verbose':
    #     w_s_act, S_act, flag, nofc  = \
    #         fminbound(lambda w: -compute_equilibrium_saturation_NaCl_mf(
    #                                 w, ambient_temperature_, m_s),
    #                   x1=1E-8, x2=w_s_effl, xtol = 1.0E-12, full_output=True )
    #     S_act = -S_act
    #     return w_s_init, w_s_act, S_act
    # else:
    return w_s_init

###########################################################################

def compute_equilibrium_saturation_negative_NaCl_vH_mf(mass_fraction_solute_,
                                           temperature_,
                                           mass_solute_):
    return -compute_equilibrium_saturation_NaCl_from_vH_mf(
               mass_fraction_solute_,
               temperature_,
               compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
               mass_solute_,
               compute_density_NaCl_solution(mass_fraction_solute_, temperature_),
               compute_surface_tension_water(temperature_))

def compute_equilibrium_saturation_minus_S_amb_NaCl_vH_mf(mass_fraction_solute_,
                                           temperature_,
                                           mass_solute_, ambient_saturation_):
    return -ambient_saturation_ + compute_equilibrium_saturation_NaCl_from_vH_mf(
               mass_fraction_solute_,
               temperature_,
               compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
               mass_solute_,
               compute_density_NaCl_solution(mass_fraction_solute_, temperature_),
               compute_surface_tension_water(temperature_))

### INITIALIZE MASS FRACTION
# input:
# R_dry
# S_amb
# T_amb
# 0. Set m_s = m_s(R_dry), T_p = T_a, calc w_s_effl 
# 1. S_effl = S_eq (w_s_effl, m_s)
# 2. if S_a <= S_effl : w_s = w_s_effl
# 3. else (S_a > S_effl): S_act, w_s_act = max( S(w_s, m_s) )
# 4a. w_s_act = 1.00001 * w_s_act (numerical stability ->
# want to be on branch of high w_s <-> low R_p for cont. fct. S(w_s) )
# 4b. S_act = S(w_s_act)   ( < S_act_real! )
# 5. if S_a > S_act : w_s_init = w_s_act
# 6. else (S_a <= S_act) : calc w_s_init from S( w_s_init ) - S_a = 0
## check for convergence at every stage... if not converged
# -> set to activation radius ???

# this function was tested and yields the same results as the non-vectorized
# version. The old version had to be modified because inside vectorized
# function, you can not create another function via lambda: 
#@vectorize( "float64(float64,float64,float64)", forceobj=True )
#def compute_initial_mass_fraction_solute_NaCl(radius_dry_,
#                                              ambient_saturation_,
#                                              ambient_temperature_,
#                                              # opt = 'None'
#                                              ):
#    # 0.
#    m_s = compute_mass_from_radius(radius_dry_, c.mass_density_NaCl_dry)
#    w_s_effl = compute_efflorescence_mass_fraction_NaCl(ambient_temperature_)
#    # 1.
#    S_effl = compute_equilibrium_saturation_NaCl_mf(w_s_effl,
#                                                ambient_temperature_, m_s)
#    # 2.
#    # np.where(ambient_saturation_ <= S_effl, w_s_init = w_s_effl,)
#    if ambient_saturation_ <= S_effl:
#        w_s_init = w_s_effl
#    else:
#        # 3.
#        w_s_act, S_act, flag, nofc  = \
#            fminbound(compute_equilibrium_saturation_negative_NaCl_mf,
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
#            w_s_act = compute_solubility_NaCl(ambient_temperature_)
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
#                    compute_equilibrium_saturation_minus_S_amb_NaCl_mf,
#                    # lambda w: compute_equilibrium_saturation_NaCl_mf(
#                    #               w, ambient_temperature_, m_s)\
#                    #           - ambient_saturation_,
#                    w_s_act,
#                    w_s_effl,
#                    (ambient_temperature_, m_s, ambient_saturation_),
#                    xtol = 1e-15,
#                    full_output=True)
#            if solve_result[1].converged:
#                w_s_init = solve_result[0]
#    #         solute_mass_fraction
#    # = brentq(droplet.compute_equilibrium_saturation_mf_init,
#    #            mf_max, mf_del, args = S_a)
#            else:
#                w_s_init = w_s_act        
#    
#    # if opt == 'verbose':
#    #     w_s_act, S_act, flag, nofc  = \
#    #         fminbound(lambda w: -compute_equilibrium_saturation_NaCl_mf(
#    #                                 w, ambient_temperature_, m_s),
#    #                   x1=1E-8, x2=w_s_effl, xtol = 1.0E-12, full_output=True )
#    #     S_act = -S_act
#    #     return w_s_init, w_s_act, S_act
#    # else:
#    return w_s_init

# this function was tested and yields the same results as the non-vectorized
# version. The old version had to be modified because inside vectorized
# function, you can not create another function via lambda: 
@vectorize( "float64(float64,float64,float64)", forceobj=True )
def compute_initial_mass_fraction_solute_m_s_NaCl_vH(m_s,
                                                  ambient_saturation_,
                                                  ambient_temperature_,
                                                  # opt = 'None'
                                                  ):
    # 0.
#    m_s = compute_mass_from_radius(radius_dry_, c.mass_density_NaCl_dry)
    w_s_effl = compute_efflorescence_mass_fraction_NaCl(ambient_temperature_)
    # 1.
    S_effl = compute_equilibrium_saturation_NaCl_mf(w_s_effl,
                                                ambient_temperature_, m_s)
    # 2.
    # np.where(ambient_saturation_ <= S_effl, w_s_init = w_s_effl,)
    if ambient_saturation_ <= S_effl:
        w_s_init = w_s_effl
    else:
        # 3.
        w_s_act, S_act, flag, nofc  = \
            fminbound(compute_equilibrium_saturation_negative_NaCl_vH_mf,
                      x1=1E-8, x2=w_s_effl, args=(ambient_temperature_, m_s),
                      xtol = 1.0E-12, full_output=True )
        # 4.
        # increase w_s_act slightly to avoid numerical problems
        # in solving with brentq() below
        if flag == 0:
            w_s_act *= 1.000001
        # set w_s_act (i.e. the min bound for brentq() solve below )
        # to deliqu. mass fraction if fminbound does not converge
        else:
            w_s_act = compute_solubility_NaCl(ambient_temperature_)
        # update S_act to S_act* < S_act (right branch of S_eq vs w_s curve)
        S_act = compute_equilibrium_saturation_NaCl_mf(w_s_act,
                                                   ambient_temperature_, m_s)
        # 5.
        if ambient_saturation_ > S_act:
            w_s_init = w_s_act
        else:
            # 6.
            solve_result = \
                brentq(
                    compute_equilibrium_saturation_minus_S_amb_NaCl_vH_mf,
                    # lambda w: compute_equilibrium_saturation_NaCl_mf(
                    #               w, ambient_temperature_, m_s)\
                    #           - ambient_saturation_,
                    w_s_act,
                    w_s_effl,
                    (ambient_temperature_, m_s, ambient_saturation_),
                    xtol = 1e-15,
                    full_output=True)
            if solve_result[1].converged:
                w_s_init = solve_result[0]
    #         solute_mass_fraction
    # = brentq(droplet.compute_equilibrium_saturation_mf_init,
    #            mf_max, mf_del, args = S_a)
            else:
                w_s_init = w_s_act        
    
    # if opt == 'verbose':
    #     w_s_act, S_act, flag, nofc  = \
    #         fminbound(lambda w: -compute_equilibrium_saturation_NaCl_mf(
    #                                 w, ambient_temperature_, m_s),
    #                   x1=1E-8, x2=w_s_effl, xtol = 1.0E-12, full_output=True )
    #     S_act = -S_act
    #     return w_s_init, w_s_act, S_act
    # else:
    return w_s_init

#%% INITIAL MASS FRACTION AMMONIUM SULFATE
# for ammonium sulfate: fix a maximum border for w_s: w_s_max = 0.78
# w_s can not get larger than that.
# the border is chosen, because the approximation of sigma_AS(w_s)
# is only given for 0 < w_s < 0.78
@njit()
def compute_equilibrium_saturation_negative_AS_mf(w_s, T_p, m_s):
    return -compute_equilibrium_saturation_AS_mf(
                w_s, T_p, m_s)

def compute_equilibrium_saturation_minus_S_amb_AS_mf(w_s, T_p, m_s, 
                                                     ambient_saturation_):
    return -ambient_saturation_ \
           + compute_equilibrium_saturation_AS_mf(
                 w_s, T_p, m_s)

# this function was tested and yields the same results as the non-vectorized
# version. The old version had to be modified because inside vectorized
# function, you can not create another function via lambda: 
#@vectorize( "float64(float64,float64,float64)")
@vectorize( "float64(float64,float64,float64)", forceobj=True )
def compute_initial_mass_fraction_solute_m_s_AS(m_s,
                                                  ambient_saturation_,
                                                  ambient_temperature_,
                                                  ):
                                                  # opt = 'None'
    # 0.
#    m_s = compute_mass_from_radius(radius_dry_, c.mass_density_NaCl_dry)
#    w_s_effl = compute_efflorescence_mass_fraction_NaCl(ambient_temperature_)
    # 1.
    S_effl = compute_equilibrium_saturation_AS_mf(w_s_max_AS,
                                                  ambient_temperature_, m_s)
    # 2.
    # np.where(ambient_saturation_ <= S_effl, w_s_init = w_s_effl,)
    if ambient_saturation_ <= S_effl:
        w_s_init = w_s_max_AS
    else:
        # 3.
        w_s_act, S_act, flag, nofc  = \
            fminbound(compute_equilibrium_saturation_negative_AS_mf,
                      x1=1E-8, x2=w_s_max_AS, args=(ambient_temperature_, m_s),
                      xtol = 1.0E-12, full_output=True )
        # 4.
        # increase w_s_act slightly to avoid numerical problems
        # in solving with brentq() below
        if flag == 0:
            w_s_act *= 1.000001
        # set w_s_act (i.e. the min bound for brentq() solve below )
        # to deliqu. mass fraction if fminbound does not converge
        else:
            w_s_act = compute_solubility_AS(ambient_temperature_)
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
                    # lambda w: compute_equilibrium_saturation_NaCl_mf(
                    #               w, ambient_temperature_, m_s)\
                    #           - ambient_saturation_,
                    w_s_act,
                    w_s_max_AS,
                    (ambient_temperature_, m_s, ambient_saturation_),
                    xtol = 1e-15,
                    full_output=True)
            if solve_result[1].converged:
                w_s_init = solve_result[0]
    #         solute_mass_fraction
    # = brentq(droplet.compute_equilibrium_saturation_mf_init,
    #            mf_max, mf_del, args = S_a)
            else:
                w_s_init = w_s_act        
    
    # if opt == 'verbose':
    #     w_s_act, S_act, flag, nofc  = \
    #         fminbound(lambda w: -compute_equilibrium_saturation_NaCl_mf(
    #                                 w, ambient_temperature_, m_s),
    #                   x1=1E-8, x2=w_s_effl, xtol = 1.0E-12, full_output=True )
    #     S_act = -S_act
    #     return w_s_init, w_s_act, S_act
    # else:
    return w_s_init

###############################################################################
#%% MASS RATE FUNCTIONS
    
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

#@vectorize(
#"float64(\
#float64, float64, float64, float64, float64, float64, float64, float64)",
#target = "parallel")
#def compute_gamma_denom_par(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v  ):
#    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb)
#    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
#    l_alpha = compute_l_alpha_lin(T_amb, p_amb, K)
#    l_beta = compute_l_beta_lin(T_amb, D_v)
#    return 1.0E-6 * ( c1 * S_eq * (R_p + l_alpha) + c2 * (R_p + l_beta) )

### the functions mass_rate, mass_rate_deriv and mass_rate_and_deriv
# were checked with the old versions.. -> rel err 1E-12 (numeric I guess)
### the linearization of l_alpha, l_beta has small effects
# for small radii, but the coefficients are somewhat arbitrary anyways.
# in fg/s = 1.0E-18 kg/s
@vectorize(
"float64(\
float64, float64, float64, float64, float64, float64, float64, float64,\
float64, float64, float64, float64, float64, float64)")
def compute_mass_rate_NaCl_vH(m_w, m_s, w_s, R_p, T_p, rho_p,
                      T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w):
    S_eq = compute_equilibrium_saturation_NaCl_vH (m_w, m_s, w_s, R_p,
                                          T_p, rho_p, sigma_w)
    return 4.0E6 * np.pi * R_p * R_p * (S_amb - S_eq)\
           / compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v)

# in fg/s = 1.0E-18 kg/s
# function was compared to   compute_mass_rate_NaCl_vH:
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

#@vectorize(
#"float64(float64, float64, float64, float64, float64, float64, float64,\
#float64, float64, float64, float64, float64, float64, float64)",
#target = "parallel")
#def compute_mass_rate_NaCl_par(m_w, m_s, w_s, R_p, T_p, rho_p,
#                          T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w):
#    S_eq = compute_equilibrium_saturation_NaCl(m_w, m_s, w_s, R_p,
#                                          T_p, rho_p, sigma_w)
#    return 4.0E6 * np.pi * R_p * R_p * (S_amb - S_eq)\
#           / compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v)

#@vectorize(
# "float64(float64, float64, float64, float64, float64, float64, float64,\
# float64, float64, float64, float64, float64, float64, float64)")
def compute_mass_rate_derivative_NaCl_vH_np(
        m_w, m_s, w_s, R_p, T_p, rho_p, T_amb, p_amb, S_amb, e_s_amb,
        L_v, K, D_v, sigma_w):
    R_p_SI = 1.0E-6 * R_p # in SI: meter   
    
    # thermal size correction in SI
    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
#    l_alpha_plus_R_p =\
#        1.0E-6 * (R_p + ( c_alpha_1 + c_alpha_2 * (T_amb - T_alpha_0) )\
#                        * K / p_amb)
    # diffusive size correction in SI
    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
#    l_beta_plus_R_p =\
#        1.0E-6 * (R_p + c_beta_1\
#                        * ( 1.0 + c_beta_2 * (T_amb - T_alpha_0) ) * D_v )
       
    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
    # dont use piecewise for now to avoid discontinuity in density...
    drho_dm_over_rho = -w_s * m_p_inv_SI / rho_p\
                       * (par_sol_dens_NaCl[1] + 2.0 * par_sol_dens_NaCl[3] * w_s\
                          + par_sol_dens_NaCl[4] * T_p )

    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = dR_p_dm_over_R_p * R_p_SI
#    dR_p_dm = one_third * R_p_SI * ( m_p_inv_SI - drho_dm_over_rho) # in SI
#    dR_p_dm_over_R = dR_p_dm / R_p_SI
    
    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_w) # in SI - no unit
    
    vH = compute_vant_Hoff_factor_NaCl(w_s)
    dvH_dws = compute_dvH_dws_NaCl(w_s)
#    dvH_dws = np.where(w_s < mf_cross_NaCl, 0.0, par_vH_NaCl[1])
    
    # dont convert masses here
    h1_inv = 1.0 / (m_w + m_s * molar_mass_ratio_w_NaCl * vH) 
        
    S_eq = m_w * h1_inv * np.exp(eps_k)
    
    dSeq_dm =\
        S_eq * (1.0E18 / m_w - eps_k * ( dR_p_dm_over_R_p + drho_dm_over_rho )\
                - (1 - molar_mass_ratio_w_NaCl * dvH_dws * w_s * w_s)\
                  * h1_inv * 1.0E18)
    
    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb)
    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
    # in SI : m^2 s / kg
    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
    
    f1f3 = 4.0 * np.pi * R_p_SI * R_p_SI * f3 # SI
    
    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1\
             + dR_p_dm * c2
#    f2 = S_amb - S_eq
    return f1f3 * ( ( S_amb - S_eq )\
                    * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
compute_mass_rate_derivative_NaCl_vH =\
njit()(compute_mass_rate_derivative_NaCl_vH_np)
#compute_mass_rate_derivative_NaCl_par =\
#njit(parallel = True)(compute_mass_rate_derivative_NaCl_np)

# return mass rate in fg/s and mass rate deriv in SI: 1/s
def compute_mass_rate_and_derivative_NaCl_vH_np(m_w, m_s, w_s, R_p, T_p, rho_p,
                                        T_amb, p_amb, S_amb, e_s_amb,
                                        L_v, K, D_v, sigma_p):
#    R_p_SI = 1.0E-6 * R_p # in SI: meter   
    
    # thermal size correction in SI
    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
    # diffusive size correction in SI
    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
       
    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
    # dont use piecewise for now to avoid discontinuity in density...
    drho_dm_over_rho = -w_s * m_p_inv_SI / rho_p\
                       * (par_sol_dens_NaCl[1] + 2.0 * par_sol_dens_NaCl[3] * w_s\
                          + par_sol_dens_NaCl[4] * T_p )

    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = 1.0E-6 * dR_p_dm_over_R_p * R_p
    
    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_p) # in SI - no unit
    
    vH = compute_vant_Hoff_factor_NaCl(w_s)
    dvH_dws = compute_dvH_dws_NaCl(w_s)
#    dvH_dws = np.where(w_s < mf_cross_NaCl, np.zeros_like(w_s),
#                       np.ones_like(w_s) * par_vH_NaCl[1])
    # dont convert masses here
    h1_inv = 1.0 / (m_w + m_s * molar_mass_ratio_w_NaCl * vH) 
        
    S_eq = m_w * h1_inv * np.exp(eps_k)
    
    dSeq_dm =\
        S_eq * (1.0E18 / m_w - eps_k * ( dR_p_dm_over_R_p + drho_dm_over_rho )\
                - (1 - molar_mass_ratio_w_NaCl * dvH_dws * w_s * w_s)\
                  * h1_inv * 1.0E18)
    
    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb )
    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
    # in SI : m^2 s / kg
    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
    
    f1f3 = 4.0 * np.pi * R_p * R_p * f3 # in 1E-12
    # set l_alpha l_beta constant, i.e. neglect their change with m_p here
    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1 + dR_p_dm*c2
    # use name S_eq = f2
    S_eq = S_amb - S_eq
#    f2 = S_amb - S_eq
    # NOTE: here S_eq = f2 = S_amb - S_eq
#    return 1.0E-12 * f1f3\
#           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
    return 1.0E6 * f1f3 * S_eq,\
           1.0E-12 * f1f3\
           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
#    return 1.0E6 * f1f3 * f2,\
#           1.0E-12 * f1f3\
#           * ( f2 * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
compute_mass_rate_and_derivative_NaCl_vH =\
njit()(compute_mass_rate_and_derivative_NaCl_vH_np)
#compute_mass_rate_and_derivative_NaCl_par =\
#njit(parallel = True)(compute_mass_rate_and_derivative_NaCl_np)

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
#    R_p_SI = 1.0E-6 * R_p # in SI: meter   
    
    # thermal size correction in SI
    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
    # diffusive size correction in SI
    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
       
    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
    # dont use piecewise for now to avoid discontinuity in density...
    drho_dm_over_rho = -w_s * m_p_inv_SI / rho_p\
                       * ( par_sol_dens_NaCl[1] \
                           + 2.0 * par_sol_dens_NaCl[3] * w_s \
                           + par_sol_dens_NaCl[4] * T_p)

    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = 1.0E-6 * dR_p_dm_over_R_p * R_p
    
    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_p) # in SI - no unit
    kelvin_term = np.exp(eps_k)
    
#    vH = compute_vant_Hoff_factor_NaCl(w_s)
#    dvH_dws = compute_dvH_dws_NaCl(w_s)
#    dvH_dws = np.where(w_s < mf_cross_NaCl, np.zeros_like(w_s),
#                       np.ones_like(w_s) * par_vH_NaCl[1])
    # dont convert masses here
#    h1_inv = 1.0 / (m_w + m_s * molar_mass_ratio_w_NaCl * vH) 
    
    # no unit
    a_w = compute_water_activity_NaCl(w_s)
    
    
    # in 1/kg
    da_w_dm = -m_p_inv_SI * w_s * compute_polynom(par_wat_act_deriv_NaCl, w_s)
    
    # IN WORK: UNITS?
    dsigma_dm = -par_sigma_NaCl * compute_surface_tension_water(T_p) \
                * m_p_inv_SI * ( w_s / ( (1.-w_s)*(1.-w_s) ) )
            
#    S_eq = m_w * h1_inv * np.exp(eps_k)
#    S_eq = a_w * np.exp(eps_k)
    S_eq = a_w * kelvin_term    
    
#    S_eq = m_w * h1_inv * np.exp(eps_k)
    
#    dSeq_dm =\
#        S_eq * (1.0E18 / m_w - eps_k * ( dR_p_dm_over_R_p + drho_dm_over_rho )\
#                - (1 - molar_mass_ratio_w_NaCl * dvH_dws * w_s * w_s)\
#                  * h1_inv * 1.0E18)
    
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
#    f2 = S_amb - S_eq
    # NOTE: here S_eq = f2 = S_amb - S_eq
#    return 1.0E-12 * f1f3\
#           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
    return 1.0E6 * f1f3 * S_eq,\
           1.0E-12 * f1f3\
           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
#    return 1.0E6 * f1f3 * f2,\
#           1.0E-12 * f1f3\
#           * ( f2 * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
compute_mass_rate_and_derivative_NaCl =\
njit()(compute_mass_rate_and_derivative_NaCl_np)
#compute_mass_rate_and_derivative_NaCl_par =\
#njit(parallel = True)(compute_mass_rate_and_derivative_NaCl_np)


#par_rho_deriv_AS = np.copy(par_rho_AS)
#for n in range(len(par_rho_deriv_AS)):
#    par_rho_deriv_AS[n] *= (len(par_rho_deriv_AS)-1-n)
#par_wat_act_deriv_AS = np.copy(par_wat_act_AS)
#for n in range(len(par_wat_act_deriv_AS)):
#    par_wat_act_deriv_AS[n] *= (len(par_wat_act_deriv_AS)-1-n)
#par_rho_deriv_AS = np.copy(par_rho_AS[::-1][1:])
#for n in range(1,len(par_rho_deriv_AS)):
#    par_rho_deriv_AS[n] *= n+1

par_rho_deriv_AS = np.copy(par_rho_AS[:-1]) \
                       * np.arange(1,len(par_rho_AS))[::-1]

par_wat_act_deriv_AS = np.copy(par_wat_act_AS[:-1]) \
                       * np.arange(1,len(par_wat_act_AS))[::-1]

           
# convert now sigma_w to sigma_p (surface tension)
# return mass rate in fg/s and mass rate deriv in SI: 1/s
def compute_mass_rate_and_derivative_AS_np(m_w, m_s, w_s, R_p, T_p, rho_p,
                                           T_amb, p_amb, S_amb, e_s_amb,
                                           L_v, K, D_v, sigma_p):
#    R_p_SI = 1.0E-6 * R_p # in SI: meter   
    
    # thermal size correction in SI
    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
    # diffusive size correction in SI
    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
       
    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
    # dont use piecewise for now to avoid discontinuity in density...
    # IN WORK: UNITS?
    drho_dm_over_rho = -compute_density_water(T_p) * m_p_inv_SI / rho_p\
                       * w_s * compute_polynom(par_rho_deriv_AS, w_s)
                           
    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = 1.0E-6 * dR_p_dm_over_R_p * R_p
    
    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_p) # in SI - no unit
    kelvin_term = np.exp(eps_k)
#    vH = compute_vant_Hoff_factor_NaCl(w_s)
#    dvH_dws = compute_dvH_dws(w_s)
#    dvH_dws = np.where(w_s < mf_cross_NaCl, np.zeros_like(w_s),
#                       np.ones_like(w_s) * par_vH_NaCl[1])
    # dont convert masses here
#    h1_inv = 1.0 / (m_w + m_s * molar_mass_ratio_w_NaCl * vH) 
    
    # no unit
    a_w = compute_water_activity_AS(w_s)
    
    # in 1/kg
    da_w_dm = -m_p_inv_SI * w_s * compute_polynom(par_wat_act_deriv_AS, w_s)
    
    # IN WORK: UNITS?
    dsigma_dm = -par_sigma_AS * compute_surface_tension_water(T_p) \
                * m_p_inv_SI * ( w_s / ( (1.-w_s)*(1.-w_s) ) )
            
#    S_eq = m_w * h1_inv * np.exp(eps_k)
#    S_eq = a_w * np.exp(eps_k)
    S_eq = a_w * kelvin_term
    
    dSeq_dm = da_w_dm * kelvin_term \
              + S_eq * eps_k * ( dsigma_dm / sigma_p
                                 - drho_dm_over_rho - dR_p_dm_over_R_p )
#        S_eq * (1.0E18 / m_w - eps_k * ( dR_p_dm_over_R_p + drho_dm_over_rho )\
#                - (1 - molar_mass_ratio_w_NaCl * dvH_dws * w_s * w_s)\
#                  * h1_inv * 1.0E18)
    
    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb )
    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
    # in SI : m^2 s / kg
    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
    
    f1f3 = 4.0 * np.pi * R_p * R_p * f3 # in 1E-12
    # set l_alpha l_beta constant, i.e. neglect their change with m_p here
    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1 + dR_p_dm*c2
    # use name S_eq = f2
    S_eq = S_amb - S_eq
#    f2 = S_amb - S_eq
    # NOTE: here S_eq = f2 = S_amb - S_eq
#    return 1.0E-12 * f1f3\
#           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
    return 1.0E6 * f1f3 * S_eq,\
           1.0E-12 * f1f3\
           * ( S_eq * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
#           , \
#           dR_p_dm_over_R_p, dg1_dm, dSeq_dm
#    return 1.0E6 * f1f3 * f2,\
#           1.0E-12 * f1f3\
#           * ( f2 * ( 2.0 * dR_p_dm_over_R_p - f3 * dg1_dm ) - dSeq_dm )
compute_mass_rate_and_derivative_AS =\
njit()(compute_mass_rate_and_derivative_AS_np)
#compute_mass_rate_and_derivative_AS_par =\
#njit(parallel = True)(compute_mass_rate_and_derivative_AS_np)

   
    