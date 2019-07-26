#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:28:02 2019

@author: jdesk
"""

import math
import numpy as np
from scipy.optimize import fminbound
from scipy.optimize import brentq
from numba import njit, vectorize

import constants as c
from atmosphere import compute_surface_tension_water,\
                       compute_specific_heat_capacity_air_moist,\
                       compute_diffusion_constant,\
                       compute_thermal_conductivity_air
                       
### conversions

# compute mass in femto gram = 10^-18 kg 
# from radius in microns 
# and density in kg/m^3
@vectorize("float64(float64,float64)")
def compute_mass_from_radius_vec(radius_, density_):
    return c.pi_times_4_over_3 * density_ * radius_ * radius_ * radius_ 

@njit()
# jitted for single mass value when function is used in another jitted function
def compute_mass_from_radius_jit(radius_, density_):
    return c.pi_times_4_over_3 * density_ * radius_ * radius_ * radius_ 

# compute radius in microns
# mass in 10^-18 kg, density in kg/m^3, radius in micro meter
# vectorize for mass array
@vectorize("float64(float64,float64)")
def compute_radius_from_mass_vec(mass_, density_):
    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)

@njit()
# jitted for single mass value when function is used in another jitted function
def compute_radius_from_mass_jit(mass_, density_):
    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)
@vectorize("float64(float64,float64)", target="parallel")
def compute_radius_from_mass_par(mass_, density_):
    return   ( c.pi_times_4_over_3_inv * mass_ / density_ ) ** (c.one_third)

# molality and molec. weight have to be in inverse units, e.g.
# mol/kg and kg/mol 
def compute_mass_fraction_from_molality(molality_, molecular_weight_):
    return 1.0 / ( 1.0 + 1.0 / (molality_ * molecular_weight_) )

# mass_frac in [-] (not percent!!)
# mol weight in kg/mol
# result in mol/kg
def compute_molality_from_mass_fraction(mass_fraction_, molecular_weight_):
    return mass_fraction_ / ( (1 - mass_fraction_) * molecular_weight_ )
#############################################################################
### material properties

# mass in 10^-18 kg, density in kg/m^3, radius in micro meter
#def compute_mass_solute_from_radius(self):
#    return 1.3333333 * np.pi * self.radius_solute_dry**3
#           * self.density_solute_dry

# water density in kg/m^3
# quad. fit to data from CRC 2005
# relative error is below 0.05 % in the range 0 .. 60°C
par_water_dens = np.array([  1.00013502e+03, -4.68112708e-03, 2.72389977e+02])
@vectorize("float64(float64)")  
def compute_density_water(temperature_):
    return par_water_dens[0]\
         + par_water_dens[1] * (temperature_ - par_water_dens[2])**2

# NaCl solution density in kg/m^3
# quad. fit to data from CRC 2005
# for w_s < 0.226:
# relative error is below 0.1 % for range 0 .. 50 °C
#                   below 0.17 % in range 0 .. 60 °C
# (only high w_s lead to high error)
par_sol_dens = np.array([  7.619443952135e+02,   1.021264281453e+03,
                  1.828970151543e+00, 2.405352122804e+02,
                 -1.080547892416e+00,  -3.492805028749e-03 ] )
@vectorize("float64(float64,float64)")  
def compute_density_NaCl_solution(mass_fraction_solute_, temperature_):
    return    par_sol_dens[0] \
            + par_sol_dens[1] * mass_fraction_solute_ \
            + par_sol_dens[2] * temperature_ \
            + par_sol_dens[3] * mass_fraction_solute_ * mass_fraction_solute_ \
            + par_sol_dens[4] * mass_fraction_solute_ * temperature_ \
            + par_sol_dens[5] * temperature_ * temperature_

# approx. density of the particle (droplet)
# For now, use rho(w_s,T) for all ranges, no if then else...
# to avoid discontinuity in the density
w_s_rho_p = 0.001
@vectorize("float64(float64,float64)")
def compute_density_particle(mass_fraction_solute_, temperature_):
    return    par_sol_dens[0] \
            + par_sol_dens[1] * mass_fraction_solute_ \
            + par_sol_dens[2] * temperature_ \
            + par_sol_dens[3] * mass_fraction_solute_ * mass_fraction_solute_ \
            + par_sol_dens[4] * mass_fraction_solute_ * temperature_ \
            + par_sol_dens[5] * temperature_ * temperature_
@njit()
def compute_density_particle_jit(mass_fraction_solute_, temperature_):
    return    par_sol_dens[0] \
            + par_sol_dens[1] * mass_fraction_solute_ \
            + par_sol_dens[2] * temperature_ \
            + par_sol_dens[3] * mass_fraction_solute_ * mass_fraction_solute_ \
            + par_sol_dens[4] * mass_fraction_solute_ * temperature_ \
            + par_sol_dens[5] * temperature_ * temperature_
#    return compute_density_NaCl_solution( mass_fraction_solute_, temperature_ )
#    return compute_density_water(temperature_)
#    Combine the two density functions
#    Use water density if mass fraction < 0.001,
#   then rel dev of the two density functions is < 0.1 %
#    return np.where( mass_fraction_solute_ < w_s_rho_p,
#                    compute_density_water(temperature_),
#                    compute_density_NaCl_solution(mass_fraction_solute_,
#                                                  temperature_))

# @njit("UniTuple(float64[::1], 3)(float64[::1], float64[::1], float64[::1])")
@njit()
def compute_R_p_w_s_rho_p(m_w, m_s, T_p):
    m_p = m_w + m_s
    w_s = m_s / m_p
    rho_p = compute_density_particle_jit(w_s, T_p)
    return compute_radius_from_mass_jit(m_p, rho_p), w_s, rho_p
# def compute_R_p_w_s_rho_p(m_w, m_s, T_p):
#     m_p = m_w + m_s
#     w_s = m_s / m_p
#     rho_p = compute_density_particle(w_s, T_p)
#     return compute_radius_from_mass(m_p, rho_p), w_s, rho_p

# @njit("UniTuple(float64[::1], 3)(float64[::1], float64[::1], float64[::1])",
#       parallel = True)
@njit(parallel = True)
def compute_R_p_w_s_rho_p_par(m_w, m_s, T_p):
    m_p = m_w + m_s
    w_s = m_s / m_p
    rho_p = compute_density_particle(w_s, T_p)
    return compute_radius_from_mass(m_p, rho_p), w_s, rho_p
    
def compute_particle_radius_from_ws_T_ms( mass_fraction_solute_,
                                         temperature_, dry_mass_):
#     rhop = np.where( mass_fraction_ < 0.001, 
#                     compute_density_NaCl_solution(mass_fraction_,
#                                                   temperature_),
#                     compute_density_water(temperature_))
#     if ( mass_fraction_ < 0.001 ):
#         rhop = compute_density_NaCl_solution(mass_fraction_, temperature_)
#     else:
#         rhop = compute_density_water(temperature_)
    Vp = dry_mass_\
        / ( mass_fraction_solute_ \
           * compute_density_particle(mass_fraction_solute_, temperature_) )
    return (c.pi_times_4_over_3_inv * Vp)**(c.one_third)

# solubility of NaCl in water as mass fraction w_s_sol
# saturation mass fraction (kg_solute/kg_solution)    
# fit to data from CRC 2005 page 8-115
par_solub = np.array([  3.77253081e-01,  -8.68998172e-04,   1.64705858e-06])
@vectorize("float64(float64)") 
def compute_solubility_NaCl(temperature_):
    return par_solub[0] + par_solub[1] * temperature_\
         + par_solub[2] * temperature_ * temperature_

# the supersat factor is a crude approximation
# such that the EffRH curve fits data from Biskos better
# there was no math. fitting algorithm included.
# Just graphical shifting the curve... until value
# for D_dry = 10 nm fits with fitted curve of Biskos
supersaturation_factor_NaCl = 1.92
# efflorescence mass fraction at a given temperature
# @njit("[float64[:](float64[:]),float64(float64)]")
@njit()
def compute_efflorescence_mass_fraction_NaCl(temperature_):
    return supersaturation_factor_NaCl * compute_solubility_NaCl(temperature_)

# fitted to match data from Archer 1972
# in formula from Clegg 1997 in table Hargreaves 2010
# rel dev (data from Archer 1972 in formula
# from Clegg 1997 in table Hargreaves 2010) is 
# < 0.12 % for w_s < 0.22
# < 1 % for w_s < 0.25
# approx +6 % for w_s = 0.37
# approx +15 % for w_s = 0.46
# we overestimate the water activity a_w by 6 % and rising
# 0.308250118 = M_w/M_s
par_vH_NaCl = np.array([ 1.55199086,  4.95679863])
@vectorize("float64(float64)") 
def compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_):
    return par_vH_NaCl[0] + par_vH_NaCl[1] * mass_fraction_solute_

vant_Hoff_factor_NaCl_const = 2.0

molar_mass_ratio_w_NaCl = c.molar_mass_water/c.molar_mass_NaCl

#mf_cross_NaCl = 0.090382759623349337
mf_cross_NaCl = 0.09038275962335

# vectorized version
@vectorize("float64(float64)") 
def compute_vant_Hoff_factor_NaCl(mass_fraction_solute_):
#    return compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_)
#    return vant_Hoff_factor_NaCl_const
    if mass_fraction_solute_ < mf_cross_NaCl: return vant_Hoff_factor_NaCl_const
    else: return compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_)

# numpy version
def compute_vant_Hoff_factor_NaCl_np(mass_fraction_solute_):
#    return compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_)
#    return vant_Hoff_factor_NaCl_const
    return np.where(mass_fraction_solute_ < mf_cross_NaCl,
                    vant_Hoff_factor_NaCl_const\
                    * np.ones_like(mass_fraction_solute_),
                    compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_))

@vectorize("float64(float64)")
def compute_dvH_dws(w_s):
    if w_s < mf_cross_NaCl: return 0.0
    else: return par_vH_NaCl[1]

##############################################################################
### forces
           
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

#############################################################################
### microphysics

# use continuous version for now...  
@njit()
def compute_water_activity_mf(mass_fraction_solute_, vant_Hoff_):
#    vant_Hoff =\
#        np.where(mass_fraction_solute_ < mf_cross,
#                 2.0,
#                 self.compute_vant_Hoff_factor_fit_init(mass_fraction_solute_))
#    return ( 1 - mass_fraction_solute_ )\
#           / ( 1 - (1 - molar_mass_ratio_w_NaCl\
#                       * compute_vant_Hoff_factor_NaCl(mass_fraction_solute_))\
#                   * mass_fraction_solute_ )
    return (1 - mass_fraction_solute_)\
         / ( 1 - ( 1 - molar_mass_ratio_w_NaCl * vant_Hoff_ )
                 * mass_fraction_solute_ )
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

# IN WORK
# def compute_kelvin_raoult_term(radius_particle_,
#                                    mass_fraction_solute_,
#                                    temperature_,
#                                    vant_Hoff_,
#                                    mass_density_particle_,
#                                    surface_tension_):
#     return   compute_water_activity(mass_fraction_solute_, vant_Hoff_) \
#            * compute_kelvin_term(radius_particle_,
#                                temperature_,
#                                mass_density_particle_,
#                                surface_tension_)
# IN WORK
# def compute_kelvin_raoult_term_NaCl(radius_particle_,
#                                         mass_fraction_solute_,
#                                         temperature_):
#     return compute_kelvin_raoult_term(
#                radius_particle_,
#                mass_fraction_solute_,
#                temperature_,
#                compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
#                compute_density_particle(mass_fraction_solute_, temperature_),
#                compute_surface_tension_water(temperature_))
@njit()
def compute_kelvin_raoult_term_mf(mass_fraction_solute_,
                                      temperature_,
                                      vant_Hoff_,
                                      mass_solute_,
                                      mass_density_particle_,
                                      surface_tension_):
    return   compute_water_activity_mf(mass_fraction_solute_, vant_Hoff_) \
           * compute_kelvin_term_mf(mass_fraction_solute_,
                               temperature_,
                               mass_solute_,
                               mass_density_particle_,
                               surface_tension_)
@njit()
def compute_kelvin_raoult_term_NaCl_mf(mass_fraction_solute_,
                                           temperature_,
                                           mass_solute_):
    return compute_kelvin_raoult_term_mf(
               mass_fraction_solute_,
               temperature_,
               compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
               mass_solute_,
               compute_density_particle(mass_fraction_solute_, temperature_),
               compute_surface_tension_water(temperature_))

def compute_kelvin_raoult_term_negative_NaCl_mf(mass_fraction_solute_,
                                           temperature_,
                                           mass_solute_):
    return -compute_kelvin_raoult_term_mf(
               mass_fraction_solute_,
               temperature_,
               compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
               mass_solute_,
               compute_density_particle(mass_fraction_solute_, temperature_),
               compute_surface_tension_water(temperature_))

def compute_kelvin_raoult_term_minus_S_amb_NaCl_mf(mass_fraction_solute_,
                                           temperature_,
                                           mass_solute_, ambient_saturation_):
    return -ambient_saturation_ + compute_kelvin_raoult_term_mf(
               mass_fraction_solute_,
               temperature_,
               compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
               mass_solute_,
               compute_density_particle(mass_fraction_solute_, temperature_),
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
@vectorize( "float64(float64,float64,float64)", forceobj=True )
def compute_initial_mass_fraction_solute_NaCl(radius_dry_,
                                              ambient_saturation_,
                                              ambient_temperature_,
                                              # opt = 'None'
                                              ):
    # 0.
    m_s = compute_mass_from_radius(radius_dry_, c.mass_density_NaCl_dry)
    w_s_effl = compute_efflorescence_mass_fraction_NaCl(ambient_temperature_)
    # 1.
    S_effl = compute_kelvin_raoult_term_NaCl_mf(w_s_effl,
                                                ambient_temperature_, m_s)
    # 2.
    # np.where(ambient_saturation_ <= S_effl, w_s_init = w_s_effl,)
    if ambient_saturation_ <= S_effl:
        w_s_init = w_s_effl
    else:
        # 3.
        w_s_act, S_act, flag, nofc  = \
            fminbound(compute_kelvin_raoult_term_negative_NaCl_mf,
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
        S_act = compute_kelvin_raoult_term_NaCl_mf(w_s_act,
                                                   ambient_temperature_, m_s)
        # 5.
        if ambient_saturation_ > S_act:
            w_s_init = w_s_act
        else:
            # 6.
            solve_result = \
                brentq(
                    compute_kelvin_raoult_term_minus_S_amb_NaCl_mf,
                    # lambda w: compute_kelvin_raoult_term_NaCl_mf(
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
    # = brentq(droplet.compute_kelvin_raoult_term_mf_init,
    #            mf_max, mf_del, args = S_a)
            else:
                w_s_init = w_s_act        
    
    # if opt == 'verbose':
    #     w_s_act, S_act, flag, nofc  = \
    #         fminbound(lambda w: -compute_kelvin_raoult_term_NaCl_mf(
    #                                 w, ambient_temperature_, m_s),
    #                   x1=1E-8, x2=w_s_effl, xtol = 1.0E-12, full_output=True )
    #     S_act = -S_act
    #     return w_s_init, w_s_act, S_act
    # else:
    return w_s_init

# def compute_initial_mass_fraction_solute_NaCl_old(radius_dry_,
#                                               ambient_saturation_,
#                                               ambient_temperature_,
#                                               # opt = 'None'
#                                               ):
#     # 0.
#     m_s = compute_mass_from_radius(radius_dry_, c.mass_density_NaCl_dry)
#     w_s_effl = compute_efflorescence_mass_fraction_NaCl(ambient_temperature_)
#     # 1.
#     S_effl = compute_kelvin_raoult_term_NaCl_mf(w_s_effl,
#                                                 ambient_temperature_, m_s)
#     # 2.
#     # np.where(ambient_saturation_ <= S_effl, w_s_init = w_s_effl,)
#     if ambient_saturation_ <= S_effl:
#         w_s_init = w_s_effl
#     else:
#         # 3.
#         w_s_act, S_act, flag, nofc  = \
#             fminbound(lambda w: -compute_kelvin_raoult_term_NaCl_mf(
#                                                 w, ambient_temperature_, m_s),
#                       x1=1E-8, x2=w_s_effl, xtol = 1.0E-12, full_output=True )
#         # 4.
#         # increase w_s_act slightly to avoid numerical problems
#         # in solving with brentq() below
#         if flag == 0:
#             w_s_act *= 1.000001
#         # set w_s_act (i.e. the min bound for brentq() solve below )
#         # to deliqu. mass fraction if fminbound does not converge
#         else:
#             w_s_act = compute_solubility_NaCl(ambient_temperature_)
#         # update S_act to S_act* < S_act (right branch of S_eq vs w_s curve)
#         S_act = compute_kelvin_raoult_term_NaCl_mf(w_s_act,
#                                                     ambient_temperature_, m_s)
#         # 5.
#         if ambient_saturation_ > S_act:
#             w_s_init = w_s_act
#         else:
#             # 6.
#             solve_result = \
#                 brentq(
#                     lambda w: compute_kelvin_raoult_term_NaCl_mf(
#                                   w, ambient_temperature_, m_s)\
#                               - ambient_saturation_,
#                     w_s_act,
#                     w_s_effl,
#                     xtol = 1e-15,
#                     full_output=True)
#             if solve_result[1].converged:
#                 w_s_init = solve_result[0]
#     #         solute_mass_fraction
#     # = brentq(droplet.compute_kelvin_raoult_term_mf_init,
#     #            mf_max, mf_del, args = S_a)
#             else:
#                 w_s_init = w_s_act        
    
#     # if opt == 'verbose':
#     #     w_s_act, S_act, flag, nofc  = \
#     #         fminbound(lambda w: -compute_kelvin_raoult_term_NaCl_mf(
#     #                                 w, ambient_temperature_, m_s),
#     #                 x1=1E-8, x2=w_s_effl, xtol = 1.0E-12, full_output=True )
#     #     S_act = -S_act
#     #     return w_s_init, w_s_act, S_act
#     # else:
#     return w_s_init

# Rs = np.array([0.01,0.02,0.03])
# ws = compute_initial_mass_fraction_solute_NaCl(Rs,1.01,289)

# print(ws)
# ws2 = []
# for i,R in enumerate(Rs):
#     ws2.append(compute_initial_mass_fraction_solute_NaCl_old(R,1.01,289))
#     print(ws2[i]-ws[i])


##############################################################################
### mass rate
    
# Size corrections Fukuta (both in mu!)
# size corrections in droplet growth equation of Fukuta 1970
# used in Szumowski 1998 
# we use adiabatic index = 1.4 = 7/5 -> 1/1.4 = 5/7 = 0.7142857142857143
# also accommodation coeff = 1.0
# and c_v_air = c_v_air_dry_NTP
accommodation_coeff = 1.0
adiabatic_index_inv = 0.7142857142857143

T_alpha_0 = 289 # K
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

### 
# R_p in mu
# result without unit -> conversion from mu 
@vectorize("float64(float64,float64,float64,float64)")
def compute_kelvin_argument(R_p, T_p, rho_p, sigma_w):
    return 2.0E6 * sigma_w\
                   / ( c.specific_gas_constant_water_vapor * T_p * rho_p * R_p )

#@vectorize(
#   "float64[::1](float64[::1], float64[::1], float64[::1], float64[::1])")
@vectorize( "float64(float64, float64, float64, float64)")
def compute_kelvin_term(R_p, T_p, rho_p, sigma_w):
    return np.exp(compute_kelvin_argument(R_p, T_p, rho_p, sigma_w))

@vectorize( "float64(float64, float64, float64, float64)", target = "parallel")
def compute_kelvin_term_par(R_p, T_p, rho_p, sigma_w):
    return np.exp(compute_kelvin_argument(R_p, T_p, rho_p, sigma_w))

@vectorize( "float64(float64, float64, float64)")
def compute_water_activity(m_w, m_s, w_s):
    return m_w / ( m_w + molar_mass_ratio_w_NaCl\
                   * compute_vant_Hoff_factor_NaCl(w_s) * m_s )

@vectorize(
    "float64(float64, float64, float64, float64, float64, float64, float64)")
def compute_equilibrium_saturation(m_w, m_s, w_s, R_p, rho_p, T, sigma_w):
    return compute_water_activity(m_w, m_s, w_s)\
           * compute_kelvin_term(R_p, rho_p, T, sigma_w)

# @vectorize(
#   "float64(float64, float64, float64, float64, float64, float64, float64)",
#   target = "parallel")
# def compute_equilibrium_saturation_par(m_w, m_s, w_s, R_p, rho_p, T, sigma_w):
    # return compute_water_activity(m_w, m_s, w_s)\
    #        * compute_kelvin_term(R_p, rho_p, T, sigma_w)

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

@vectorize(
"float64(\
float64, float64, float64, float64, float64, float64, float64, float64)",
target = "parallel")
def compute_gamma_denom_par(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v  ):
    c1 = L_v * L_v / (c.specific_gas_constant_water_vapor * K * T_amb * T_amb)
    c2 = c.specific_gas_constant_water_vapor * T_amb / (D_v * e_s_amb)
    l_alpha = compute_l_alpha_lin(T_amb, p_amb, K)
    l_beta = compute_l_beta_lin(T_amb, D_v)
    return 1.0E-6 * ( c1 * S_eq * (R_p + l_alpha) + c2 * (R_p + l_beta) )

### the functions mass_rate, mass_rate_deriv and mass_rate_and_deriv
# were checked with the old versions.. -> rel err 1E-12 (numeric I guess)
### the linearization of l_alpha, l_beta has small effects
# for small radii, but the coefficients are somewhat arbitrary anyways.
# in fg/s = 1.0E-18 kg/s
@vectorize(
"float64(\
float64, float64, float64, float64, float64, float64, float64, float64,\
float64, float64, float64, float64, float64, float64)")
def compute_mass_rate(m_w, m_s, w_s, R_p, T_p, rho_p,
                      T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w):
    S_eq = compute_equilibrium_saturation(m_w, m_s, w_s, R_p,
                                          T_p, rho_p, sigma_w)
    return 4.0E6 * np.pi * R_p * R_p * (S_amb - S_eq)\
           / compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v)

@vectorize(
"float64(float64, float64, float64, float64, float64, float64, float64,\
float64, float64, float64, float64, float64, float64, float64)",
target = "parallel")
def compute_mass_rate_par(m_w, m_s, w_s, R_p, T_p, rho_p,
                          T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w):
    S_eq = compute_equilibrium_saturation(m_w, m_s, w_s, R_p,
                                          T_p, rho_p, sigma_w)
    return 4.0E6 * np.pi * R_p * R_p * (S_amb - S_eq)\
           / compute_gamma_denom(R_p, S_eq, T_amb, p_amb, e_s_amb, L_v, K, D_v)

#@vectorize(
# "float64(float64, float64, float64, float64, float64, float64, float64,\
# float64, float64, float64, float64, float64, float64, float64)")
def compute_mass_rate_derivative_np(
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
                       * (par_sol_dens[1] + 2.0 * par_sol_dens[3] * w_s\
                          + par_sol_dens[4] * T_p )

    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = dR_p_dm_over_R_p * R_p_SI
#    dR_p_dm = one_third * R_p_SI * ( m_p_inv_SI - drho_dm_over_rho) # in SI
#    dR_p_dm_over_R = dR_p_dm / R_p_SI
    
    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_w) # in SI - no unit
    
    vH = compute_vant_Hoff_factor_NaCl(w_s)
    dvH_dws = compute_dvH_dws(w_s)
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
compute_mass_rate_derivative =\
njit()(compute_mass_rate_derivative_np)
compute_mass_rate_derivative_par =\
njit(parallel = True)(compute_mass_rate_derivative_np)

# return mass rate in fg/s and mass rate deriv in SI: 1/s
def compute_mass_rate_and_derivative_np(m_w, m_s, w_s, R_p, T_p, rho_p,
                                        T_amb, p_amb, S_amb, e_s_amb,
                                        L_v, K, D_v, sigma_w):
#    R_p_SI = 1.0E-6 * R_p # in SI: meter   
    
    # thermal size correction in SI
    l_alpha_plus_R_p = 1.0E-6 * (R_p + compute_l_alpha_lin(T_amb, p_amb, K))
    # diffusive size correction in SI
    l_beta_plus_R_p = 1.0E-6 * (R_p + compute_l_beta_lin(T_amb, D_v) )
       
    m_p_inv_SI = 1.0E18 / (m_w + m_s) # in 1/kg
    # dont use piecewise for now to avoid discontinuity in density...
    drho_dm_over_rho = -w_s * m_p_inv_SI / rho_p\
                       * (par_sol_dens[1] + 2.0 * par_sol_dens[3] * w_s\
                          + par_sol_dens[4] * T_p )

    dR_p_dm_over_R_p = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = 1.0E-6 * dR_p_dm_over_R_p * R_p
    
    eps_k = compute_kelvin_argument(R_p, T_p, rho_p, sigma_w) # in SI - no unit
    
    vH = compute_vant_Hoff_factor_NaCl(w_s)
    dvH_dws = compute_dvH_dws(w_s)
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
compute_mass_rate_and_derivative =\
njit()(compute_mass_rate_and_derivative_np)
compute_mass_rate_and_derivative_par =\
njit(parallel = True)(compute_mass_rate_and_derivative_np)
 
##############################################################################
### integration
# mass:
# returns the difference dm_w = m_w_n+1 - m_w_n during condensation/evaporation
# during one timestep using linear implicit explicit euler
# masses in femto gram    
# IN WORK: NOT UPDATED TO NUMBA
# def compute_delta_water_liquid_imex_linear( dt_, mass_water_, mass_solute_, 
#                                                   temperature_particle_,
#                                                   amb_temp_, amb_press_,
#                                                   amb_sat_, amb_sat_press_,
#                                                   diffusion_constant_,
#                                                   thermal_conductivity_air_,
#                                                   specific_heat_capacity_air_,
#                                                   surface_tension_,
#                                                   adiabatic_index_,
#                                                   accomodation_coefficient_,
#                                                   condensation_coefficient_, 
#                                                   heat_of_vaporization_,
#                                                   verbose = False):

#     dt_left = dt_
# #    dt = dt_
#     mass_water_new = mass_water_
    
#     mass_fraction_solute_effl = compute_efflorescence_mass_fraction_NaCl(
#                                     temperature_particle_)
    
#     while (dt_left > 0.0):
#         m_p = mass_water_new + mass_solute_
#         w_s = mass_solute_ / m_p
#         rho = compute_density_particle(w_s, temperature_particle_)
#         R = compute_radius_from_mass(m_p, rho)
#         # mass_rate = compute_mass_rate_from_water_mass_Szumowski(
#         #                 mass_water_new, mass_solute_, #  in femto gram
#         #                 temperature_particle_,
#         #                 amb_temp_, amb_press_,
#         #                 amb_sat_, amb_sat_press_,
#         #                 diffusion_constant_,
#         #                 thermal_conductivity_air_,
#         #                 specific_heat_capacity_air_,
#         #                 adiabatic_index_,
#         #                 accomodation_coefficient_,
#         #                 condensation_coefficient_, 
#         #                 heat_of_vaporization_)
#         # # masses in femto gram
#         # mass_rate_derivative = compute_mass_rate_derivative_Szumowski(
#         #                            mass_water_new, mass_solute_,
#         #                            temperature_particle_,
#         #                            amb_temp_, amb_press_,
#         #                            amb_sat_, amb_sat_press_,
#         #                            diffusion_constant_,
#         #                            thermal_conductivity_air_,
#         #                            specific_heat_capacity_air_,
#         #                            adiabatic_index_,
#         #                            accomodation_coefficient_,
#         #                            condensation_coefficient_, 
#         #                            heat_of_vaporization_)
#         mass_rate, mass_rate_derivative\
#             = compute_mass_rate_and_mass_rate_derivative_Szumowski(
#                     mass_water_, mass_solute_,
#                     m_p, w_s, R,
#                     temperature_particle_, rho,
#                     amb_temp_, amb_press_,
#                     amb_sat_, amb_sat_press_,
#                     diffusion_constant_,
#                     thermal_conductivity_air_,
#                     specific_heat_capacity_air_,
#                     heat_of_vaporization_,
#                     surface_tension_,
#                     adiabatic_index_,
#                     accomodation_coefficient_,
#                     condensation_coefficient_)
#         if (verbose):
#             print('mass_rate, mass_rate_derivative:')
#             print(mass_rate, mass_rate_derivative)
#         # safety to avoid (1 - dt/2 * f'(m_n)) going to zero
#         if mass_rate_derivative * dt_ < 1.0:
#             dt = dt_left
#             dt_left = -1.0
#         else:
#             dt = 1.0 / mass_rate_derivative
#             dt_left -= dt
    
#         mass_water_new +=  mass_rate * dt\
#                            / ( 1.0 - 0.5 * mass_rate_derivative * dt )
        
#         mass_water_effl = mass_solute_\
#                               * (1.0 / mass_fraction_solute_effl - 1.0)
        
# #        mass_fraction_solute_new =\
# #            mass_solute_ / (mass_water_new + mass_solute_)
        
# #        if (mass_fraction_solute_new > mass_fraction_solute_effl
# #            or mass_water_new < 0.0):
#         if (mass_water_new  < mass_water_effl):
# #            mass_water_new = mass_solute_\
# #                             * (1.0 / mass_fraction_solute_effl - 1.0)
#             mass_water_new = mass_water_effl
#             dt_left = -1.0
# #            print('w_s_effl reached')
    
#     return mass_water_new - mass_water_

### NEW 04.05.2019
# Newton method with no_iter iterations, the derivative is calculated only once
# IN WORK: might return gamma and not gamma0 for particle heat,
# but this is not important right now
def compute_dml_and_gamma_impl_Newton_lin_np(
        dt_sub, no_iter, m_w, m_s, w_s, R_p, T_p, rho_p,
        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w):
    
    w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
                             T_p)
    m_w_effl = m_s * (w_s_effl_inv - 1.0)
    gamma0, dgamma_dm = compute_mass_rate_and_derivative(
                            m_w, m_s, w_s, R_p, T_p, rho_p,
                            T_amb, p_amb, S_amb, e_s_amb,
                            L_v, K, D_v, sigma_w)
#    no_iter = 3
    dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                         1.0 / (1.0 - dt_sub_times_dgamma_dm),
                         np.ones_like(dt_sub_times_dgamma_dm)*10.0)
#    if (dt_sub_ * dgamma_dm < 0.9):
#        denom_inv = 
#    else:
#        denom_inv = 10.0
     
    mass_new = np.maximum(m_w_effl, m_w + dt_sub * gamma0 * denom_inv)
    
    for cnt in range(no_iter-1):
        m_p = mass_new + m_s
        w_s = m_s / m_p
        rho = compute_density_particle(w_s, T_p)
        R = compute_radius_from_mass(m_p, rho)
        gamma = compute_mass_rate(
                    mass_new, m_s, w_s, R, T_p, rho,
                    T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w)
                    
        mass_new += ( dt_sub * gamma + m_w - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - m_w, gamma0
compute_dml_and_gamma_impl_Newton_lin =\
    njit()(compute_dml_and_gamma_impl_Newton_lin_np)
compute_dml_and_gamma_impl_Newton_lin_par =\
njit(parallel = True)(compute_dml_and_gamma_impl_Newton_lin_np)

# NEW 04.05.19
# Full Newton method with no_iter iterations,
# the derivative is calculated every iteration
def compute_dml_and_gamma_impl_Newton_full_np(
        dt_sub, Newton_iter, m_w, m_s, w_s, R_p, T_p, rho_p,
        T_amb, p_amb, S_amb, e_s_amb, L_v, K, D_v, sigma_w):
    w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
                             T_p)
    m_w_effl = m_s * (w_s_effl_inv - 1.0)
    
    gamma0, dgamma_dm = compute_mass_rate_and_derivative(
                            m_w, m_s, w_s, R_p, T_p, rho_p,
                            T_amb, p_amb, S_amb, e_s_amb,
                            L_v, K, D_v, sigma_w)
#    Newton_iter = 3
    dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                         1.0 / (1.0 - dt_sub_times_dgamma_dm),
                         np.ones_like(dt_sub_times_dgamma_dm) * 10.0)
#    if (dt_sub_ * dgamma_dm < 0.9):
#        denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
#    else:
#        denom_inv = 10.0
     
    mass_new = np.maximum(m_w_effl, m_w + dt_sub * gamma0 * denom_inv)
    
    for cnt in range(Newton_iter-1):
        m_p = mass_new + m_s
        w_s = m_s / m_p
        rho = compute_density_particle(w_s, T_p)
        R = compute_radius_from_mass(m_p, rho)
        gamma, dgamma_dm = compute_mass_rate_and_derivative(
                               mass_new, m_s, w_s, R, T_p, rho,
                               T_amb, p_amb, S_amb, e_s_amb,
                               L_v, K, D_v, sigma_w)
                               
        dt_sub_times_dgamma_dm = dt_sub * dgamma_dm
        denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                             1.0 / (1.0 - dt_sub_times_dgamma_dm),
                     np.ones_like(dt_sub_times_dgamma_dm) * 10.0)
#        if (dt_sub_ * dgamma_dm < 0.9):
#            denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
#        else:
#            denom_inv = 10.0
        mass_new += ( dt_sub * gamma + m_w - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - m_w, gamma0
compute_dml_and_gamma_impl_Newton_full =\
njit()(compute_dml_and_gamma_impl_Newton_full_np)
compute_dml_and_gamma_impl_Newton_full_par =\
njit(parallel = True)(compute_dml_and_gamma_impl_Newton_full_np)

#%% NOT UPDATED WITH NUMBA
# NOT UPDATED WITH NUMBA
# def compute_delta_water_liquid_and_mass_rate_implicit_Newton_full_const_l(
#         dt_sub_, Newton_iter_, mass_water_, mass_solute_,
#         mass_particle_, mass_fraction_solute_, radius_particle_,
#         temperature_particle_, density_particle_,
#         amb_temp_, amb_press_,
#         amb_sat_, amb_sat_press_,
#         diffusion_constant_,
#         thermal_conductivity_air_,
#         specific_heat_capacity_air_,
#         heat_of_vaporization_,
#         surface_tension_,
#         adiabatic_index_,
#         accomodation_coefficient_,
#         condensation_coefficient_):
#     w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
#                              temperature_particle_)
#     m_w_effl = mass_solute_ * (w_s_effl_inv - 1.0)
    
#     gamma0, dgamma_dm =\
#         compute_mass_rate_and_mass_rate_derivative_Szumowski_const_l(
#             mass_water_, mass_solute_,
#             mass_particle_, mass_fraction_solute_, radius_particle_,
#             temperature_particle_, density_particle_,
#             amb_temp_, amb_press_,
#             amb_sat_, amb_sat_press_,
#             diffusion_constant_,
#             thermal_conductivity_air_,
#             specific_heat_capacity_air_,
#             heat_of_vaporization_,
#             surface_tension_,
#             adiabatic_index_,
#             accomodation_coefficient_,
#             condensation_coefficient_)
# #    Newton_iter = 3
#     dt_sub_times_dgamma_dm = dt_sub_ * dgamma_dm
#     denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
#                          1.0 / (1.0 - dt_sub_times_dgamma_dm),
#                          10.0)
# #    if (dt_sub_ * dgamma_dm < 0.9):
# #        denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
# #    else:
# #        denom_inv = 10.0
     
#     mass_new = np.maximum(m_w_effl,
#                           mass_water_ + dt_sub_ * gamma0 * denom_inv)
    
#     for cnt in range(no_iter_-1):
#         m_p = mass_new + mass_solute_
#         w_s = mass_solute_ / m_p
#         rho = compute_density_particle(w_s, temperature_particle_)
#         R = compute_radius_from_mass(m_p, rho)
#         gamma, dgamma_dm =\
#             compute_mass_rate_and_mass_rate_derivative_Szumowski_const_l(
#                 mass_new, mass_solute_,
#                 m_p, w_s, R,
#                 temperature_particle_, rho,
#                 amb_temp_, amb_press_,
#                 amb_sat_, amb_sat_press_,
#                 diffusion_constant_,
#                 thermal_conductivity_air_,
#                 specific_heat_capacity_air_,
#                 heat_of_vaporization_,
#                 surface_tension_,
#                 adiabatic_index_,
#                 accomodation_coefficient_,
#                 condensation_coefficient_)
#         dt_sub_times_dgamma_dm = dt_sub_ * dgamma_dm
#         denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
#                              1.0 / (1.0 - dt_sub_times_dgamma_dm),
#                      10.0)
# #        if (dt_sub_ * dgamma_dm < 0.9):
# #            denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
# #        else:
# #            denom_inv = 10.0
#         mass_new += ( dt_sub_* gamma + mass_water_ - mass_new) * denom_inv
#         mass_new = np.maximum( m_w_effl, mass_new )
        
#     return mass_new - mass_water_, gamma0

# NOT UPDATED WITH NUMBA
# def compute_delta_water_liquid_and_mass_rate_implicit_Newton_inverse_full(
#         dt_sub_, no_iter_, mass_water_, mass_solute_,
#         mass_particle_, mass_fraction_solute_, radius_particle_,
#         temperature_particle_, density_particle_,
#         amb_temp_, amb_press_,
#         amb_sat_, amb_sat_press_,
#         diffusion_constant_,
#         thermal_conductivity_air_,
#         specific_heat_capacity_air_,
#         heat_of_vaporization_,
#         surface_tension_,
#         adiabatic_index_,
#         accomodation_coefficient_,
#         condensation_coefficient_):
#     w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
#                              temperature_particle_)
#     m_w_effl = mass_solute_ * (w_s_effl_inv - 1.0)
    
#     gamma0, dgamma_dm = compute_mass_rate_and_mass_rate_derivative_Szumowski(
#                             mass_water_, mass_solute_,
#                             mass_particle_, mass_fraction_solute_,
#                             radius_particle_,
#                             temperature_particle_, density_particle_,
#                             amb_temp_, amb_press_,
#                             amb_sat_, amb_sat_press_,
#                             diffusion_constant_,
#                             thermal_conductivity_air_,
#                             specific_heat_capacity_air_,
#                             heat_of_vaporization_,
#                             surface_tension_,
#                             adiabatic_index_,
#                             accomodation_coefficient_,
#                             condensation_coefficient_)
# #    mass_new = mass_water_
# #    no_iter = 3
#     dgamma_factor = dt_sub_ * dgamma_dm
#     # dgamma_factor = dt_sub * F'(m) * m
#     dgamma_factor = np.where(dgamma_factor < 0.9,
#                              mass_water_ * (1.0 - dgamma_factor),
#                              mass_water_ * 0.1)
# #    if (dt_sub_ * dgamma_dm < 0.9):
# #        denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
# #    else:
# #        denom_inv = 10.0
    
#     print("iter = 1",
#           mass_water_ * dgamma_factor / (dgamma_factor - gamma0 * dt_sub_))
#     mass_new = np.maximum( m_w_effl,
#                            mass_water_ * dgamma_factor\
#                            / (dgamma_factor - gamma0 * dt_sub_) )
#     print("iter = 1", mass_new)
    
#     for cnt in range(no_iter_-1):
#         m_p = mass_new + mass_solute_
#         w_s = mass_solute_ / m_p
#         rho = compute_density_particle(w_s, temperature_particle_)
#         R = compute_radius_from_mass(m_p, rho)
#         gamma, dgamma_dm =\
#             compute_mass_rate_and_mass_rate_derivative_Szumowski(
#                                mass_new, mass_solute_,
#                                m_p, w_s, R,
#                                temperature_particle_, rho,
#                                amb_temp_, amb_press_,
#                                amb_sat_, amb_sat_press_,
#                                diffusion_constant_,
#                                thermal_conductivity_air_,
#                                specific_heat_capacity_air_,
#                                heat_of_vaporization_,
#                                surface_tension_,
#                                adiabatic_index_,
#                                accomodation_coefficient_,
#                                condensation_coefficient_)
#         dgamma_factor = dt_sub_ * dgamma_dm
#         # dgamma_factor = dt_sub * F'(m) * m
#         dgamma_factor = np.where(dgamma_factor < 0.9,
#                                  mass_new * (1.0 - dgamma_factor),
#                                  mass_new * 0.1)
# #        mass_new *= ( dt_sub_* gamma + mass_water_ - mass_new) * denom_inv
#         print("iter = ", cnt + 2 ,
#               mass_new * dgamma_factor\
#               / ( dgamma_factor - gamma * dt_sub_ + mass_new - mass_water_ ))
#         mass_new = np.maximum( m_w_effl,
#                                mass_new * dgamma_factor\
#                                / ( dgamma_factor - gamma * dt_sub_
#                                    + mass_new - mass_water_ ) )
#         print("iter = ", cnt+2, mass_new)
#     return mass_new - mass_water_, gamma0
    
# returns the difference dm_w = m_w_n+1 - m_w_n
# during condensation/evaporation
# during one timestep using linear implicit explicit euler
# also: returns mass_rate
# NOT UPDATED WITH NUMBA
# def compute_delta_water_liquid_and_mass_rate_imex_linear(
#         dt_, mass_water_, mass_solute_, #  in femto gram
#         temperature_particle_,
#         amb_temp_, amb_press_,
#         amb_sat_, amb_sat_press_,
#         diffusion_constant_,
#         thermal_conductivity_air_,
#         specific_heat_capacity_air_,
#         adiabatic_index_,
#         accomodation_coefficient_,
#         condensation_coefficient_, 
#         heat_of_vaporization_,
#         verbose = False):

#     dt_left = dt_
# #    dt = dt_
#     mass_water_new = mass_water_
    
#     mass_fraction_solute_effl = compute_efflorescence_mass_fraction_NaCl(
#                                     temperature_particle_)
    
#     while (dt_left > 0.0):
#         mass_rate = compute_mass_rate_from_water_mass_Szumowski(
#                         mass_water_new, mass_solute_, #  in femto gram
#                         temperature_particle_,
#                         amb_temp_, amb_press_,
#                         amb_sat_, amb_sat_press_,
#                         diffusion_constant_,
#                         thermal_conductivity_air_,
#                         specific_heat_capacity_air_,
#                         adiabatic_index_,
#                         accomodation_coefficient_,
#                         condensation_coefficient_, 
#                         heat_of_vaporization_)
#         mass_rate_derivative =\
#             compute_mass_rate_derivative_Szumowski_numerical(
                  # in femto gram
                  # mass_water_new, mass_solute_,
                  # temperature_particle_,
                  # amb_temp_, amb_press_,
                  # amb_sat_, amb_sat_press_,
                  # diffusion_constant_,
                  # thermal_conductivity_air_,
                  # specific_heat_capacity_air_,
                  # adiabatic_index_,
                  # accomodation_coefficient_,
                  # condensation_coefficient_, 
                  # heat_of_vaporization_
                  # )
#         if (verbose):
#             print('mass_rate, mass_rate_derivative:')
#             print(mass_rate, mass_rate_derivative)
#         # safety to avoid (1 - dt/2 * f'(m_n)) going to zero
#         if mass_rate_derivative * dt_ < 1.0:
#             dt = dt_left
#             dt_left = -1.0
#         else:
#             dt = 1.0 / mass_rate_derivative
#             dt_left -= dt
    
#         mass_water_new += mass_rate * dt\
#                           / ( 1.0 - 0.5 * mass_rate_derivative * dt )
        
#         mass_water_effl = mass_solute_\
#                           * (1.0 / mass_fraction_solute_effl - 1.0)
        
# #        mass_fraction_solute_new = mass_solute_\
# #           / (mass_water_new + mass_solute_)
        
# #        if (mass_fraction_solute_new >
# #             mass_fraction_solute_effl or mass_water_new < 0.0):
#         if (mass_water_new  < mass_water_effl):
# # mass_water_new = mass_solute_ * (1.0 / mass_fraction_solute_effl - 1.0)
#             mass_water_new = mass_water_effl
#             dt_left = -1.0
# #            print('w_s_effl reached')
    
#     return mass_water_new - mass_water_, mass_rate

# returns the difference dm_w = m_w_n+1 - m_w_n
# during condensation/evaporation
# during one timestep using linear implicit euler
# masses in femto gram
# NOT UPDATED WITH NUMBA
# def compute_delta_water_liquid_implicit_linear( dt_, mass_water_,
#                                                 mass_solute_,
#                                                 temperature_particle_,
#                                                 amb_temp_, amb_press_,
#                                                 amb_sat_, amb_sat_press_,
#                                                 diffusion_constant_,
#                                                 thermal_conductivity_air_,
#                                                 specific_heat_capacity_air_,
#                                                 adiabatic_index_,
#                                                 accomodation_coefficient_,
#                                                 condensation_coefficient_, 
#                                                 heat_of_vaporization_,
#                                                 verbose = False):

#     dt_left = dt_
# #    dt = dt_
#     mass_water_new = mass_water_
    
#     mass_fraction_solute_effl = compute_efflorescence_mass_fraction_NaCl(
#                                     temperature_particle_)
    
#     surface_tension_ = compute_surface_tension_water(temperature_particle_)
    
#     while (dt_left > 0.0):
# #        mass_rate =\
# #            compute_mass_rate_from_water_mass_Szumowski(
                    # mass_water_new, mass_solute_, #  in femto gram
                    #                   temperature_particle_,
                    #                   amb_temp_, amb_press_,
                    #                   amb_sat_, amb_sat_press_,
                    #                   diffusion_constant_,
                    #                   thermal_conductivity_air_,
                    #                   specific_heat_capacity_air_,
                    #                   adiabatic_index_,
                    #                   accomodation_coefficient_,
                    #                   condensation_coefficient_, 
                    #                   heat_of_vaporization_)
#         m_p = mass_water_new + mass_solute_
#         w_s = mass_solute_ / m_p
#         rho = compute_density_particle(w_s, temperature_particle_)
#         R = compute_radius_from_mass(m_p, rho)
#         mass_rate, mass_rate_derivative =\
#             compute_mass_rate_and_mass_rate_derivative_Szumowski(
#                 mass_water_new, mass_solute_,
#                 m_p, w_s, R,
#                 temperature_particle_, rho,
#                 amb_temp_, amb_press_,
#                 amb_sat_, amb_sat_press_,
#                 diffusion_constant_,
#                 thermal_conductivity_air_,
#                 specific_heat_capacity_air_,
#                 heat_of_vaporization_,
#                 surface_tension_,
#                 adiabatic_index_,
#                 accomodation_coefficient_,
#                 condensation_coefficient_)
#         if (verbose):
#             print('mass_rate, mass_rate_derivative:')
#             print(mass_rate, mass_rate_derivative)
#         if mass_rate_derivative * dt_ < 0.5:
#             dt = dt_left
#             dt_left = -1.0
#         else:
#             dt = 0.5 / mass_rate_derivative
#             dt_left -= dt
    
#         mass_water_new += mass_rate * dt\
#                           / ( 1.0 - mass_rate_derivative * dt )
        
#         mass_water_effl = mass_solute_\
#                           * (1.0 / mass_fraction_solute_effl - 1.0)
        
# #        mass_fraction_solute_new =\
# #  mass_solute_ / (mass_water_new + mass_solute_)
        
# #        if (mass_fraction_solute_new
# #            > mass_fraction_solute_effl or mass_water_new < 0.0):
#         if (mass_water_new  < mass_water_effl):
# #            mass_water_new = mass_solute_\
# #                             * (1.0 / mass_fraction_solute_effl - 1.0)
#             mass_water_new = mass_water_effl
#             dt_left = -1.0
# #            print('w_s_effl reached')
    
#     return mass_water_new - mass_water_

# def compute_mass_rate_from_surface_partial_pressure(amb_temp_,
#                                                     amb_sat_, amb_sat_press_,
#                                                     diffusion_constant_,
#                                                     radius_,
#                                                     surface_partial_pressure_,
#                                                     particle_temperature_,
#                                                     ):
#     return 4.0E12 * np.pi * radius_ * diffusion_constant_\
#            / c.specific_gas_constant_water_vapor \
#            * ( amb_sat_ * amb_sat_press_ / amb_temp_
#                - surface_partial_pressure_ / particle_temperature_ )
           


    
    