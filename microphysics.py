#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:28:02 2019

@author: jdesk
"""


import math.sqrt
import numpy as np
from scipy.optimize import fminbound
from scipy.optimize import brentq

import constants as c
from atmosphere import compute_surface_tension_water,\
                       compute_specific_heat_capacity_air_moist,\
                       compute_diffusion_constant,\
                       compute_thermal_conductivity_air
                       
### conversions

# compute mass in femto gram = 10^-18 kg 
# from radius in microns 
# and density in kg/m^3
def compute_mass_from_radius(radius_, density_):
    return c.pi_times_4_over_3 * density_ * radius_ * radius_ * radius_ 

# compute radius in microns
# mass in 10^-18 kg, density in kg/m^3, radius in micro meter
def compute_radius_from_mass(mass_, density_):
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
par_water_dens = [  1.00013502e+03,  -4.68112708e-03,   2.72389977e+02]
def compute_density_water(temperature_):
    return par_water_dens[0]\
         + par_water_dens[1] * (temperature_ - par_water_dens[2])**2

# NaCl solution density in kg/m^3
# quad. fit to data from CRC 2005
# for w_s < 0.226:
# relative error is below 0.1 % for range 0 .. 50 °C
#                   below 0.17 % in range 0 .. 60 °C
# (only high w_s lead to high error)
par_sol_dens = [  7.619443952135e+02,   1.021264281453e+03,
                  1.828970151543e+00, 2.405352122804e+02,
                 -1.080547892416e+00,  -3.492805028749e-03 ]
def compute_density_NaCl_solution(mass_fraction_solute_, temperature_):
    return    par_sol_dens[0] \
            + par_sol_dens[1] * mass_fraction_solute_ \
            + par_sol_dens[2] * temperature_ \
            + par_sol_dens[3] * mass_fraction_solute_ * mass_fraction_solute_ \
            + par_sol_dens[4] * mass_fraction_solute_ * temperature_ \
            + par_sol_dens[5] * temperature_ * temperature_

#    approx. density of the particle (droplet)
#    Combine the two density functions
#    Use water density if mass fraction < 0.001,
#    then rel dev of the two density functions is < 0.1 %
#    For now, use rho(w_s,T) for all ranges,
#    no if then else... to avoid discontinuity in the density
w_s_rho_p = 0.001
def compute_density_particle(mass_fraction_solute_, temperature_):
    return compute_density_NaCl_solution( mass_fraction_solute_, temperature_ )
#    return compute_density_water(temperature_)
#    return np.where( mass_fraction_solute_ < w_s_rho_p,
#                    compute_density_water(temperature_),
#                    compute_density_NaCl_solution(mass_fraction_solute_,
#                                        temperature_))
    
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
par_solub = [  3.77253081e-01,  -8.68998172e-04,   1.64705858e-06]
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
par_vH_NaCl = [ 1.55199086,  4.95679863]
def compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_):
    return par_vH_NaCl[0] + par_vH_NaCl[1] * mass_fraction_solute_

vant_Hoff_factor_NaCl_const = 2.0

molar_mass_ratio_w_NaCl = c.molar_mass_water/c.molar_mass_NaCl

#mf_cross_NaCl = 0.090382759623349337
mf_cross_NaCl = 0.09038275962335

def compute_vant_Hoff_factor_NaCl(mass_fraction_solute_):
#    return compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_)
#    return vant_Hoff_factor_NaCl_const
    return np.where(mass_fraction_solute_ < mf_cross_NaCl,
                    vant_Hoff_factor_NaCl_const,
                    compute_vant_Hoff_factor_NaCl_fit(mass_fraction_solute_))

##############################################################################
### forces
           
# Particle Reynolds number as given in Sommerfeld 2008
# radius in mu (1E-6 m)
# velocity dev = |u_f-v_p| in m/s
# density in kg/m^3
# viscosity in N s/m^2
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
    
# in particle class:
# calc m_s, m_w, w_s, R_p first => vant_Hoff(w_s), a_w (w_s, T_p)
# IN WORK
def compute_water_activity(mass_fraction_solute_, vant_Hoff_):
#    vant_Hoff = np.where( mass_fraction_solute_ < mf_cross,
#                          2.0,
#                          self.compute_vant_Hoff_factor_fit_init(mass_fraction_solute_))
#    return (1 - mass_fraction_solute_)\
#         / ( 1 - ( 1 - molar_mass_ratio_w_NaCl * compute_vant_Hoff_factor_NaCl(mass_fraction_solute_) ) * mass_fraction_solute_ )
    return ( 1 - mass_fraction_solute_ )\
         / ( 1 - ( 1 - molar_mass_ratio_w_NaCl * vant_Hoff_ ) \
                 * mass_fraction_solute_ )

def compute_water_activity_mf(mass_fraction_solute_, vant_Hoff_):
#    vant_Hoff = np.where( mass_fraction_solute_ < mf_cross,
#                          2.0,
#                          self.compute_vant_Hoff_factor_fit_init(mass_fraction_solute_))
#    return (1 - mass_fraction_solute_)\
#         / ( 1 - ( 1 - molar_mass_ratio_w_NaCl * compute_vant_Hoff_factor_NaCl(mass_fraction_solute_) ) * mass_fraction_solute_ )
    return (1 - mass_fraction_solute_)\
         / ( 1 - ( 1 - molar_mass_ratio_w_NaCl * vant_Hoff_ )
                 * mass_fraction_solute_ )

# IN WORK
# factor E6 comes from conversion mu -> m
def compute_kelvin_term(radius_particle_,
                        temperature_,
                        mass_density_particle_,
                        surface_tension_ ):
    return np.exp(
                   2.0E6 * surface_tension_
                   / ( c.specific_gas_constant_water_vapor * temperature_
                       * mass_density_particle_* radius_particle_)
                 )

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
def compute_kelvin_raoult_term(radius_particle_,
                                   mass_fraction_solute_,
                                   temperature_,
                                   vant_Hoff_,
                                   mass_density_particle_,
                                   surface_tension_):
    return   compute_water_activity(mass_fraction_solute_, vant_Hoff_) \
           * compute_kelvin_term(radius_particle_,
                               temperature_,
                               mass_density_particle_,
                               surface_tension_)
# IN WORK
def compute_kelvin_raoult_term_NaCl(radius_particle_,
                                        mass_fraction_solute_,
                                        temperature_):
    return compute_kelvin_raoult_term(
               radius_particle_,
               mass_fraction_solute_,
               temperature_,
               compute_vant_Hoff_factor_NaCl( mass_fraction_solute_ ),
               compute_density_particle(mass_fraction_solute_, temperature_),
               compute_surface_tension_water(temperature_))

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

def compute_initial_mass_fraction_solute_NaCl(radius_dry_,
                                              ambient_saturation_,
                                              ambient_temperature_,
                                              opt = 'None'):
    # 0.
    m_s = compute_mass_from_radius(radius_dry_, c.mass_density_NaCl_dry)
    w_s_effl = compute_efflorescence_mass_fraction_NaCl(ambient_temperature_)
    # 1.
    S_effl = compute_kelvin_raoult_term_NaCl_mf(w_s_effl,
                                                ambient_temperature_, m_s)
    # 2.
    if ambient_saturation_ <= S_effl:
        w_s_init = w_s_effl
    else:
        # 3.
        w_s_act, S_act, flag, nofc  = \
            fminbound(lambda w: -compute_kelvin_raoult_term_NaCl_mf(
                                                w, ambient_temperature_, m_s),
                      x1=1E-8, x2=w_s_effl, xtol = 1.0E-12, full_output=True )
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
                    lambda w: compute_kelvin_raoult_term_NaCl_mf(
                                  w, ambient_temperature_, m_s)\
                              - ambient_saturation_,
                    w_s_act,
                    w_s_effl,
                    xtol = 1e-15,
                    full_output=True)
            if solve_result[1].converged:
                w_s_init = solve_result[0]
    #         solute_mass_fraction
    # = brentq(droplet.compute_kelvin_raoult_term_mf_init,
    #            mf_max, mf_del, args = S_a)
            else:
                w_s_init = w_s_act        
    
    if opt == 'verbose':
        w_s_act, S_act, flag, nofc  = \
            fminbound(lambda w: -compute_kelvin_raoult_term_NaCl_mf(
                                    w, ambient_temperature_, m_s),
                      x1=1E-8, x2=w_s_effl, xtol = 1.0E-12, full_output=True )
        S_act = -S_act
        return w_s_init, w_s_act, S_act
    else:
        return w_s_init





##############################################################################
### mass rate
accommodation_coeff = 1.0
adiabatic_index_inv = 0.7142857142857143
def compute_thermal_size_correction_Fukuta_short(amb_temp_, amb_press_,
                                                 thermal_cond_air_):
    return thermal_cond_air_ * np.sqrt(2.0 * np.pi
                                       * c.specific_gas_constant_air_dry
                                       * amb_temp_) \
        / ( accommodation_coeff * amb_press_
           * (c.specific_heat_capacity_air_dry_NTP * adiabatic_index_inv
              + 0.5 * c.specific_gas_constant_air_dry) )

T_alpha_0 = 289 # K
c_alpha_1 =\
    math.sqrt(2.0 * np.pi * c.specific_gas_constant_air_dry * T_alpha_0 )\
    / ( accommodation_coeff\
       * (c.specific_heat_capacity_air_dry_NTP * adiabatic_index_inv 
          + 0.5 * c.specific_gas_constant_air_dry) )
c_alpha_2 =\
    0.5 * math.sqrt(2.0 * np.pi * c.specific_gas_constant_air_dry / T_alpha_0 )\
    / ( accommodation_coeff
       * (c.specific_heat_capacity_air_dry_NTP * adiabatic_index_inv
          + 0.5 * c.specific_gas_constant_air_dry) )
def compute_thermal_size_correction_Fukuta_lin(amb_temp_, amb_press_,
                                               thermal_cond_air_):
    return ( c_alpha_1 + c_alpha_2 * (amb_temp_ - T_alpha_0) )\
           * thermal_cond_air_ / amb_press_

def compute_thermal_size_correction_Fukuta2(amb_temp_, amb_press_,
                                            thermal_cond_air_, r_v_):
    return thermal_cond_air_\
           * np.sqrt(2.0 * np.pi * c.specific_gas_constant_air_dry * amb_temp_) \
        / ( accommodation_coeff * amb_press_
           * ( compute_specific_heat_capacity_air_moist(r_v_) 
               * 0.7142857142857143 + 0.5 * c.specific_gas_constant_air_dry) )
def compute_thermal_size_correction_Fukuta(amb_temp_, amb_press_,
                                           thermal_cond_air_,
                                           spec_heat_cap_air_,
                                           adiabatic_index_,
                                           accomm_coeff_):
    return thermal_cond_air_\
         * np.sqrt(2.0 * np.pi * c.specific_gas_constant_air_dry * amb_temp_) \
        / ( accomm_coeff_ * amb_press_\
           * (spec_heat_cap_air_ / adiabatic_index_\
              + 0.5 * c.specific_gas_constant_air_dry) )

# l_alpha, fixed T = 289 K, p = 900 hPa
thermal_size_correction_Fukuta_T289_p900 =\
    compute_thermal_size_correction_Fukuta_short(
            289.0, 9.0E4, compute_thermal_conductivity_air(289.0))

condensation_coeff = 0.0415
def compute_diffusive_size_correction_Fukuta_short(amb_temp_, diff_const_air_):
    return np.sqrt( 2.0 * np.pi * c.molar_mass_water
                   / ( c.universal_gas_constant * amb_temp_ ) ) \
            * diff_const_air_ / condensation_coeff
c_beta_1 = math.sqrt( 2.0 * np.pi * c.molar_mass_water
                     / ( c.universal_gas_constant * T_alpha_0 ) )\
           / condensation_coeff
c_beta_2 = -0.5 / T_alpha_0
def compute_diffusive_size_correction_Fukuta_lin(amb_temp_, diff_const_air_):
    return c_beta_1 * ( 1.0 + c_beta_2 * (amb_temp_ - T_alpha_0) )\
                    * diff_const_air_

def compute_diffusive_size_correction_Fukuta(amb_temp_, diff_const_air_,
                                             condens_coeff_):
    return np.sqrt( 2.0 * np.pi * c.molar_mass_water
                   / ( c.universal_gas_constant * amb_temp_ ) ) \
            * diff_const_air_ / condens_coeff_

# l_beta, fixed T = 289 K, p = 900 hPa
diffusive_size_correction_Fukuta_T289_p900\
    = compute_diffusive_size_correction_Fukuta_short(
            289.0, compute_diffusion_constant(289.0, 9.0E4))

# mass rate Szumowski
#     besides the other properties, we need the diffusion constant, which
#     is among others dependent on the ambient pressure
#     i.e. the diff. const. must be updated before the mass rate is computed
#     note that the E12 in 4.0E12 is coming from the unit conversion 
#     to get mass rate in 10^-18 kg/s = 1 fg/s (femtogram)
#     requires T_amb, S_amb, e_s_amb, D_v(p_amb, T_amb), K(p_amb, T_amb)
#     AND S_eq, L_v, e_ps, Tp, 
def compute_mass_rate_Szumowski_from_Seq(  amb_temp_, amb_press_,
                                  amb_sat_, amb_sat_press_,
                                  diffusion_constant_,
                                  thermal_conductivity_air_,
                                  specific_heat_capacity_air_,
                                  adiabatic_index_,
                                  accomodation_coefficient_,
                                  condensation_coefficient_, 
                                  radius_,
                                  heat_of_vaporization_,
                                  equilibrium_saturation_
                                  ):
    # thermal size correction
    l_alpha = compute_thermal_size_correction_Fukuta(amb_temp_, amb_press_,
                                                 thermal_conductivity_air_,
                                                 specific_heat_capacity_air_,
                                                 adiabatic_index_,
                                                 accomodation_coefficient_)
    # diffusive size correction
    l_beta = compute_diffusive_size_correction_Fukuta(
                amb_temp_,
                diffusion_constant_,
                condensation_coefficient_)
    thermal_term = heat_of_vaporization_ * heat_of_vaporization_\
                   * equilibrium_saturation_ \
        / ( thermal_conductivity_air_ * c.specific_gas_constant_water_vapor
            * amb_temp_ * amb_temp_ ) \
        * ( radius_ + l_alpha * 1.0E6)
    diffusion_term = c.specific_gas_constant_water_vapor * amb_temp_ \
                     / ( diffusion_constant_ * amb_sat_press_ )\
                     * (radius_ + l_beta * 1.0E6)
    return 4.0E12 * np.pi * radius_ * radius_\
           * (amb_sat_ - equilibrium_saturation_) \
           / (thermal_term + diffusion_term)            

#NEW
def compute_mass_rate_Szumowski(mass_water_, mass_solute_,
                                  mass_fraction_solute_, radius_particle_,
                                  density_particle_,
                                  amb_temp_, amb_press_,
                                  amb_sat_, amb_sat_press_,
                                  diffusion_constant_,
                                  thermal_conductivity_air_,
                                  specific_heat_capacity_air_,
                                  heat_of_vaporization_,
                                  surface_tension_,
                                  adiabatic_index_,
                                  accomodation_coefficient_,
                                  condensation_coefficient_):
    
    R_p_SI = 1.0E-6 * radius_particle_ # in SI: meter   
    # thermal size correction
    l_alpha_plus_R_p = R_p_SI\
        + compute_thermal_size_correction_Fukuta(amb_temp_, amb_press_,
                                                 thermal_conductivity_air_,
                                                 specific_heat_capacity_air_,
                                                 adiabatic_index_,
                                                 accomodation_coefficient_)
    # diffusive size correction
    l_beta_plus_R_p = R_p_SI\
        + compute_diffusive_size_correction_Fukuta(amb_temp_,
                                                   diffusion_constant_,
                                                   condensation_coefficient_)
    
    vH = compute_vant_Hoff_factor_NaCl(mass_fraction_solute_)
    a_k_over_R_p = 2.0E6 * surface_tension_ \
             / ( c.specific_gas_constant_water_vapor
                 * amb_temp_
                 * density_particle_* radius_particle_)# in SI -> no unit
    S_eq = np.exp(a_k_over_R_p) * mass_water_\
           / (mass_water_ + mass_solute_ * molar_mass_ratio_w_NaCl * vH)
    c1 = heat_of_vaporization_ * heat_of_vaporization_ \
         / ( thermal_conductivity_air_
             * c.specific_gas_constant_water_vapor * amb_temp_ * amb_temp_ )
    c2 = c.specific_gas_constant_water_vapor * amb_temp_\
         / (diffusion_constant_ * amb_sat_press_)
    return 4.0E6 * np.pi * radius_particle_ * radius_particle_\
           * (amb_sat_ - S_eq)\
           / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 )

# IN WORK: MODIFY WITH LINEAR SIZE CORRECTION
# modified WITH density derivative
def compute_mass_rate_and_mass_rate_derivative_Szumowski(
        mass_water_, mass_solute_,
        mass_particle_, mass_fraction_solute_, radius_particle_,
        temperature_particle_, density_particle_,
        amb_temp_, amb_press_,
        amb_sat_, amb_sat_press_,
        diffusion_constant_,
        thermal_conductivity_air_,
        specific_heat_capacity_air_,
        heat_of_vaporization_,
        surface_tension_,
        adiabatic_index_,
        accomodation_coefficient_,
        condensation_coefficient_):
    R_p_SI = 1.0E-6 * radius_particle_ # in SI: meter   
    
    # thermal size correction
#    l_alpha_plus_R_p = R_p_SI 
#+ compute_thermal_size_correction_Fukuta_lin(
#amb_temp_, amb_press_, thermal_conductivity_air_)
    l_alpha_plus_R_p = R_p_SI\
        + compute_thermal_size_correction_Fukuta(amb_temp_, amb_press_,
                                                 thermal_conductivity_air_,
                                                 specific_heat_capacity_air_,
                                                 adiabatic_index_,
                                                 accomodation_coefficient_)
    # diffusive size correction
#    l_beta_plus_R_p = R_p_SI 
#+ compute_diffusive_size_correction_Fukuta_lin(amb_temp_, diffusion_constant_)
    l_beta_plus_R_p = R_p_SI\
        + compute_diffusive_size_correction_Fukuta(amb_temp_,
                                                  diffusion_constant_,
                                                  condensation_coefficient_)
       
    m_p_inv_SI = 1.0E18 / mass_particle_ # in 1 / 1.0E-18 kg
    # dont use piecewise for now to avoid discontinuity in density...
    drho_dm_over_rho = -mass_fraction_solute_ * m_p_inv_SI / density_particle_\
                       * (par_sol_dens[1]
                          + 2.0 * par_sol_dens[3] * mass_fraction_solute_
                          + par_sol_dens[4] * temperature_particle_  )
#    drho_dm_over_rho = np.where(mass_fraction_solute_ < w_s_rho_p,
#                       0.0,
#                       -mass_fraction_solute_
#                   * m_p_inv_SI / density_particle_ \
#  * (par_sol_dens[1] 
# + 2.0*par_sol_dens[3]*mass_fraction_solute_
# + par_sol_dens[4]*temperature_particle_  ) ) 


    dR_p_dm_over_R = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = dR_p_dm_over_R * R_p_SI
#    dR_p_dm = one_third * R_p_SI * ( m_p_inv_SI - drho_dm_over_rho) # in SI
#    dR_p_dm_over_R = dR_p_dm / R_p_SI
        
    a_k_over_R_p = 2.0E6 * surface_tension_ \
                 / ( c.specific_gas_constant_water_vapor * amb_temp_
                    * density_particle_* radius_particle_)# in SI -> no unit
    
    vH = compute_vant_Hoff_factor_NaCl(mass_fraction_solute_)
    ###########
    dvH_dws = np.where(mass_fraction_solute_ < mf_cross_NaCl,
                       0.0, par_vH_NaCl[1])
#    dvH_dws = 0.0
#    dvH_dws = par_vH_NaCl[1]
    ##########
    # masses in 1.0E-18 kg
#    h1 = (mass_water_ + mass_solute_ * molar_mass_ratio_w_NaCl * vH) 
    # dont convert masses here
    h1_inv = 1.0 / (mass_water_ + mass_solute_ * molar_mass_ratio_w_NaCl * vH) 
        
    S_eq = mass_water_ * h1_inv * np.exp(a_k_over_R_p)
    
    dSeq_dm = S_eq * (1.0E18 / mass_water_
                      - a_k_over_R_p * ( dR_p_dm_over_R + drho_dm_over_rho )
                      - ( 1 - molar_mass_ratio_w_NaCl * dvH_dws
                          * mass_fraction_solute_ * mass_fraction_solute_)
                        * h1_inv * 1.0E18 )
    
#    f1 = 4.0 * np.pi * radius_particle_ * radius_particle_#too large by 1.0E12
    
    
#    df1_dm = 2.0 * f1 * dR_p_dm_over_R
    
    c1 = heat_of_vaporization_ * heat_of_vaporization_ \
         / ( thermal_conductivity_air_ * c.specific_gas_constant_water_vapor
             * amb_temp_ * amb_temp_ )
    c2 = c.specific_gas_constant_water_vapor * amb_temp_\
         / (diffusion_constant_ * amb_sat_press_)
    # in SI : m^2 s / kg:     
    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 )
    
    f1f3 = 4.0 * np.pi * R_p_SI * R_p_SI * f3 # SI
    
    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm )\
             * c1 + dR_p_dm * c2
    f2 = amb_sat_ - S_eq
    
#    gamma = f1 * f2 * f3 # in kg/s
#    dgamma_dm = f1*f3*( f2*( 2.0*dR_p_dm_over_R - f3*dg1_dm ) - dSeq_dm )
    # return 1.0E18 * gamma, dgamma_dm
    return 1.0E18 * f1f3 * f2,\
           f1f3 * ( f2 * ( 2.0 * dR_p_dm_over_R - f3 * dg1_dm ) - dSeq_dm )

def compute_mass_rate_and_mass_rate_derivative_Szumowski_lin_l(
        mass_water_, mass_solute_,
        mass_particle_, mass_fraction_solute_, radius_particle_,
        temperature_particle_, density_particle_,
        amb_temp_, amb_press_,
        amb_sat_, amb_sat_press_,
        diffusion_constant_,
        thermal_conductivity_air_,
        specific_heat_capacity_air_,
        heat_of_vaporization_,
        surface_tension_,
        adiabatic_index_,
        accomodation_coefficient_,
        condensation_coefficient_):
    
    R_p_SI = 1.0E-6 * radius_particle_ # in SI: meter   
    
    # thermal size correction
    l_alpha_plus_R_p = R_p_SI\
        + compute_thermal_size_correction_Fukuta_lin(amb_temp_, amb_press_,
                                                     thermal_conductivity_air_)
#    l_alpha_plus_R_p = R_p_SI\
#    + compute_thermal_size_correction_Fukuta(amb_temp_, amb_press_,
#                                             thermal_conductivity_air_,
#                                             specific_heat_capacity_air_,
#                                             adiabatic_index_,
#                                             accomodation_coefficient_)
    # diffusive size correction
    l_beta_plus_R_p = R_p_SI\
        + compute_diffusive_size_correction_Fukuta_lin(amb_temp_,
                                                       diffusion_constant_)
#    l_beta_plus_R_p = R_p_SI\
#   + compute_diffusive_size_correction_Fukuta(amb_temp_,
#                                                  diffusion_constant_,
#                                                  condensation_coefficient_)
       
    m_p_inv_SI = 1.0E18 / mass_particle_ # in 1 / 1.0E-18 kg
    # dont use piecewise for now to avoid discontinuity in density...
    drho_dm_over_rho = -mass_fraction_solute_ * m_p_inv_SI / density_particle_\
                       * ( par_sol_dens[1]
                           + 2.0 * par_sol_dens[3] * mass_fraction_solute_
                           + par_sol_dens[4] * temperature_particle_  )
#    drho_dm_over_rho = np.where(mass_fraction_solute_ < w_s_rho_p,
#                       0.0,
#                       -mass_fraction_solute_ * m_p_inv_SI
#                       / density_particle_ \
#   * (par_sol_dens[1] 
#   + 2.0*par_sol_dens[3]*mass_fraction_solute_
# + par_sol_dens[4]*temperature_particle_  ) ) 


    dR_p_dm_over_R = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = dR_p_dm_over_R * R_p_SI
#    dR_p_dm = one_third * R_p_SI * ( m_p_inv_SI - drho_dm_over_rho) # in SI
#    dR_p_dm_over_R = dR_p_dm / R_p_SI
        
    a_k_over_R_p = 2.0E6 * surface_tension_ \
                 / ( c.specific_gas_constant_water_vapor * amb_temp_
                    * density_particle_* radius_particle_)# in SI -> no unit
    
    vH = compute_vant_Hoff_factor_NaCl(mass_fraction_solute_)
    ###########
    dvH_dws = np.where(mass_fraction_solute_ < mf_cross_NaCl,
                       0.0, par_vH_NaCl[1])
#    dvH_dws = 0.0
#    dvH_dws = par_vH_NaCl[1]
    ##########
    # masses in 1.0E-18 kg
#    h1 = (mass_water_ + mass_solute_ * molar_mass_ratio_w_NaCl * vH) 
    # dont convert masses here
    h1_inv = 1.0 / (mass_water_ + mass_solute_ * molar_mass_ratio_w_NaCl * vH) 
        
    S_eq = mass_water_ * h1_inv * np.exp(a_k_over_R_p)
    
    dSeq_dm = S_eq * (1.0E18 / mass_water_
                      - a_k_over_R_p * ( dR_p_dm_over_R + drho_dm_over_rho )
                      - (1 - molar_mass_ratio_w_NaCl * dvH_dws
                         * mass_fraction_solute_ * mass_fraction_solute_)
                        * h1_inv * 1.0E18)
# too large by 1.0E12    
#    f1 = 4.0 * np.pi * radius_particle_ * radius_particle_ 
    
#    df1_dm = 2.0 * f1 * dR_p_dm_over_R
    
    c1 = heat_of_vaporization_ * heat_of_vaporization_ \
         / ( thermal_conductivity_air_ * c.specific_gas_constant_water_vapor
             * amb_temp_ * amb_temp_ )
    c2 = c.specific_gas_constant_water_vapor * amb_temp_\
         / (diffusion_constant_ * amb_sat_press_)
    # in SI : m^2 s / kg
    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1 + (l_beta_plus_R_p) * c2 ) 
    
    f1f3 = 4.0 * np.pi * R_p_SI * R_p_SI * f3 # SI
    
    dg1_dm = (dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1\
             + dR_p_dm * c2
    f2 = amb_sat_ - S_eq
    
#    gamma = f1 * f2 * f3 # in kg/s
#    dgamma_dm = f1*f3*( f2*( 2.0*dR_p_dm_over_R - f3*dg1_dm ) - dSeq_dm )
    # return 1.0E18 * gamma, dgamma_dm
    return 1.0E18 * f1f3 * f2,\
           f1f3 * ( f2 * ( 2.0 * dR_p_dm_over_R - f3 * dg1_dm ) - dSeq_dm )

def compute_mass_rate_and_mass_rate_derivative_Szumowski_const_l(
        mass_water_, mass_solute_,
        mass_particle_, mass_fraction_solute_, radius_particle_,
        temperature_particle_, density_particle_,
        amb_temp_, amb_press_,
        amb_sat_, amb_sat_press_,
        diffusion_constant_,
        thermal_conductivity_air_,
        specific_heat_capacity_air_,
        heat_of_vaporization_,
        surface_tension_,
        adiabatic_index_,
        accomodation_coefficient_,
        condensation_coefficient_):
    R_p_SI = 1.0E-6 * radius_particle_ # in SI: meter   
    
    # thermal size correction
    l_alpha_plus_R_p = R_p_SI + thermal_size_correction_Fukuta_T289_p900
#    l_alpha_plus_R_p = R_p_SI
#   + compute_thermal_size_correction_Fukuta(amb_temp_, amb_press_,
#                                                 thermal_conductivity_air_,
#                                                 specific_heat_capacity_air_,
#                                                 adiabatic_index_,
#                                                 accomodation_coefficient_)
    # diffusive size correction
    l_beta_plus_R_p = R_p_SI + diffusive_size_correction_Fukuta_T289_p900
#    l_beta_plus_R_p = R_p_SI\
#   + compute_diffusive_size_correction_Fukuta(amb_temp_,
#                                                  diffusion_constant_,
#                                                  condensation_coefficient_)
       
    m_p_inv_SI = 1.0E18 / mass_particle_ # in 1 / 1.0E-18 kg
    # dont use piecewise for now to avoid discontinuity in density...
    drho_dm_over_rho = -mass_fraction_solute_ * m_p_inv_SI / density_particle_\
                       * ( par_sol_dens[1]
                           + 2.0 * par_sol_dens[3] * mass_fraction_solute_
                           + par_sol_dens[4] * temperature_particle_ )
#    drho_dm_over_rho = np.where(
#           mass_fraction_solute_ < w_s_rho_p,
#                       0.0,
#           -mass_fraction_solute_ * m_p_inv_SI / density_particle_ \
#           * (par_sol_dens[1]
#              + 2.0*par_sol_dens[3]*mass_fraction_solute_
#              + par_sol_dens[4]*temperature_particle_  ) ) 


    dR_p_dm_over_R = c.one_third * ( m_p_inv_SI - drho_dm_over_rho)
    dR_p_dm = dR_p_dm_over_R * R_p_SI
#    dR_p_dm = one_third * R_p_SI * ( m_p_inv_SI - drho_dm_over_rho) # in SI
#    dR_p_dm_over_R = dR_p_dm / R_p_SI
        
    a_k_over_R_p = 2.0E6 * surface_tension_ \
                 / ( c.specific_gas_constant_water_vapor * amb_temp_
                     * density_particle_* radius_particle_)# in SI -> no unit
    
    vH = compute_vant_Hoff_factor_NaCl(mass_fraction_solute_)
    ###########
    dvH_dws = np.where(mass_fraction_solute_ < mf_cross_NaCl,
                       0.0, par_vH_NaCl[1])
#    dvH_dws = 0.0
#    dvH_dws = par_vH_NaCl[1]
    ##########
    # masses in 1.0E-18 kg
#    h1 = (mass_water_ + mass_solute_ * molar_mass_ratio_w_NaCl * vH) 
    # dont convert masses here
    h1_inv = 1.0 / (mass_water_ + mass_solute_ * molar_mass_ratio_w_NaCl * vH) 
        
    S_eq = mass_water_ * h1_inv * np.exp(a_k_over_R_p)
    
    dSeq_dm = S_eq * (1.0E18 / mass_water_
                      - a_k_over_R_p * ( dR_p_dm_over_R + drho_dm_over_rho ) \
                      - (1 - molar_mass_ratio_w_NaCl * dvH_dws
                         * mass_fraction_solute_ * mass_fraction_solute_)
                        * h1_inv * 1.0E18)
# too large by 1.0E12    
#    f1 = 4.0 * np.pi * radius_particle_ * radius_particle_ 
    
#    df1_dm = 2.0 * f1 * dR_p_dm_over_R
    
    c1 = heat_of_vaporization_ * heat_of_vaporization_ \
         / ( thermal_conductivity_air_ * c.specific_gas_constant_water_vapor
             * amb_temp_ * amb_temp_ )
    c2 = c.specific_gas_constant_water_vapor * amb_temp_\
         / (diffusion_constant_ * amb_sat_press_)
    f3 = 1.0 / ( (l_alpha_plus_R_p) * S_eq * c1
                 + (l_beta_plus_R_p) * c2 ) # in SI : m^2 s / kg
    
    f1f3 = 4.0 * np.pi * R_p_SI * R_p_SI * f3 # SI
    
    dg1_dm = ( dSeq_dm * (l_alpha_plus_R_p) + S_eq * dR_p_dm ) * c1\
             + dR_p_dm * c2
    f2 = amb_sat_ - S_eq
    
#    gamma = f1 * f2 * f3 # in kg/s
#    dgamma_dm = f1*f3*( f2*( 2.0*dR_p_dm_over_R - f3*dg1_dm ) - dSeq_dm )
    # return 1.0E18 * gamma, dgamma_dm
    return 1.0E18 * f1f3 * f2,\
           f1f3*( f2*( 2.0*dR_p_dm_over_R - f3*dg1_dm ) - dSeq_dm )

# # WORKS FOR SMALL w_s < 0.001 (w_s_rho_crit), large deviations for large w_s > 0.05
### new: analytic derivative, but with rho_p (m_w, T) = rho_p(m_w^n, T^n) (fix at start value)
#def compute_mass_rate_and_mass_rate_derivative_Szumowski2(mass_water_, mass_solute_,
#                                                         mass_particle_, mass_fraction_solute_, radius_particle_,
#                                                          temperature_particle_, density_particle_,
#                                                          amb_temp_, amb_press_,
#                                                          amb_sat_, amb_sat_press_,
#                                                          diffusion_constant_,
#                                                          thermal_conductivity_air_,
#                                                          specific_heat_capacity_air_,
#                                                          heat_of_vaporization_,
#                                                          adiabatic_index_,
#                                                          accomodation_coefficient_,
#                                                          condensation_coefficient_):
#    
#    # thermal size correction
#    l_alpha = compute_thermal_size_correction_Fukuta(amb_temp_, amb_press_,
#                                                     thermal_conductivity_air_,
#                                                     specific_heat_capacity_air_,
#                                                     adiabatic_index_,
#                                                     accomodation_coefficient_)
#    # diffusive size correction
#    l_beta = compute_diffusive_size_correction_Fukuta(amb_temp_,
#                                                      diffusion_constant_,
#                                                      condensation_coefficient_)
#    # better below...
##    S_eq = compute_kelvin_raoult_term(radius_particle,
##                                      mass_fraction_solute,
##                                      temperature_particle_,
##                                      vant_Hoff,
##                                      density_particle,
##                                      surface_tension_water)
#       
#    R_p_SI = 1.0E-6 * radius_particle_
#    dR_p_dm = 1.0 / ( 4.0 * np.pi * density_particle_ * R_p_SI * R_p_SI ) # in SI: m/kg
##    df1_dm = 2.0 / (density_particle_ * R_p_SI) # in SI
#    
#    m_p_inv = 1.0 / mass_particle_ # in 1 / 1.0E-18 kg
#    
#    surface_tension_ = compute_surface_tension_water(temperature_particle_)
#    a_k_over_R_p = 2.0E6 * surface_tension_ \
#                 / ( specific_gas_constant_water_vapor * amb_temp_ * density_particle_* radius_particle_)# in SI -> no unit
#    
#    vH = compute_vant_Hoff_factor_NaCl(mass_fraction_solute_)
#    dvH_dws = np.where(mass_fraction_solute_ < mf_cross_NaCl, 0.0, par_vH_NaCl[1])
#    
#    # masses in 1.0E-18 kg
#    h1 = (mass_water_ + mass_solute_ * molar_mass_ratio_w_NaCl * vH) 
#    # dont convert masses here
#    h1_inv = 1.0 / h1
#        
#    ### IN WORK, use a_k_over_R_p and h1
#    S_eq = mass_water_ * h1_inv * np.exp(a_k_over_R_p)
#    
#    dSeq_dm = S_eq * 1.0E18 * (1.0 / mass_water_ - one_third * m_p_inv * a_k_over_R_p\
#                      - (1 - molar_mass_ratio_w_NaCl * dvH_dws * mass_fraction_solute_ * mass_fraction_solute_) * h1_inv )
#    
#    f1 = 4.0 * np.pi * radius_particle_ * radius_particle_ # too large by 1.0E12
#    
##    f1 = 4.0 * np.pi * radius_particle_ * radius_particle_ R_p_SI * R_p_SI
#    
#    c1 = heat_of_vaporization_ * heat_of_vaporization_ \
#         / ( thermal_conductivity_air_ * specific_gas_constant_water_vapor * amb_temp_ * amb_temp_ )
#    c2 = specific_gas_constant_water_vapor * amb_temp_ / (diffusion_constant_ * amb_sat_press_)
#    f3 = 1.0 / ( (R_p_SI + l_alpha) * S_eq * c1 + (R_p_SI + l_beta) * c2 ) # in SI : m^2 s / kg
#    
#    dg1_dm = (dSeq_dm * (R_p_SI + l_alpha) + S_eq * dR_p_dm ) * c1 + dR_p_dm * c2
#        
#    gamma = 1.0E6 * f1 * (amb_sat_ - S_eq) * f3 # in 1.0E-18 kg/s
#    
#    dgamma_dm = gamma * (2.0 * one_third * m_p_inv - f3 * dg1_dm * 1.0E-18 ) - 1.0E-12 * f1 * dSeq_dm * f3 # in SI: 1/s
#    
#    return gamma, dgamma_dm
# masses  in femto gram
def compute_mass_rate_from_water_mass_Szumowski(  mass_water_, mass_solute_,
                                                  temperature_particle_,
                                                  amb_temp_, amb_press_,
                                                  amb_sat_, amb_sat_press_,
                                                  diffusion_constant_,
                                                  thermal_conductivity_air_,
                                                  specific_heat_capacity_air_,
                                                  adiabatic_index_,
                                                  accomodation_coefficient_,
                                                  condensation_coefficient_, 
                                                  heat_of_vaporization_
                                                  ):
    # first term
#    mass_water = mass_water_ + h
    mass_particle = mass_water_ + mass_solute_
    mass_fraction_solute = mass_solute_ / mass_particle
    density_particle = compute_density_particle(  mass_fraction_solute,
                                                temperature_particle_  )
     # in micro meter
    radius_particle = c.const_volume_to_radius\
                      * ( mass_particle / density_particle )**(c.one_third)
#    radius_particle_sq = radius_particle*radius_particle
    vant_Hoff = compute_vant_Hoff_factor_NaCl( mass_fraction_solute )
    surface_tension_water = compute_surface_tension_water(temperature_particle_)
    kelvin_raoult_term = compute_kelvin_raoult_term(radius_particle,
                                                      mass_fraction_solute,
                                                      temperature_particle_,
                                                      vant_Hoff,
                                                      density_particle,
                                                      surface_tension_water)
    
    mass_rate = compute_mass_rate_Szumowski_from_Seq(  amb_temp_, amb_press_,
                                              amb_sat_, amb_sat_press_,
                                              diffusion_constant_,
                                              thermal_conductivity_air_,
                                              specific_heat_capacity_air_,
                                              adiabatic_index_,
                                              accomodation_coefficient_,
                                              condensation_coefficient_, 
                                              radius_particle,
                                              heat_of_vaporization_,
                                              kelvin_raoult_term
                                              )
#    print('mass_water_')
#    print('mass_particle')
#    print('mass_fraction_solute')
#    print('density_particle')
#    print('radius_particle')
#    print('surface_tension_water')
#    print('kelvin_raoult_term')
#    print(mass_water_)
#    print(mass_particle)
#    print(mass_fraction_solute)
#    print(density_particle)
#    print(radius_particle)
#    print(surface_tension_water)
#    print(kelvin_raoult_term)
    return mass_rate

###############################################################################

## OLD: numeric derivative
machine_epsilon_sqrt = 1.5E-8
def compute_mass_rate_derivative_Szumowski_numerical(
        mass_water_, mass_solute_, #  in femto gram
        temperature_particle_,
        amb_temp_, amb_press_,
        amb_sat_, amb_sat_press_,
        diffusion_constant_,
        thermal_conductivity_air_,
        specific_heat_capacity_air_,
        adiabatic_index_,
        accomodation_coefficient_,
        condensation_coefficient_, 
        heat_of_vaporization_):
    
    h = mass_water_ * machine_epsilon_sqrt
    # surface_tension_water = compute_surface_tension_water(temperature_particle_)
    
    # first term
    mass_water1 = mass_water_ + h
    
    mass_rate = compute_mass_rate_from_water_mass_Szumowski(
                    mass_water1, mass_solute_, #  in femto gram
                    temperature_particle_,
                    amb_temp_, amb_press_,
                    amb_sat_, amb_sat_press_,
                    diffusion_constant_,
                    thermal_conductivity_air_,
                    specific_heat_capacity_air_,
                    adiabatic_index_,
                    accomodation_coefficient_,
                    condensation_coefficient_, 
                    heat_of_vaporization_)
    # second term
    mass_water2 = mass_water_ - h
    
    mass_rate -= compute_mass_rate_from_water_mass_Szumowski(
                     mass_water2, mass_solute_, #  in femto gram
                     temperature_particle_,
                     amb_temp_, amb_press_,
                     amb_sat_, amb_sat_press_,
                     diffusion_constant_,
                     thermal_conductivity_air_,
                     specific_heat_capacity_air_,
                     adiabatic_index_,
                     accomodation_coefficient_,
                     condensation_coefficient_, 
                     heat_of_vaporization_)
    
    return mass_rate / (mass_water1 - mass_water2)
#    return mass_rate / (2*h)
    
#    mass_particle = mass_water + mass_solute_
#    mass_fraction_solute = mass_solute_ / mass_particle
#    density_particle = compute_density_particle(  mass_fraction_solute,
#                                                  temperature_particle_  )
#    # in micro meter
#    radius_particle = const_volume_to_radius\
#                      * ( mass_particle / density_particle )**(0.33333333) 
##    radius_particle_sq = radius_particle*radius_particle
#    vant_Hoff = compute_vant_Hoff_factor_NaCl( mass_fraction_solute )
#    kelvin_raoult_term = compute_kelvin_raoult_term(radius_particle,
#                                                      mass_fraction_solute,
#                                                      temperature_particle_,
#                                                      vant_Hoff,
#                                                      density_particle,
#                                                      surface_tension_water)
    
#    mass_rate = compute_mass_rate_Szumowski(  amb_temp_, amb_press_,
#                                              amb_sat_, amb_sat_press_,
#                                              diffusion_constant_,
#                                              thermal_conductivity_air_,
#                                              specific_heat_capacity_air_,
#                                              adiabatic_index_,
#                                              accomodation_coefficient_,
#                                              condensation_coefficient_, 
#                                              radius_particle,
#                                              heat_of_vaporization_,
#                                              kelvin_raoult_term
#                                              )
    
    # second term

    
#    mass_particle = mass_water + mass_solute_
#    mass_fraction_solute = mass_solute_ / mass_particle
#    density_particle = compute_density_particle(  mass_fraction_solute, temperature_particle_  )
#    radius_particle = const_volume_to_radius * ( mass_particle / density_particle )**(0.33333333)
##    radius_particle_sq = radius_particle*radius_particle
#    vant_Hoff = compute_vant_Hoff_factor_NaCl( mass_fraction_solute )
#    kelvin_raoult_term = compute_kelvin_raoult_term(radius_particle,
#                                                      mass_fraction_solute,
#                                                      temperature_particle_,
#                                                      vant_Hoff,
#                                                      density_particle,
#                                                      surface_tension_water)
    
#    mass_rate -= compute_mass_rate_Szumowski(  amb_temp_, amb_press_,
#                                              amb_sat_, amb_sat_press_,
#                                              diffusion_constant_,
#                                              thermal_conductivity_air_,
#                                              specific_heat_capacity_air_,
#                                              adiabatic_index_,
#                                              accomodation_coefficient_,
#                                              condensation_coefficient_, 
#                                              radius_particle,
#                                              heat_of_vaporization_,
#                                              kelvin_raoult_term
#                                              )
 
##############################################################################
### integration
# mass:
# returns the difference dm_w = m_w_n+1 - m_w_n during condensation/evaporation
# during one timestep using linear implicit explicit euler
# masses in femto gram    
def compute_delta_water_liquid_imex_linear( dt_, mass_water_, mass_solute_, 
                                                  temperature_particle_,
                                                  amb_temp_, amb_press_,
                                                  amb_sat_, amb_sat_press_,
                                                  diffusion_constant_,
                                                  thermal_conductivity_air_,
                                                  specific_heat_capacity_air_,
                                                  surface_tension_,
                                                  adiabatic_index_,
                                                  accomodation_coefficient_,
                                                  condensation_coefficient_, 
                                                  heat_of_vaporization_,
                                                  verbose = False):

    dt_left = dt_
#    dt = dt_
    mass_water_new = mass_water_
    
    mass_fraction_solute_effl = compute_efflorescence_mass_fraction_NaCl(
                                    temperature_particle_)
    
    while (dt_left > 0.0):
        m_p = mass_water_new + mass_solute_
        w_s = mass_solute_ / m_p
        rho = compute_density_particle(w_s, temperature_particle_)
        R = compute_radius_from_mass(m_p, rho)
        # mass_rate = compute_mass_rate_from_water_mass_Szumowski(
        #                 mass_water_new, mass_solute_, #  in femto gram
        #                 temperature_particle_,
        #                 amb_temp_, amb_press_,
        #                 amb_sat_, amb_sat_press_,
        #                 diffusion_constant_,
        #                 thermal_conductivity_air_,
        #                 specific_heat_capacity_air_,
        #                 adiabatic_index_,
        #                 accomodation_coefficient_,
        #                 condensation_coefficient_, 
        #                 heat_of_vaporization_)
        # # masses in femto gram
        # mass_rate_derivative = compute_mass_rate_derivative_Szumowski(
        #                            mass_water_new, mass_solute_,
        #                            temperature_particle_,
        #                            amb_temp_, amb_press_,
        #                            amb_sat_, amb_sat_press_,
        #                            diffusion_constant_,
        #                            thermal_conductivity_air_,
        #                            specific_heat_capacity_air_,
        #                            adiabatic_index_,
        #                            accomodation_coefficient_,
        #                            condensation_coefficient_, 
        #                            heat_of_vaporization_)
        mass_rate, mass_rate_derivative\
            = compute_mass_rate_and_mass_rate_derivative_Szumowski(
                    mass_water_, mass_solute_,
                    m_p, w_s, R,
                    temperature_particle_, rho,
                    amb_temp_, amb_press_,
                    amb_sat_, amb_sat_press_,
                    diffusion_constant_,
                    thermal_conductivity_air_,
                    specific_heat_capacity_air_,
                    heat_of_vaporization_,
                    surface_tension_,
                    adiabatic_index_,
                    accomodation_coefficient_,
                    condensation_coefficient_)
        if (verbose):
            print('mass_rate, mass_rate_derivative:')
            print(mass_rate, mass_rate_derivative)
        # safety to avoid (1 - dt/2 * f'(m_n)) going to zero
        if mass_rate_derivative * dt_ < 1.0:
            dt = dt_left
            dt_left = -1.0
        else:
            dt = 1.0 / mass_rate_derivative
            dt_left -= dt
    
        mass_water_new +=  mass_rate * dt\
                           / ( 1.0 - 0.5 * mass_rate_derivative * dt )
        
        mass_water_effl = mass_solute_ * (1.0 / mass_fraction_solute_effl - 1.0)
        
#        mass_fraction_solute_new = mass_solute_ / (mass_water_new + mass_solute_)
        
#        if (mass_fraction_solute_new > mass_fraction_solute_effl or mass_water_new < 0.0):
        if (mass_water_new  < mass_water_effl):
#            mass_water_new = mass_solute_ * (1.0 / mass_fraction_solute_effl - 1.0)
            mass_water_new = mass_water_effl
            dt_left = -1.0
#            print('w_s_effl reached')
    
    return mass_water_new - mass_water_

### NEW 23.02.19
# IN WORK: might return gamma and not gamma0 for particle heat, but this is not important right now
def compute_delta_water_liquid_and_mass_rate_implicit_Newton(
        dt_sub_, no_iter_, mass_water_, mass_solute_,
        mass_particle_, mass_fraction_solute_, radius_particle_,
        temperature_particle_, density_particle_,
        amb_temp_, amb_press_,
        amb_sat_, amb_sat_press_,
        diffusion_constant_,
        thermal_conductivity_air_,
        specific_heat_capacity_air_,
        heat_of_vaporization_,
        surface_tension_,
        adiabatic_index_,
        accomodation_coefficient_,
        condensation_coefficient_):
    
    w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
                             temperature_particle_)
    m_w_effl = mass_solute_ * (w_s_effl_inv - 1.0)
    gamma0, dgamma_dm = compute_mass_rate_and_mass_rate_derivative_Szumowski(
                            mass_water_, mass_solute_,
                            mass_particle_, mass_fraction_solute_, radius_particle_,
                            temperature_particle_, density_particle_,
                            amb_temp_, amb_press_,
                            amb_sat_, amb_sat_press_,
                            diffusion_constant_,
                            thermal_conductivity_air_,
                            specific_heat_capacity_air_,
                            heat_of_vaporization_,
                            surface_tension_,
                            adiabatic_index_,
                            accomodation_coefficient_,
                            condensation_coefficient_)
#    no_iter = 3
    dt_sub_times_dgamma_dm = dt_sub_ * dgamma_dm
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                         1.0 / (1.0 - dt_sub_times_dgamma_dm),
                         10.0)
#    if (dt_sub_ * dgamma_dm < 0.9):
#        denom_inv = 
#    else:
#        denom_inv = 10.0
     
    mass_new = np.maximum(m_w_effl, mass_water_ + dt_sub_ * gamma0 * denom_inv)
    
    for cnt in range(no_iter_-1):
        m_p = mass_new + mass_solute_
        w_s = mass_solute_ / m_p
        rho = compute_density_particle(w_s, temperature_particle_)
        R = compute_radius_from_mass(m_p, rho)
        gamma = compute_mass_rate_Szumowski(mass_new, mass_solute_,
                                              w_s, R,
                                              rho,
                                              amb_temp_, amb_press_,
                                              amb_sat_, amb_sat_press_,
                                              diffusion_constant_,
                                              thermal_conductivity_air_,
                                              specific_heat_capacity_air_,
                                              heat_of_vaporization_,
                                              surface_tension_,
                                              adiabatic_index_,
                                              accomodation_coefficient_,
                                              condensation_coefficient_)
        mass_new += ( dt_sub_* gamma + mass_water_ - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - mass_water_, gamma0

def compute_delta_water_liquid_and_mass_rate_implicit_Newton_full(
        dt_sub_, no_iter_, mass_water_, mass_solute_,
        mass_particle_, mass_fraction_solute_, radius_particle_,
        temperature_particle_, density_particle_,
        amb_temp_, amb_press_,
        amb_sat_, amb_sat_press_,
        diffusion_constant_,
        thermal_conductivity_air_,
        specific_heat_capacity_air_,
        heat_of_vaporization_,
        surface_tension_,
        adiabatic_index_,
        accomodation_coefficient_,
        condensation_coefficient_):
    w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
                             temperature_particle_)
    m_w_effl = mass_solute_ * (w_s_effl_inv - 1.0)
    
    gamma0, dgamma_dm = compute_mass_rate_and_mass_rate_derivative_Szumowski(
                            mass_water_, mass_solute_,
                            mass_particle_, mass_fraction_solute_,
                            radius_particle_,
                            temperature_particle_, density_particle_,
                            amb_temp_, amb_press_,
                            amb_sat_, amb_sat_press_,
                            diffusion_constant_,
                            thermal_conductivity_air_,
                            specific_heat_capacity_air_,
                            heat_of_vaporization_,
                            surface_tension_,
                            adiabatic_index_,
                            accomodation_coefficient_,
                            condensation_coefficient_)
#    no_iter = 3
    dt_sub_times_dgamma_dm = dt_sub_ * dgamma_dm
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                         1.0 / (1.0 - dt_sub_times_dgamma_dm),
                         10.0)
#    if (dt_sub_ * dgamma_dm < 0.9):
#        denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
#    else:
#        denom_inv = 10.0
     
    mass_new = np.maximum(m_w_effl, mass_water_ + dt_sub_ * gamma0 * denom_inv)
    
    for cnt in range(no_iter_-1):
        m_p = mass_new + mass_solute_
        w_s = mass_solute_ / m_p
        rho = compute_density_particle(w_s, temperature_particle_)
        R = compute_radius_from_mass(m_p, rho)
        gamma, dgamma_dm = compute_mass_rate_and_mass_rate_derivative_Szumowski(
                               mass_new, mass_solute_,
                               m_p, w_s, R,
                               temperature_particle_, rho,
                               amb_temp_, amb_press_,
                               amb_sat_, amb_sat_press_,
                               diffusion_constant_,
                               thermal_conductivity_air_,
                               specific_heat_capacity_air_,
                               heat_of_vaporization_,
                               surface_tension_,
                               adiabatic_index_,
                               accomodation_coefficient_,
                               condensation_coefficient_)
        dt_sub_times_dgamma_dm = dt_sub_ * dgamma_dm
        denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                             1.0 / (1.0 - dt_sub_times_dgamma_dm),
                     10.0)
#        if (dt_sub_ * dgamma_dm < 0.9):
#            denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
#        else:
#            denom_inv = 10.0
        mass_new += ( dt_sub_* gamma + mass_water_ - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - mass_water_, gamma0

def compute_delta_water_liquid_and_mass_rate_implicit_Newton_full_const_l(
        dt_sub_, no_iter_, mass_water_, mass_solute_,
        mass_particle_, mass_fraction_solute_, radius_particle_,
        temperature_particle_, density_particle_,
        amb_temp_, amb_press_,
        amb_sat_, amb_sat_press_,
        diffusion_constant_,
        thermal_conductivity_air_,
        specific_heat_capacity_air_,
        heat_of_vaporization_,
        surface_tension_,
        adiabatic_index_,
        accomodation_coefficient_,
        condensation_coefficient_):
    w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
                             temperature_particle_)
    m_w_effl = mass_solute_ * (w_s_effl_inv - 1.0)
    
    gamma0, dgamma_dm =\
        compute_mass_rate_and_mass_rate_derivative_Szumowski_const_l(
            mass_water_, mass_solute_,
            mass_particle_, mass_fraction_solute_, radius_particle_,
            temperature_particle_, density_particle_,
            amb_temp_, amb_press_,
            amb_sat_, amb_sat_press_,
            diffusion_constant_,
            thermal_conductivity_air_,
            specific_heat_capacity_air_,
            heat_of_vaporization_,
            surface_tension_,
            adiabatic_index_,
            accomodation_coefficient_,
            condensation_coefficient_)
#    no_iter = 3
    dt_sub_times_dgamma_dm = dt_sub_ * dgamma_dm
    denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                         1.0 / (1.0 - dt_sub_times_dgamma_dm),
                         10.0)
#    if (dt_sub_ * dgamma_dm < 0.9):
#        denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
#    else:
#        denom_inv = 10.0
     
    mass_new = np.maximum(m_w_effl, mass_water_ + dt_sub_ * gamma0 * denom_inv)
    
    for cnt in range(no_iter_-1):
        m_p = mass_new + mass_solute_
        w_s = mass_solute_ / m_p
        rho = compute_density_particle(w_s, temperature_particle_)
        R = compute_radius_from_mass(m_p, rho)
        gamma, dgamma_dm =\
            compute_mass_rate_and_mass_rate_derivative_Szumowski_const_l(
                mass_new, mass_solute_,
                m_p, w_s, R,
                temperature_particle_, rho,
                amb_temp_, amb_press_,
                amb_sat_, amb_sat_press_,
                diffusion_constant_,
                thermal_conductivity_air_,
                specific_heat_capacity_air_,
                heat_of_vaporization_,
                surface_tension_,
                adiabatic_index_,
                accomodation_coefficient_,
                condensation_coefficient_)
        dt_sub_times_dgamma_dm = dt_sub_ * dgamma_dm
        denom_inv = np.where(dt_sub_times_dgamma_dm < 0.9,
                             1.0 / (1.0 - dt_sub_times_dgamma_dm),
                     10.0)
#        if (dt_sub_ * dgamma_dm < 0.9):
#            denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
#        else:
#            denom_inv = 10.0
        mass_new += ( dt_sub_* gamma + mass_water_ - mass_new) * denom_inv
        mass_new = np.maximum( m_w_effl, mass_new )
        
    return mass_new - mass_water_, gamma0

def compute_delta_water_liquid_and_mass_rate_implicit_Newton_inverse_full(
        dt_sub_, no_iter_, mass_water_, mass_solute_,
        mass_particle_, mass_fraction_solute_, radius_particle_,
        temperature_particle_, density_particle_,
        amb_temp_, amb_press_,
        amb_sat_, amb_sat_press_,
        diffusion_constant_,
        thermal_conductivity_air_,
        specific_heat_capacity_air_,
        heat_of_vaporization_,
        surface_tension_,
        adiabatic_index_,
        accomodation_coefficient_,
        condensation_coefficient_):
    w_s_effl_inv = 1.0 / compute_efflorescence_mass_fraction_NaCl(
                             temperature_particle_)
    m_w_effl = mass_solute_ * (w_s_effl_inv - 1.0)
    
    gamma0, dgamma_dm = compute_mass_rate_and_mass_rate_derivative_Szumowski(
                            mass_water_, mass_solute_,
                            mass_particle_, mass_fraction_solute_,
                            radius_particle_,
                            temperature_particle_, density_particle_,
                            amb_temp_, amb_press_,
                            amb_sat_, amb_sat_press_,
                            diffusion_constant_,
                            thermal_conductivity_air_,
                            specific_heat_capacity_air_,
                            heat_of_vaporization_,
                            surface_tension_,
                            adiabatic_index_,
                            accomodation_coefficient_,
                            condensation_coefficient_)
#    mass_new = mass_water_
#    no_iter = 3
    dgamma_factor = dt_sub_ * dgamma_dm
    # dgamma_factor = dt_sub * F'(m) * m
    dgamma_factor = np.where(dgamma_factor < 0.9,
                             mass_water_ * (1.0 - dgamma_factor),
                             mass_water_ * 0.1)
#    if (dt_sub_ * dgamma_dm < 0.9):
#        denom_inv = 1.0 / (1.0 - dt_sub_ * dgamma_dm)
#    else:
#        denom_inv = 10.0
    
    print("iter = 1",
          mass_water_ * dgamma_factor / (dgamma_factor - gamma0 * dt_sub_))
    mass_new = np.maximum( m_w_effl,
                           mass_water_ * dgamma_factor\
                           / (dgamma_factor - gamma0 * dt_sub_) )
    print("iter = 1", mass_new)
    
    for cnt in range(no_iter_-1):
        m_p = mass_new + mass_solute_
        w_s = mass_solute_ / m_p
        rho = compute_density_particle(w_s, temperature_particle_)
        R = compute_radius_from_mass(m_p, rho)
        gamma, dgamma_dm = compute_mass_rate_and_mass_rate_derivative_Szumowski(
                               mass_new, mass_solute_,
                               m_p, w_s, R,
                               temperature_particle_, rho,
                               amb_temp_, amb_press_,
                               amb_sat_, amb_sat_press_,
                               diffusion_constant_,
                               thermal_conductivity_air_,
                               specific_heat_capacity_air_,
                               heat_of_vaporization_,
                               surface_tension_,
                               adiabatic_index_,
                               accomodation_coefficient_,
                               condensation_coefficient_)
        dgamma_factor = dt_sub_ * dgamma_dm
        # dgamma_factor = dt_sub * F'(m) * m
        dgamma_factor = np.where(dgamma_factor < 0.9,
                                 mass_new * (1.0 - dgamma_factor),
                                 mass_new * 0.1)
#        mass_new *= ( dt_sub_* gamma + mass_water_ - mass_new) * denom_inv
        print("iter = ", cnt + 2 ,
              mass_new * dgamma_factor\
              / ( dgamma_factor - gamma * dt_sub_ + mass_new - mass_water_ ))
        mass_new = np.maximum( m_w_effl,
                               mass_new * dgamma_factor\
                               / ( dgamma_factor - gamma * dt_sub_
                                   + mass_new - mass_water_ ) )
        print("iter = ", cnt+2, mass_new)
    return mass_new - mass_water_, gamma0
    
# returns the difference dm_w = m_w_n+1 - m_w_n
# during condensation/evaporation
# during one timestep using linear implicit explicit euler
# also: returns mass_rate
def compute_delta_water_liquid_and_mass_rate_imex_linear(
        dt_, mass_water_, mass_solute_, #  in femto gram
        temperature_particle_,
        amb_temp_, amb_press_,
        amb_sat_, amb_sat_press_,
        diffusion_constant_,
        thermal_conductivity_air_,
        specific_heat_capacity_air_,
        adiabatic_index_,
        accomodation_coefficient_,
        condensation_coefficient_, 
        heat_of_vaporization_,
        verbose = False):

    dt_left = dt_
#    dt = dt_
    mass_water_new = mass_water_
    
    mass_fraction_solute_effl = compute_efflorescence_mass_fraction_NaCl(
                                    temperature_particle_)
    
    while (dt_left > 0.0):
        mass_rate = compute_mass_rate_from_water_mass_Szumowski(
                        mass_water_new, mass_solute_, #  in femto gram
                        temperature_particle_,
                        amb_temp_, amb_press_,
                        amb_sat_, amb_sat_press_,
                        diffusion_constant_,
                        thermal_conductivity_air_,
                        specific_heat_capacity_air_,
                        adiabatic_index_,
                        accomodation_coefficient_,
                        condensation_coefficient_, 
                        heat_of_vaporization_)
        mass_rate_derivative = compute_mass_rate_derivative_Szumowski_numerical(
                                                  mass_water_new, mass_solute_, #  in femto gram
                                                  temperature_particle_,
                                                  amb_temp_, amb_press_,
                                                  amb_sat_, amb_sat_press_,
                                                  diffusion_constant_,
                                                  thermal_conductivity_air_,
                                                  specific_heat_capacity_air_,
                                                  adiabatic_index_,
                                                  accomodation_coefficient_,
                                                  condensation_coefficient_, 
                                                  heat_of_vaporization_
                                                  )
        if (verbose):
            print('mass_rate, mass_rate_derivative:')
            print(mass_rate, mass_rate_derivative)
        # safety to avoid (1 - dt/2 * f'(m_n)) going to zero
        if mass_rate_derivative * dt_ < 1.0:
            dt = dt_left
            dt_left = -1.0
        else:
            dt = 1.0 / mass_rate_derivative
            dt_left -= dt
    
        mass_water_new += mass_rate * dt\
                          / ( 1.0 - 0.5 * mass_rate_derivative * dt )
        
        mass_water_effl = mass_solute_\
                          * (1.0 / mass_fraction_solute_effl - 1.0)
        
#        mass_fraction_solute_new = mass_solute_\
#           / (mass_water_new + mass_solute_)
        
#        if (mass_fraction_solute_new >
#             mass_fraction_solute_effl or mass_water_new < 0.0):
        if (mass_water_new  < mass_water_effl):
# mass_water_new = mass_solute_ * (1.0 / mass_fraction_solute_effl - 1.0)
            mass_water_new = mass_water_effl
            dt_left = -1.0
#            print('w_s_effl reached')
    
    return mass_water_new - mass_water_, mass_rate

# returns the difference dm_w = m_w_n+1 - m_w_n
# during condensation/evaporation
# during one timestep using linear implicit euler
# masses in femto gram
def compute_delta_water_liquid_implicit_linear( dt_, mass_water_, mass_solute_,
                                                temperature_particle_,
                                                amb_temp_, amb_press_,
                                                amb_sat_, amb_sat_press_,
                                                diffusion_constant_,
                                                thermal_conductivity_air_,
                                                specific_heat_capacity_air_,
                                                adiabatic_index_,
                                                accomodation_coefficient_,
                                                condensation_coefficient_, 
                                                heat_of_vaporization_,
                                                verbose = False):

    dt_left = dt_
#    dt = dt_
    mass_water_new = mass_water_
    
    mass_fraction_solute_effl = compute_efflorescence_mass_fraction_NaCl(
                                    temperature_particle_)
    
    surface_tension_ = compute_surface_tension_water(temperature_particle_)
    
    while (dt_left > 0.0):
#        mass_rate = compute_mass_rate_from_water_mass_Szumowski(
#                                mass_water_new, mass_solute_, #  in femto gram
#                                                  temperature_particle_,
#                                                  amb_temp_, amb_press_,
#                                                  amb_sat_, amb_sat_press_,
#                                                  diffusion_constant_,
#                                                  thermal_conductivity_air_,
#                                                  specific_heat_capacity_air_,
#                                                  adiabatic_index_,
#                                                  accomodation_coefficient_,
#                                                  condensation_coefficient_, 
#                                                  heat_of_vaporization_
#                                                      )
        m_p = mass_water_new + mass_solute_
        w_s = mass_solute_ / m_p
        rho = compute_density_particle(w_s, temperature_particle_)
        R = compute_radius_from_mass(m_p, rho)
        mass_rate, mass_rate_derivative =\
            compute_mass_rate_and_mass_rate_derivative_Szumowski(
                mass_water_new, mass_solute_,
                m_p, w_s, R,
                temperature_particle_, rho,
                amb_temp_, amb_press_,
                amb_sat_, amb_sat_press_,
                diffusion_constant_,
                thermal_conductivity_air_,
                specific_heat_capacity_air_,
                heat_of_vaporization_,
                surface_tension_,
                adiabatic_index_,
                accomodation_coefficient_,
                condensation_coefficient_)
        if (verbose):
            print('mass_rate, mass_rate_derivative:')
            print(mass_rate, mass_rate_derivative)
        if mass_rate_derivative * dt_ < 0.5:
            dt = dt_left
            dt_left = -1.0
        else:
            dt = 0.5 / mass_rate_derivative
            dt_left -= dt
    
        mass_water_new += mass_rate * dt\
                          / ( 1.0 - mass_rate_derivative * dt )
        
        mass_water_effl = mass_solute_ * (1.0 / mass_fraction_solute_effl - 1.0)
        
#        mass_fraction_solute_new =\
#  mass_solute_ / (mass_water_new + mass_solute_)
        
#        if (mass_fraction_solute_new
#            > mass_fraction_solute_effl or mass_water_new < 0.0):
        if (mass_water_new  < mass_water_effl):
#            mass_water_new = mass_solute_\
#                             * (1.0 / mass_fraction_solute_effl - 1.0)
            mass_water_new = mass_water_effl
            dt_left = -1.0
#            print('w_s_effl reached')
    
    return mass_water_new - mass_water_

def compute_mass_rate_from_surface_partial_pressure(  amb_temp_,
                                                      amb_sat_, amb_sat_press_,
                                                      diffusion_constant_,
                                                      radius_,
                                                      surface_partial_pressure_,
                                                      particle_temperature_,
                                                      ):
    return 4.0E12 * np.pi * radius_ * diffusion_constant_\
           / c.specific_gas_constant_water_vapor \
           * ( amb_sat_ * amb_sat_press_ / amb_temp_
               - surface_partial_pressure_ / particle_temperature_ )
           
           


    
    