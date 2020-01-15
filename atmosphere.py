#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 12:33:07 2019

@author: jdesk
"""

import constants as c
from numba import vectorize, njit

# J/(kg K)
def compute_specific_gas_constant_air_moist(specific_humidity_):
    return c.specific_gas_constant_air_dry * (1 + 0.608 * specific_humidity_ )

# J/(kg K)
@njit()
def compute_specific_heat_capacity_air_moist(mixing_ratio_vapor_):
    return c.specific_heat_capacity_air_dry_NTP * \
            ( 1 + 0.897 * mixing_ratio_vapor_ )

#%% ATMOSPHERIC ENVIRONMENTAL PROFILE
kappa_air_dry = c.specific_gas_constant_air_dry\
                / c.specific_heat_capacity_air_dry_NTP

def compute_kappa_air_moist(mixing_ratio_vapor_):
    return kappa_air_dry * ( 1 - 0.289 * mixing_ratio_vapor_ )

epsilon_gc = c.specific_gas_constant_air_dry\
             / c.specific_gas_constant_water_vapor

epsilon_gc_prime = 1.0 / epsilon_gc - 1

def compute_beta_without_liquid(mixing_ratio_total_,
                                liquid_potential_temperature_):
    return c.earth_gravity * compute_kappa_air_moist(mixing_ratio_total_)\
           * (1 + mixing_ratio_total_) \
            / ( c.specific_gas_constant_air_dry * liquid_potential_temperature_
                * (1 + mixing_ratio_total_/epsilon_gc) )            

# general formula for any Theta_l, r_tot
# for z0 = 0 (surface)
def compute_T_over_Theta_l_without_liquid( z_, p_0_over_p_ref_to_kappa_tot_,
                                          beta_tot_ ):
    return p_0_over_p_ref_to_kappa_tot_ - beta_tot_ * z_

def compute_potential_temperature_moist( temperature_, pressure_,
                                        pressure_reference_,
                                        mixing_ratio_vapor_ ):
    return temperature_ \
            * ( pressure_reference_ / pressure_ )\
            **( compute_kappa_air_moist(mixing_ratio_vapor_) )

def compute_potential_temperature_dry( temperature_, pressure_,
                                      pressure_reference_ ):
    return temperature_ * ( pressure_reference_ / pressure_ )\
                        **( kappa_air_dry )

def compute_temperature_from_potential_temperature_moist(potential_temperature_, 
                                                         pressure_, 
                                                         pressure_reference_, 
                                                         mixing_ratio_vapor_ ):
    return potential_temperature_ * \
           ( pressure_ / pressure_reference_ )\
           **( compute_kappa_air_moist(mixing_ratio_vapor_) )
@njit()
def compute_temperature_from_potential_temperature_dry( potential_temperature_,
                                                       pressure_,
                                                       pressure_reference_ ):
    return potential_temperature_ * \
            ( pressure_ / pressure_reference_ )**( kappa_air_dry )

@vectorize("float64(float64, float64, float64)") 
def compute_pressure_ideal_gas( mass_density_, temperature_, 
                                specific_gas_constant_ ):
    return mass_density_ * temperature_ * specific_gas_constant_ 

@vectorize("float64(float64, float64)") 
def compute_pressure_vapor( density_vapor_, temperature_ ):
    return compute_pressure_ideal_gas( density_vapor_,
                                      temperature_,
                                      c.specific_gas_constant_water_vapor )
@vectorize("float64(float64, float64)") 
def compute_density_air_dry(temperature_, pressure_):
    return pressure_ / ( c.specific_gas_constant_air_dry * temperature_ )
@vectorize("float64(float64, float64)", target="parallel") 
def compute_density_air_dry_par(temperature_, pressure_):
    return pressure_ / ( c.specific_gas_constant_air_dry * temperature_ )

### CONVERSION DRY POTENTIAL TEMPERATURE
c_pv_over_c_pd = c.specific_heat_capacity_water_vapor_20C \
                 / c.specific_heat_capacity_air_dry_NTP
kappa_factor = 1.0 / (1.0 - kappa_air_dry)
kappa_factor2 = -kappa_air_dry * kappa_factor
@njit()
def compute_p_dry_over_p_ref(grid_mass_density_air_dry,
                             grid_potential_temperature,
                             p_ref_inv):
    return ( grid_mass_density_air_dry * grid_potential_temperature \
             * c.specific_gas_constant_air_dry * p_ref_inv )**kappa_factor

# Theta/T from Theta and rho_dry 
# NOTE THAT "kappa factor 2" is negative
# grid.p_ref_inv needs to be right (default is p_ref = 1.0E5)
@njit()
def compute_Theta_over_T(grid_mass_density_air_dry, grid_potential_temperature,
                         p_ref_inv):
    return (p_ref_inv * c.specific_gas_constant_air_dry\
            * grid_mass_density_air_dry * grid_potential_temperature )\
            **kappa_factor2
