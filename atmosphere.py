#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

ATMOSPHERIC ENVIRONMENT
including general physical laws and conversions

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

import constants as c
from numba import vectorize, njit

#%% MATERIAL PROPERTIES

# J/(kg K)
def compute_specific_gas_constant_air_moist(specific_humidity):
    return c.specific_gas_constant_air_dry * (1 + 0.608 * specific_humidity )

# J/(kg K)
@njit()
def compute_specific_heat_capacity_air_moist(mixing_ratio_vapor):
    return c.specific_heat_capacity_air_dry_NTP * \
            ( 1 + 0.897 * mixing_ratio_vapor )

#%% ATMOSPHERIC ENVIRONMENTAL PROFILE
kappa_air_dry = c.specific_gas_constant_air_dry\
                / c.specific_heat_capacity_air_dry_NTP

def compute_kappa_air_moist(mixing_ratio_vapor):
    return kappa_air_dry * ( 1 - 0.289 * mixing_ratio_vapor )

epsilon_gc = c.specific_gas_constant_air_dry\
             / c.specific_gas_constant_water_vapor

epsilon_gc_prime = 1.0 / epsilon_gc - 1

def compute_beta_without_liquid(mixing_ratio_total,
                                liquid_potential_temperature):
    return c.earth_gravity * compute_kappa_air_moist(mixing_ratio_total)\
           * (1 + mixing_ratio_total) \
            / ( c.specific_gas_constant_air_dry * liquid_potential_temperature
                * (1 + mixing_ratio_total/epsilon_gc) )            

# general formula for any Theta_l, r_tot
# for z0 = 0 (surface)
def compute_T_over_Theta_l_without_liquid(z, p_0_over_p_ref_to_kappa_tot,
                                          beta_tot):
    return p_0_over_p_ref_to_kappa_tot - beta_tot * z

def compute_potential_temperature_moist(temperature, pressure,
                                        pressure_reference,
                                        mixing_ratio_vapor):
    return temperature \
            * ( pressure_reference / pressure )\
            **( compute_kappa_air_moist(mixing_ratio_vapor) )

def compute_potential_temperature_dry(temperature, pressure,
                                      pressure_reference):
    return temperature * ( pressure_reference / pressure )\
                        **( kappa_air_dry )

def compute_temperature_from_potential_temperature_moist(
        potential_temperature, pressure, 
        pressure_reference, mixing_ratio_vapor):
    return potential_temperature * \
           ( pressure / pressure_reference )\
           **( compute_kappa_air_moist(mixing_ratio_vapor) )
@njit()
def compute_temperature_from_potential_temperature_dry(potential_temperature,
                                                       pressure,
                                                       pressure_reference):
    return potential_temperature * \
            ( pressure / pressure_reference )**( kappa_air_dry )

@vectorize('float64(float64, float64, float64)') 
def compute_pressure_ideal_gas( mass_density, temperature, 
                                specific_gas_constant ):
    return mass_density * temperature * specific_gas_constant 

@vectorize('float64(float64, float64)') 
def compute_pressure_vapor( density_vapor, temperature ):
    return compute_pressure_ideal_gas(density_vapor,
                                      temperature,
                                      c.specific_gas_constant_water_vapor )
@vectorize('float64(float64, float64)') 
def compute_density_air_dry(temperature, pressure):
    return pressure / ( c.specific_gas_constant_air_dry * temperature )

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
# NOTE THAT 'kappa factor 2' is negative
# grid.p_ref_inv needs to be right (default is p_ref = 1.0E5)
@njit()
def compute_Theta_over_T(grid_mass_density_air_dry, grid_potential_temperature,
                         p_ref_inv):
    return (p_ref_inv * c.specific_gas_constant_air_dry\
            * grid_mass_density_air_dry * grid_potential_temperature )\
            **kappa_factor2
