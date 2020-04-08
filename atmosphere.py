#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinematic framework
(Test Case 1 ICMW 2012)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

ATMOSPHERIC ENVIRONMENT
Including general physical laws and conversions

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units
"""

import constants as c
from numba import vectorize, njit

#%% MATERIAL PROPERTIES

def compute_specific_gas_constant_air_moist(r_v):
    """Compute the specific gas constant of moist air
    
    R_m = (rho_dry * R_dry + rho_v * R_v) / (rho_dry + rho_v)
    approx= R_dry * (1 + r_v * (R_v/R_dry - 1)), where
    R_x: specific gas constant, rho: density, v: vapor, dry: dry air
    v: vapor, dry: dry air, m: moist

    Parameters
    ----------
    r_v: float
        Water vapor mixing ratio = rho_v / rho_dry

    Returns
    -------
        float
        Specific gas constant of moist air (J/(kg K))
        
    """
    
    return c.specific_gas_constant_air_dry * (1 + 0.608 * r_v )

@njit()
def compute_specific_heat_capacity_air_moist(r_v):
    """Compute the specific isobaric heat capacity of moist air
    
    c_m = (rho_dry * c_dry + rho_v * c_v) / (rho_dry + rho_v)
    approx= c_dry * (1 + r_v * (c_v/c_dry - 1)), where
    c_x: specific heat capacity, rho: density,
    v: vapor, dry: dry air, m: moist

    Parameters
    ----------
    r_v: float
        Water vapor mixing ratio = rho_v / rho_dry

    Returns
    -------
        float
        Specific isobaric heat capacity of moist air (J/(kg K))
        
    """
    
    return c.specific_heat_capacity_air_dry_NTP * \
            ( 1 + 0.897 * r_v )

#%% ATMOSPHERIC ENVIRONMENTAL PROFILE

# 'kappa' parameter, where kappa_dry = R_dry/c_dry
# = ratio of specific gas constant and specific heat capacity of dry air
kappa_air_dry = c.specific_gas_constant_air_dry\
                / c.specific_heat_capacity_air_dry_NTP

def compute_kappa_air_moist(mixing_ratio_vapor):
    """Ratio of spec. gas const. and spec. heat capacity (moist air)
    
    'kappa' parameter, where kappa_m = R_m/c_m
    approx= R_dry/c_dry*[1 + r_v * (R_v/R_dry - c_v/c_dry)]
    c_x: specific heat capacity, rho: density, R_x: specific gas constant
    v: vapor, dry: dry air, m: moist

    Parameters
    ----------
    r_v: float
        Water vapor mixing ratio = rho_v / rho_dry

    Returns
    -------
        float
        Ratio of spec. gas const. and spec. heat capacity (moist air)
        
    """
    
    return kappa_air_dry * ( 1 - 0.289 * mixing_ratio_vapor )

epsilon_gc = c.specific_gas_constant_air_dry\
             / c.specific_gas_constant_water_vapor
epsilon_gc_prime = 1.0 / epsilon_gc - 1

def compute_beta_without_liquid(mixing_ratio_total,
                                liquid_potential_temperature):
    """Slope of height profile of temperature over liquid pot. temperature
    
    For an atmosphere without liquid water, the ratio of temperature and
    liquid potential temperature (which is constant)
    describes a linear profile with height.
    The 'beta' parameter is the negative slope of the profile.

    Parameters
    ----------
    mixing_ratio_total: float
        Total water mixing ratio
        r_tot = r_v + r_l = r_v (since r_l = 0),
        where r_v = water vapor mixing ratio,
        r_l = liquid water mixing ratio
    liquid_potential_temperature: float
        Liquid water potential temperature (K):
        moist potential temperature that an air parcel attains,
        when all of its liquid water evaporates due to reversible
        moist decent.

    Returns
    -------
        float
        Slope of height profile of temperature over liquid pot. temperature
        
    """
    
    return c.earth_gravity * compute_kappa_air_moist(mixing_ratio_total)\
           * (1 + mixing_ratio_total) \
            / ( c.specific_gas_constant_air_dry * liquid_potential_temperature
                * (1 + mixing_ratio_total/epsilon_gc) )            

def compute_T_over_Theta_l_without_liquid(z, p_0_over_p_ref_to_kappa_tot,
                                          beta_tot):
    """Profile of the ratio of temperature and liquid pot. temperature
    
    For an atmosphere without liquid water, the ratio of temperature and
    liquid potential temperature (which is constant)
    describes a linear profile with height.
    The 'beta' parameter is the negative slope of the profile.
    Assumes that z_0 = 0 (where pressure p_0 belongs to height z_0)

    Parameters
    ----------
    z: float
        Height in meter
    p_0_over_p_ref_to_kappa_tot:
        = (p_0 / p_ref)^kappa_tot, where p_0 = pressure at z = 0,
        p_ref = reference pressure of the potential temperature,
        kappa_tot = kappa_moist(r_v) = R_m/c_m (see 'kappa_air_moist' above),
        since r_tot = r_v + r_l = r_v (mixing ratios of vapor and liquid)
    beta_tot: float
        Negative slope of the height profile

    Returns
    -------
        float
        Ratio of temperature and liquid potential temperature at height z
        
    """
    
    return p_0_over_p_ref_to_kappa_tot - beta_tot * z

def compute_potential_temperature_moist(temperature, pressure,
                                        pressure_reference,
                                        mixing_ratio_vapor):
    """Compute moist potential temperature
    
    Theta_m = T * (p_ref/p)^kappa_moist
    kappa_moist = R_m/c_m
    R_m = Spec. gas const. moist air
    c_m = spec. heat capacity moist air

    Parameters
    ----------
    temperature: float
        Temperature of the atmosphere (K)
    pressure: float
        Pressure of the atmosphere (P)
    pressure_reference: float
        Reference pressure for the potential temperature
    mixing_ratio_vapor: float
        Water vapor mixing ratio r_v = rho_v / rho_dry

    Returns
    -------
        float
        Moist potential temperature (K)
        
    """
    
    return temperature \
            * ( pressure_reference / pressure )\
            **( compute_kappa_air_moist(mixing_ratio_vapor) )

def compute_potential_temperature_dry(temperature, pressure,
                                      pressure_reference):
    """Compute dry potential temperature
    
    Theta_dry = T * (p_ref/p_dry)^kappa_dry
    kappa_dry = R_dry/c_dry
    R_dry = Spec. gas const. dry air
    c_dry = spec. heat capacity dry air

    Parameters
    ----------
    temperature: float
        Temperature of the atmosphere (K)
    pressure: float
        Pressure of the atmosphere (P)
    pressure_reference: float
        Reference pressure for the potential temperature

    Returns
    -------
        float
        Dry potential temperature (K)
        
    """
    
    return temperature * ( pressure_reference / pressure )\
                        **( kappa_air_dry )

def compute_temperature_from_potential_temperature_moist(
        potential_temperature, pressure, 
        pressure_reference, mixing_ratio_vapor):
    """Compute temperature from moist potential temperature
    
    Theta_m = T * (p_ref/p)^kappa_moist
    kappa_moist = R_m/c_m
    R_m = Spec. gas const. moist air
    c_m = spec. heat capacity moist air

    Parameters
    ----------
    potential_temperature: float
        Moist potential temperature of the atmosphere (K)
    pressure: float
        Pressure of the atmosphere (P)
    pressure_reference: float
        Reference pressure for the potential temperature
    mixing_ratio_vapor: float
        Water vapor mixing ratio r_v = rho_v / rho_dry

    Returns
    -------
        float
        Temperature (K)
        
    """
    
    return potential_temperature * \
           ( pressure / pressure_reference )\
           **( compute_kappa_air_moist(mixing_ratio_vapor) )

@njit()
def compute_temperature_from_potential_temperature_dry(potential_temperature,
                                                       pressure,
                                                       pressure_reference):
    """Compute temperature from dry potential temperature
    
    Theta_dry = T * (p_ref/p_dry)^kappa_dry
    kappa_dry = R_dry/c_dry
    R_dry = Spec. gas const. dry air
    c_dry = spec. heat capacity dry air

    Parameters
    ----------
    potential_temperature: float
        Dry potential temperature of the atmosphere (K)
    pressure: float
        Pressure of the atmosphere (P)
    pressure_reference: float
        Reference pressure for the potential temperature

    Returns
    -------
        float
        Temperature (K)
        
    """
    
    return potential_temperature * \
            ( pressure / pressure_reference )**( kappa_air_dry )

@vectorize('float64(float64, float64, float64)') 
def compute_pressure_ideal_gas(mass_density, temperature, 
                               specific_gas_constant):
    """Compute pressure (ideal gas equation)
    
    Parameters
    ----------
    mass_density: float
        Mass density of the gas (kg/m^3)
    temperature: float
        Temperature (K)
    specific_gas_constant: float
        Specific gas constant

    Returns
    -------
        float
        Pressure for ideal gas (K)
        
    """
    
    return mass_density * temperature * specific_gas_constant 

@vectorize('float64(float64, float64)') 
def compute_pressure_vapor(density_vapor, temperature):
    """Compute water vapor pressure (ideal gas equation)
    
    Parameters
    ----------
    density_vapor: float
        Mass density of water vapor (kg/m^3)
    temperature: float
        Temperature (K)

    Returns
    -------
        float
        Partial pressure of water vapor as ideal gas (K)
        
    """
    
    return compute_pressure_ideal_gas(density_vapor,
                                      temperature,
                                      c.specific_gas_constant_water_vapor )

@vectorize('float64(float64, float64)')
def compute_density_air_dry(temperature, pressure):
    """Compute mass density of dry air (ideal gas equation)
    
    Theta_dry = T * (p_ref/p_dry)^kappa_dry
    kappa_dry = R_dry/c_dry
    R_dry = Spec. gas const. dry air
    c_dry = spec. heat capacity dry air

    Parameters
    ----------
    temperature: float
        Temperature (K)
    pressure: float
        Pressure of dry air

    Returns
    -------
        float
        Density of dry air (as ideal gas) (Pa)
        
    """
    
    return pressure / ( c.specific_gas_constant_air_dry * temperature )

### CONVERSION DRY POTENTIAL TEMPERATURE
c_pv_over_c_pd = c.specific_heat_capacity_water_vapor_20C \
                 / c.specific_heat_capacity_air_dry_NTP
# considers heating of water vapor in the update function for
# dry potential temperature
heat_factor_r_v = (c.specific_heat_capacity_water_vapor_20C - 
                   c.specific_gas_constant_water_vapor) \
                 / (c.specific_heat_capacity_air_dry_NTP -
                    c.specific_gas_constant_air_dry)
                 
kappa_factor = 1.0 / (1.0 - kappa_air_dry)
kappa_factor2 = -kappa_air_dry * kappa_factor
@njit()
def compute_p_dry_over_p_ref(grid_mass_density_air_dry,
                             grid_potential_temperature,
                             p_ref_inv):
    """Compute ratio of dry air pressure and reference pressure
    
    Theta_dry = T * (p_ref/p_dry)^kappa_dry
    kappa_dry = R_dry/c_dry
    R_dry = Spec. gas const. dry air
    c_dry = spec. heat capacity dry air

    Parameters
    ----------
    grid_mass_density_air_dry: ndarray, dtype=float
        2D array of the discretized atmos. dry mass density field (kg/m^3)
    grid_potential_temperature: ndarray, dtype=float
        2D array of the discretized atmos. dry potential temperature
        field (K) := T (p_ref / p_dry)^(kappa_dry), where
        kappa_dry = R_dry / c_dry, where R_dry = specific gas constant
        of dry air, c_dry = specific isobaric heat capacity of dry air.
    p_ref_inv: float
        1/p_ref, where p_ref is the reference pressure for the potential
        temperature

    Returns
    -------
        ndarray, dtype=float
        2D array of the discretized ratio of p_dry / p_ref
        (dry air pressure reference pressure)
        
    """
    
    return ( grid_mass_density_air_dry * grid_potential_temperature \
             * c.specific_gas_constant_air_dry * p_ref_inv )**kappa_factor

@njit()
def compute_Theta_over_T(grid_mass_density_air_dry,
                         grid_potential_temperature,
                         p_ref_inv):
    """Compute ratio of dry air pot. temperature and temperature
    
    Computes Theta_dry / T from Theta_dry and rho_dry
    
    Theta_dry = T * (p_ref/p_dry)^kappa_dry
    kappa_dry = R_dry/c_dry
    R_dry = Spec. gas const. dry air
    c_dry = spec. heat capacity dry air
    
    Parameters
    ----------
    grid_mass_density_air_dry: ndarray, dtype=float
        2D array of the discretized atmos. dry mass density field (kg/m^3)
    grid_potential_temperature: ndarray, dtype=float
        2D array of the discretized atmos. dry potential temperature
        field (K) := T (p_ref / p_dry)^(kappa_dry), where
        kappa_dry = R_dry / c_dry, where R_dry = specific gas constant
        of dry air, c_dry = specific isobaric heat capacity of dry air.
    p_ref_inv: float
        1/p_ref, where p_ref is the reference pressure for the potential
        temperature

    Returns
    -------
        ndarray, dtype=float
        2D array of the discretized ratio of Theta_dry / T
        (dry air potential temperature and temperature)
        
    """
    
    return (p_ref_inv * c.specific_gas_constant_air_dry\
            * grid_mass_density_air_dry * grid_potential_temperature )\
            **kappa_factor2
