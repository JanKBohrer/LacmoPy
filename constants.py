#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TROPOS LAGRANGIAN CLOUD MODEL
Super-Droplet method in two-dimensional kinetic framework
(Test Case 1 ICMW 2012, Muhlbauer et al. 2013)
Author: Jan Bohrer (bohrer@tropos.de)
Further contact: Oswald Knoth (knoth@tropos.de)

definitions of constants
module is imported in other modules by "import constants as c" 

basic units:
particle mass, water mass, solute mass in femto gram = 10^-18 kg
particle radius in micro meter ("mu")
all other quantities in SI units

NTP == normal temperature and pressure by NIST
T = 20 °C, p = 1.01325 x 10^5 Pa
"""

import math

#%% CONVERSIONS

# compute mass from diameter and density
one_third = 1.0 / 3.0
four_pi_over_three = 4.0 * math.pi / 3.0
four_pi_over_three_inv = 0.75 / math.pi

# volume to radius: R = ( 3 / (4 pi) )^(1/3) V^(1/3)
volume_to_radius = (four_pi_over_three_inv)**(one_third)

#%% DEFINED CONSTANTS 

### standard gravity
# NIST special publication 330 2008 Ed
# Eds.: Taylor, Thompson, National Institute of Standards and Technology 
# Gaithersburg, MD  20899 
# Therein declaration of 3rd CGPM 1901
# also consistent with ISO ISA 1975 as normal gravity
# note that Grabowski defines gravity as 9.72 m/s^2
# in vocals v3 (test case 1 ICMW 2012 fortran CODE)
earth_gravity = 9.80665 # m/s^2
a_gravity = -9.80665 # m/s^2

### densities and molar masses
mass_density_water_liquid_NTP = 998.2 # kg/m^3
mass_density_air_dry_NTP = 1.2041 # kg/m^3
# CRC 2005:
mass_density_NaCl_dry = 2163.0 # kg/m^3
# CRC 2005:
mass_density_AS_dry = 1774.0 # kg/m^3
# US Standard Atmosphere 1976, US Government. Printing
# Office, Washington DC, pp. 3 and 33, 1976. 
# page 9, below table 8, this is the mass at see level...
# also in ISO 1975 Int. Standard Atmosphere
molar_mass_air_dry = 28.9644E-3 # kg/mol
# CRC 2005
molar_mass_water = 18.015E-3 # kg/mol
# CRC 2005
molar_mass_NaCl = 58.4428E-3 # kg/mol
# https://pubchem.ncbi.nlm.nih.gov/compound/Ammonium-sulfate
molar_mass_AS = 132.14E-3 # kg/mol

### gas constants
# NIST
# https://physics.nist.gov/cgi-bin/cuu/Value?r
# note that CRC deviates significantly relative to NIST
# R_CRC = 8.314472
# So does the standard atmosphere ISO 1975:
# ISO standard atmosphere 1975
# universal_gas_constant = 8.31432 # J/(mol K)
# NIST (website s.a.):
universal_gas_constant = 8.3144598 # J/(mol K)
# R_v = R*/M_v using the NIST values...
# the exact values are
# 1. NIST
# 2. CRC
# 3. ISO
# 1. 461.5298251457119
# 2. 461.5305023591452
# 3. 461.52206494587847
specific_gas_constant_water_vapor = 461.53 # J/(kg K)
# 1. 287.0578986618055
# 2. 287.05831986852826
# 3. 287.0530720470647
specific_gas_constant_air_dry = 287.06 # J/(kg K)

### heat capacities
# ratio c_p/c_v = 1.4 at 300 K
# isochoric heat capacity of ideal gas = C_V = DOF/2 * R*,
# DOF of the gas molecules
# the heat capacity does not vary with T and p using an ideal gas
# assume 2-atomic molecules (N_2, O_2)
# C_V = 5/2 * R* (R* of NIST)
isochoric_molar_heat_capacity_air_dry_NTP = 20.7861 # J/(mol K)
# C_p = 7/2 * R* (R* of NIST)
molar_heat_capacity_air_dry_NTP = 29.1006 # J/(mol K)
# c_p = C_p / M_dry
specific_heat_capacity_air_dry_NTP = 1004.71 # J/(kg K)
# c_p / c_v = 7/5
adiabatic_index_air_dry = 1.4

# Lemmon 2015 in Lohmann 2016
specific_heat_capacity_water_vapor_20C = 1906 # J/(kg K)

# isobaric heat capacity water
# of data from Sabbah 1999, converted from molar with molar_mass_water
# average value from 0 .. 60 C from Sabbah
specific_heat_capacity_water_NTP = 4183.8 # J/(kg K)
# CRC:
#specific_heat_capacity_water_NTP = 4181.8 # J/(kg K)
                
### forces
# drag coefficient
drag_coefficient_high_Re_p = 0.44
